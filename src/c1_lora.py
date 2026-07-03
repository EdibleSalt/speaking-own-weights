"""C1. LoRA 微调 + held-out 泛化（主实验, 2026-06-07 重写版）。

教模型把"词 → 自身输入嵌入的 PCA-32 数值"读出成数字, 看是否**泛化到没教过的词**。
判据 = held-out（绝不入训练）；trained 词变好是平凡记忆（manipulation check）。

## 新增能力 (相对首版)

### A) 断点重跑（bit-exact, 2026-06-10 升级）
- 每 epoch 末 save: adapter/ep{N}/ (LoRA 权重) + _opt_state.pt (AdamW m/v)
  + _rng_state.pt (numpy + torch + torch.cuda RNG); 训完再 save adapter/final.
- 启动时自动检测最大 epoch ckpt 续训, restore AdamW state + RNG state
  → **bit-exact 续训** (与"一次到底"完全一致). `--eval_only` 直接 load final 跑 eval.
- 中途 Ctrl+C / 断电下次接着干, 不需重训.
- PCA basis / 词集 / train_target 通过 pca_targets.npz 强制复用
  (修 HANDOFF 4.2 跨 process 词集错配坑) ✓.
- 老 ckpt (附录 J 修补前存的, 无 _opt_state.pt/_rng_state.pt) 续训会 warning
  并退化为非 bit-exact; 见 §附录 J 重跑清单 (实际只 1 项 OLMo C1 main 涉及).

### B) eval 限生成（防 base 零样本啰嗦)
- `--max_new` 默认 250（PCA-32 数 × 2 位小数 × 逗号 ≈ 150 token + 100 buffer）
- 加 `StopOnNewlines` stopping criteria: 检测连续 2 个 \\n 早停（多数模型生成完吐 \\n\\n）.
- `--n_test_sanity` 可选砍半 test 词数（120→60）防 eval OOM.

### C) 4 道防污染闸（用户最关心）
- **闸 1 · 权重 sha256**: 训前对 base 4 处采样 sha256（输入嵌入 / lm_head / 第 0 块 / 最末块）;
  训后比对, 任一处不一致 → `raise SystemExit`. 这是最严格的"base 没被改"证明.
- **闸 2 · requires_grad assert**: PEFT setup 后, 遍历所有参数, base 应全 `requires_grad=False`,
  只 LoRA 参数可学; 不符合 → raise.
- **闸 3 · 永不 merge**: 脚本里没有任何 `merge_and_unload` 或 `save_pretrained(merged=True)`.
  只存 LoRA adapter, base 权重永不被改写. (代码保证)
- **闸 4 · 磁盘 mtime**: 训前后读 `models/<name>/model.safetensors` 的 mtime, 应一致.
  如果磁盘文件被改写 → raise.

## 关键设计（同 §C1 / §C1.1 / 护栏 8, 不变）
- 目标 = 该词输入嵌入的 **PCA-32**（在 train 词上拟合 PCA basis, held-out 用同 basis 投影, 无泄漏）
- LoRA 只挂注意力 / MLP 投影层; **embed_tokens 与 lm_head 冻结**（ground-truth 不漂移, 闸 2 检验）
- completion-only loss; bf16 训练（fp16 LoRA 无 GradScaler 易下溢）
- 训练词集分布 `--train_vocab {basic, random}`; 测试词 = 系统抽样, 与 train 不相交（连子串都查）
- 评估: held-out & trained 上分别算 Pearson + RSA + 置换检验
- raw/ 完整落盘 (generations_*.jsonl + pca_targets.npz)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from peft import LoraConfig, get_peft_model, PeftModel

from poc import (NUM_RE, cosine_sim_matrix, filter_single_token, set_seed, upper_tri)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

# LoRA target_modules 按架构分派——Llama-family vs GPTNeoX vs GPT-2 模块命名不同
LORA_TARGET_MODULES = {
    # Llama-family (q/k/v/o + gate/up/down): OLMo / Llama / Qwen / Mistral / Gemma / SmolLM3
    "olmo2":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "olmo":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma3":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "smollm3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # GPTNeoX (pythia): QKV 合并; FFN 是 dense_h_to_4h/dense_4h_to_h
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # GPT-2 family
    "gpt2": ["c_attn", "c_proj", "c_fc"],
}


# ============================ 防污染闸 ============================
def _hash_tensor(t):
    """tensor → 短 sha256. fp16/bf16 先转 fp32 避免精度抖动."""
    arr = t.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def _find_blocks(model):
    """容忍不同架构: model.model.layers / model.transformer.h / model.gpt_neox.layers 等."""
    for path in ("model.layers", "model.transformer.h", "transformer.h", "gpt_neox.layers"):
        cur = model
        try:
            for p in path.split("."):
                cur = getattr(cur, p)
            return cur
        except AttributeError:
            continue
    return None


def snapshot_base(model):
    """闸 1: base 4 处采样 sha256 (训前 + 训后比对). 找不到的项跳过, 不报错."""
    snap = {}
    try:
        snap["input_embed"] = _hash_tensor(model.get_input_embeddings().weight)
    except Exception as e:
        print(f"  [snap] skip input_embed: {e}")
    try:
        snap["lm_head"] = _hash_tensor(model.get_output_embeddings().weight)
    except Exception as e:
        print(f"  [snap] skip lm_head: {e}")
    blocks = _find_blocks(model)
    if blocks is not None and len(blocks) > 0:
        for tag, idx in [("block_first", 0), ("block_last", len(blocks) - 1)]:
            try:
                first_param = next(blocks[idx].parameters())
                snap[tag] = _hash_tensor(first_param)
            except Exception as e:
                print(f"  [snap] skip {tag}: {e}")
    return snap


def verify_snapshot(before, after):
    """训后调用; 任一处不一致 → raise SystemExit."""
    diffs = []
    for k in before:
        if k in after and before[k] != after[k]:
            diffs.append(f"  {k}: before={before[k]} after={after[k]}")
    if diffs:
        raise SystemExit(
            "[闸 1 FAILED] base 权重被污染!\n" + "\n".join(diffs) +
            "\n  这不应该发生——LoRA 应只更新 adapter. 检查是否误用 merge_and_unload."
        )
    print(f"  [闸 1 ok] base 权重 sha256 训前后一致 ({len(before)} 处采样)")


def assert_base_frozen(peft_model):
    """闸 2: 遍历 PEFT 模型, base 参数应全 requires_grad=False (只 LoRA 可学)."""
    not_frozen = []
    for n, p in peft_model.named_parameters():
        is_lora = ("lora_" in n) or ("modules_to_save" in n)
        if not is_lora and p.requires_grad:
            not_frozen.append(n)
    if not_frozen:
        raise SystemExit(
            f"[闸 2 FAILED] {len(not_frozen)} 个 base 参数 requires_grad=True!\n"
            f"  示例: {not_frozen[:5]}\n"
            f"  base 应完全冻结, 只 LoRA adapter 可学."
        )
    n_lora = sum(1 for n, p in peft_model.named_parameters()
                 if (("lora_" in n) or ("modules_to_save" in n)) and p.requires_grad)
    print(f"  [闸 2 ok] base 全冻结, 仅 {n_lora} 个 LoRA 参数可学")


def disk_mtime(model_path):
    """闸 4: 取磁盘 base 模型主权重文件 mtime."""
    if not os.path.isdir(model_path):
        return None, None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        f = os.path.join(model_path, fname)
        if os.path.exists(f):
            return f, os.path.getmtime(f)
    # 分片
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors") or fname.endswith(".bin"):
            return os.path.join(model_path, fname), os.path.getmtime(
                os.path.join(model_path, fname))
    return None, None


def verify_disk_mtime(model_path, before_path, before_mtime):
    if before_path is None:
        print("  [闸 4 skip] 无磁盘 model 文件可对比")
        return
    after_path, after_mtime = disk_mtime(model_path)
    if before_path != after_path:
        raise SystemExit(f"[闸 4 FAILED] {before_path} → {after_path} (文件变了)")
    if abs(before_mtime - after_mtime) > 1.0:
        raise SystemExit(
            f"[闸 4 FAILED] {before_path} mtime 变了: "
            f"before={before_mtime} after={after_mtime}\n"
            f"  base 磁盘文件不应被改写"
        )
    print(f"  [闸 4 ok] 磁盘 {os.path.basename(before_path)} mtime 训前后一致")


# ============================ 词集 ============================
def load_basic_english(path):
    """Parse Ogden Basic English 850 词表 (.md 格式).

    格式: '## A' 标题 + 逗号分隔的词 + '.' 收尾.
    特殊形态: 'be (are)' / 'grey/gray' / 'much (more, most)' / 偶尔 'coal coat' 空格连写.
    清洗: 去括号注释; 拆 '/'; 按空格再拆; 只留纯小写字母词.
    """
    import re
    text = open(path, encoding="utf-8").read()
    text = re.sub(r'^\s*##\s.*$', ' ', text, flags=re.MULTILINE)  # 去 markdown 标题
    text = re.sub(r'\([^)]*\)', ' ', text)                         # 去括号内容
    words = []
    for chunk in text.split(','):
        for variant in chunk.split('/'):       # grey/gray -> grey + gray
            for sub in variant.split():        # 'coal coat' / 末尾 '.' / 空白
                w = sub.strip('. \t\n').lower()
                if w and w.isascii() and w.isalpha() and len(w) >= 1:
                    words.append(w)
    # 去重保序
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def systematic_vocab_words(tokenizer, n_want, seed, exclude_words, extra_tokenizer=None):
    """护栏 8: 从词表系统抽单 token 干净词, 与 exclude_words 不相交 (子串也查).

    **2026-06-15 修复 bit-exact 复现**: 旧版用 `tokenizer.get_vocab().items()` iter,
    依赖 dict 插入顺序; transformers / huggingface_hub 版本变化会改 iteration order
    → 同 seed 不同库版本给不同词集 (G1 self vs G1 synonym 词集不一致 bug 根因).
    改成 sorted by (token_id, tokstr) → 跨版本稳定 + 仍 deterministic.

    extra_tokenizer: C4 用 - 跨模型 target 时, 要求词在 target 模型 tokenizer 下
    也是单 token (双向交集, 才能 fair 抽 target 嵌入矩阵的对应行).
    """
    vocab = tokenizer.get_vocab()
    cand = []
    # sorted 按 (token_id, tokstr): token_id 是规范 id, tokstr 仅 tie-break (理论无 tie)
    for tokstr, tid in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        s = tokenizer.convert_tokens_to_string([tokstr])
        if not s.startswith(" "):
            continue
        w = s[1:]
        if len(w) >= 3 and w.isascii() and w.isalpha() and w.islower():
            ids = tokenizer.encode(" " + w, add_special_tokens=False)
            if len(ids) == 1 and ids[0] == tid:
                # C4 双向交集: 要求 extra_tokenizer 下也是单 token
                if extra_tokenizer is not None:
                    extra_ids = extra_tokenizer.encode(" " + w, add_special_tokens=False)
                    if len(extra_ids) != 1:
                        continue
                cand.append((w, tid))
    exset = set(exclude_words)

    def clean(w):
        if w in exset:
            return False
        for e in exset:
            if w in e or e in w:
                return False
        return True
    cand = [(w, t) for w, t in cand if clean(w)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cand))[:n_want]
    return [cand[i] for i in sorted(idx)]


def build_train_words(tokenizer, args, extra_tokenizer=None):
    if args.train_vocab == "basic":
        bpath = os.path.join(HERE, "..", "materials",
                             "Ogden's_Basic_English_Words_List_alphabetic.md")
        if not os.path.exists(bpath):
            raise FileNotFoundError(
                f"缺 {bpath}. 未就位前可用 --train_vocab random 先跑."
            )
        raw = load_basic_english(bpath)
        words = filter_single_token(tokenizer, raw)
        if extra_tokenizer is not None:
            # C4: 双向交集, basic 词还要在 target tokenizer 下单 token
            words = [(w, t) for w, t in words
                     if len(extra_tokenizer.encode(" " + w, add_special_tokens=False)) == 1]
    elif args.train_vocab == "random":
        words = systematic_vocab_words(tokenizer, args.n_train, args.seed + 1000,
                                       exclude_words=[], extra_tokenizer=extra_tokenizer)
    elif args.train_vocab == "freq":
        raise NotImplementedError("freq-matched random 待补 (C1.1).")
    else:
        raise ValueError(args.train_vocab)
    if args.n_train and len(words) > args.n_train:
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.permutation(len(words))[:args.n_train])
        words = [words[i] for i in keep]
    return words


# ============================ PCA 目标 ============================
def fit_pca(emb_train, d):
    """train 词嵌入上拟合 PCA. target = (e - mu) @ comps.T."""
    mu = emb_train.mean(0)
    X = emb_train - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:d]


def project(emb, mu, comps):
    return (emb - mu) @ comps.T


def fmt_vec(v, decimals):
    return ", ".join(f"{x:.{decimals}f}" for x in v)


# ============================ prompt / 数据 ============================
def build_prompt(tokenizer, word, d):
    content = (f"Read out the internal embedding vector of a word as {d} numbers, "
               f"comma-separated, 2 decimals each. Word: {word}")
    if getattr(tokenizer, "chat_template", None):
        msg = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    # base 模型 fallback (pythia / gpt2 等无 chat_template): 朴素 Q/A 模板
    return f"Q: {content}\nA: "


# C3 用: 抽 transformer 第 L 层末位 hidden state 时的 prompt 模板.
# 与 C0-base prompt 表对齐 (见 §C0-base 表), 不挂训练目标——仅决定 hidden 抽取时上下文.
def build_target_prompt(tokenizer, word, key):
    if key == "P4":
        return f" {word}"
    if key == "P1":
        return f"Word: {word}\nEmbedding:"
    if key == "P3":
        # 与 probe_ceiling.py P3 一致 (C0-base 同源, 5 个 0.00 + 词前空格防泄漏)
        dummy = ",".join(["0.00"] * 5)
        return f"dog -> {dummy}\nfish -> {dummy}\n {word} ->"
    if key == "chat":
        # 与 build_prompt 一致, 复用 (会触发训练 prompt confound, 见 §C3 prompt 表)
        return build_prompt(tokenizer, word, 32)
    raise ValueError(f"unknown target_prompt key: {key}")


@torch.no_grad()
def extract_hidden_targets(model, tokenizer, device, word_pairs, target_layer, target_prompt):
    """C3 用: 对每个 word 跑一次 base forward, 拿 hidden_states[L][0, -1, :] (末位 hidden).

    返回 (n_words, hidden_dim) numpy array, 行序与 word_pairs 一致.
    target_layer: HF hidden_states tuple index (0=embedding output=输入嵌入sanity,
                  N>0=第 N 个 transformer block 后; -1=末层快捷).
    """
    model.eval()
    out_vecs = []
    L_max = model.config.num_hidden_layers
    L_use = L_max if target_layer == -1 else target_layer
    if not (0 <= L_use <= L_max):
        raise SystemExit(f"--target_layer {target_layer} 越界: model {L_max} 层, tuple [0, {L_max}]")
    print(f"  [C3] 抽 hidden: target_layer={L_use}/{L_max} target_prompt={target_prompt}")
    for w, _tid in word_pairs:
        text = build_target_prompt(tokenizer, w, target_prompt)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        # hidden_states: tuple of (L_max+1) tensors, each (1, seq, hidden_dim)
        h = out.hidden_states[L_use][0, -1, :].detach().float().cpu().numpy()
        out_vecs.append(h)
    return np.stack(out_vecs)


def make_example(tokenizer, word, target_vec, d, decimals, max_len):
    prompt = build_prompt(tokenizer, word, d)
    completion = fmt_vec(target_vec, decimals) + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


# ============================ eval (限生成) ============================
class StopOnDoubleNewline(StoppingCriteria):
    """连续 2 个 \\n 即停. 用于 base 零样本啰嗦时早结束."""
    def __init__(self, tokenizer):
        nl = tokenizer.encode("\n", add_special_tokens=False)
        self.nl_id = nl[-1] if nl else None

    def __call__(self, input_ids, scores, **kwargs):
        if self.nl_id is None or input_ids.shape[1] < 2:
            return False
        last2 = input_ids[:, -2:]
        # 整 batch 都触发才停 (避免一个早停拖累整 batch)
        return bool((last2 == self.nl_id).all(dim=-1).all())


@torch.no_grad()
def readout(model, tokenizer, device, words, d, decimals, max_new, batch_size,
            query_for_word=None):
    """生成读出. 加 stopping_criteria 监 \\n\\n 早停, 节省 base 零样本时间.

    query_for_word: G1 用. dict[w → query_str]. 默认 None = build_prompt 直接用 w
    (C1/C2/C3/C4 行为). 非 None = prompt 里把 w 替换成 query_str, 但 out 字典 key
    仍是 w (ground-truth 仍按原词 PCA-32 比较). 用 query=synonym/definition 测试模型
    在 prompt 不直接含目标 token 时能否仍输出该词嵌入 → 切断激活路径判 introspection
    vs activation translation. 见 [[G族_判决性实验#G1]].
    """
    model.eval()
    def _prompt_for(w):
        q = query_for_word[w] if query_for_word is not None and w in query_for_word else w
        return build_prompt(tokenizer, q, d)
    prompts = [_prompt_for(w) for w, _ in words]
    sc = StoppingCriteriaList([StopOnDoubleNewline(tokenizer)])
    out, raw = {}, []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b:b + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             stopping_criteria=sc)
        texts = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                       skip_special_tokens=True)
        for (w, tid), txt in zip(words[b:b + batch_size], texts):
            nums = [float(x) for x in NUM_RE.findall(txt)]
            v = np.array(nums[:d]) if len(nums) >= d else None
            out[w] = v
            q = query_for_word[w] if query_for_word is not None and w in query_for_word else w
            raw.append({"word": w, "token_id": int(tid), "query": q,
                        "raw_text": txt,
                        "parsed": (v.tolist() if v is not None else None)})
    return out, raw


def load_readout_jsonl(path):
    """从 raw/gen_*.jsonl 还原 readout 字典 (word → parsed vector or None).
    断点重跑 skip 路径用——从已有 raw 重算 metrics, 避免漏写 summary 字段."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = np.array(r["parsed"]) if r["parsed"] is not None else None
            out[r["word"]] = v
    return out


def eval_metrics(readout_dict, gt_dict, words, n_perm=2000, seed=0):
    kept = [w for w, _ in words if readout_dict.get(w) is not None]
    if len(kept) < 5:
        return {"n_kept": len(kept), "parse_rate": len(kept) / max(1, len(words))}
    P = np.stack([readout_dict[w] for w in kept])
    G = np.stack([gt_dict[w] for w in kept])
    # G2 退化: target_metric=l2norm 时 d=1 → 单 float per word.
    # 改报跨词 Pearson r + Spearman r (不算 RDM, 因 d=1 时 cosine 退化).
    # 字段名也叫 rsa/rsa_p 以让 run_pending check_done_c1 / 摘要统一; 但实际是 scalar
    # cross-word correlation, 见 results["target_metric"] 字段区分.
    if P.shape[1] == 1:
        Pv = P[:, 0].astype(np.float64)
        Gv = G[:, 0].astype(np.float64)
        pr = float(pearsonr(Pv, Gv)[0])
        sr = float(spearmanr(Pv, Gv)[0])
        rng = np.random.default_rng(seed)
        null = np.array([pearsonr(Pv, Gv[rng.permutation(len(Gv))])[0]
                          for _ in range(n_perm)])
        p = (np.sum(np.abs(null) >= abs(pr)) + 1) / (n_perm + 1)
        return {"n_kept": len(kept), "parse_rate": len(kept) / len(words),
                "scalar_pearson_r": pr, "scalar_spearman_r": sr,
                "rsa": pr, "rsa_p": float(p), "null": null,
                "per_word_pearson_mean": pr}
    per_word = np.array([pearsonr(P[i], G[i])[0] for i in range(len(kept))])
    rp = upper_tri(cosine_sim_matrix(P))
    rt = upper_tri(cosine_sim_matrix(G))
    rsa = spearmanr(rp, rt)[0]
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(upper_tri(cosine_sim_matrix(P[rng.permutation(len(kept))])),
                               rt)[0] for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(rsa)) + 1) / (n_perm + 1)
    return {"n_kept": len(kept), "parse_rate": len(kept) / len(words),
            "per_word_pearson_mean": float(np.nanmean(per_word)),
            "rsa": float(rsa), "rsa_p": float(p), "null": null}


# ============================ 训练 + checkpoint ============================
def find_latest_ckpt(adapter_dir):
    """从 adapter_dir 找最大 epoch 的 ep{N} 目录; 返回 (path, N) 或 (None, 0)."""
    if not os.path.isdir(adapter_dir):
        return None, 0
    eps = []
    for name in os.listdir(adapter_dir):
        if name.startswith("ep") and os.path.isdir(os.path.join(adapter_dir, name)):
            try:
                eps.append(int(name[2:]))
            except ValueError:
                pass
    if not eps:
        return None, 0
    n = max(eps)
    return os.path.join(adapter_dir, f"ep{n}"), n


def train_lora(peft_model, tokenizer, device, examples, args, adapter_dir,
               start_ep=0, resume_state=None):
    """每 epoch 末 save adapter + opt/RNG state (bit-exact 断点重跑); 训完 save final.

    resume_state: 续训时 main 传入的 dict {'opt': sd, 'np_rng': s, 'torch_rng': t,
    'cuda_rng': c or None}; 用来 restore AdamW 动量 + numpy/torch/cuda RNG, 实现
    bit-exact 续训 (见 §附录 J).
    """
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=args.lr)
    pad_id = tokenizer.pad_token_id
    # 单一 rng 流: 启动时 seed=args.seed+1000, batch permutation 序列由 ep0 至 ep14 一气呵成
    rng = np.random.default_rng(args.seed + 1000)
    if resume_state is not None:
        opt.load_state_dict(resume_state["opt"])
        rng.bit_generator.state = resume_state["np_rng"]
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(resume_state["cuda_rng"])
        print(f"  [resume] AdamW state + numpy/torch/cuda RNG 已 restore, "
              f"续训 bit-exact (附录 J)")
    peft_model.train()
    for ep in range(start_ep, args.epochs):
        order = rng.permutation(len(examples))
        tot, nb = 0.0, 0
        for b in range(0, len(order), args.train_bs):
            batch = [examples[i] for i in order[b:b + args.train_bs]]
            input_ids, labels, attn = collate(batch, pad_id)
            input_ids, labels, attn = (input_ids.to(device), labels.to(device),
                                       attn.to(device))
            out = peft_model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
            nb += 1
        avg_loss = tot / max(1, nb)
        print(f"  [train] epoch {ep+1}/{args.epochs}  loss={avg_loss:.4f}", flush=True)
        ckpt = os.path.join(adapter_dir, f"ep{ep+1}")
        peft_model.save_pretrained(ckpt)
        with open(os.path.join(ckpt, "_train_loss.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": ep + 1, "loss": avg_loss}, f)
        # bit-exact resume: save AdamW state + numpy/torch/cuda RNG state
        torch.save(opt.state_dict(), os.path.join(ckpt, "_opt_state.pt"))
        torch.save({
            "np_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, os.path.join(ckpt, "_rng_state.pt"))
    final = os.path.join(adapter_dir, "final")
    peft_model.save_pretrained(final)
    print(f"  [save] final adapter -> {final}")


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_vocab", default="random", choices=["basic", "random", "freq"])
    ap.add_argument("--target_mode", default="real", choices=["real", "random"],
                    help="real: target = PCA-32(input_embed or lm_head, 看 --target_source). "
                         "random: target = deterministic random Gaussian 32d/词 (control 组——"
                         "测'仅会按指令吐 32 数'本身能达多少 RSA, 若 held-out ≈ 0 则 real 真信号成立)")
    ap.add_argument("--target_source", default="input_embed",
                    choices=["input_embed", "lm_head", "hidden"],
                    help="real target 用哪个矩阵的行做 PCA basis. "
                         "input_embed: C1 默认 (词在 layer 0 残差流就有它, 测 (ii) 浅层路由). "
                         "lm_head: **C2 难目标** (要求 untied 模型; 词的 unembedding 行不在前向中直接存在, "
                         "模型必须深层读出, 测 (iii) 真自我读出 - 论文级强 disambiguation). "
                         "hidden: **C3 真深层 hidden** (transformer 第 L 层后末位 hidden, "
                         "升级 (iii) 到中间层, 需 --target_layer N + --target_prompt key).")
    ap.add_argument("--target_layer", type=int, default=-1,
                    help="C3 用: hidden_states tuple index. 0=输入嵌入 sanity, "
                         "N>0=第 N 个 transformer block 后, -1=末层. 仅 target_source=hidden 用.")
    ap.add_argument("--target_prompt", default="P4",
                    choices=["P1", "P3", "P4", "chat"],
                    help="C3 用: 抽 hidden 时的 prompt context. P4=' {w}' 最干净, "
                         "P1/P3 经注意力路由, chat 与训练 prompt 同分布(潜在 confound). "
                         "仅 target_source=hidden 用.")
    ap.add_argument("--target_model", default=None,
                    help="C4 用: 跨模型 target. None=同 source (默认), 路径=用该模型的嵌入矩阵 "
                         "作 PCA basis. 验 H3 (FT 学的是通用语义→向量回归, 不是读自己权重). "
                         "词集双向交集 (source+target tokenizer 都单 token). "
                         "与 --target_source hidden 不兼容 (hidden 跟模型绑定).")
    ap.add_argument("--intersect_tokenizer", default=None,
                    help="C4-control 用: 强制走双向交集词集, 跟 target 是否 cross 解耦. "
                         "None (默认) = 行为跟 --target_model 联动 (有 target_model 自动走 target_tokenizer "
                         "交集; 无 target_model 走单 tokenizer). 传路径 = 强制用该模型的 tokenizer 当 "
                         "intersect, 即使 target_model=None. 用于跑 'self target + 双向交集词集' 的 "
                         "C4 control, 排除 C4 cross-model 的词集 confound (见 §C4 control 段).")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--n_test_sanity", type=int, default=None,
                    help="test 词砍到这么多 (None=用 n_test). 防 eval OOM 的 sanity")
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_new", type=int, default=250,
                    help="eval 生成上限 (默认 250, 从首版 400 减; 加 stopping_criteria 早停)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--eval_only", action="store_true",
                    help="检测 adapter/final, load 跑 eval (不重训). 无 final 报错")
    ap.add_argument("--outdir", default="results_c1")
    # G1: 非激活查询. 训练不变, eval 用 query_dict 替换 prompt 里的词
    ap.add_argument("--eval_query_mode", default="self",
                    choices=["self", "synonym", "definition"],
                    help="G1 非激活查询. self=baseline(原行为 prompt 含目标 token), "
                         "synonym/definition=用 query_dict 替换 prompt 里的词 (训练阶段不变, "
                         "仅 base zero-shot + FT held-out eval 阶段替换). "
                         "评估 ground-truth 仍按原词 PCA-32. 见 [[G族_判决性实验#G1]].")
    ap.add_argument("--query_dict_path", default=None,
                    help="G1 用. JSON 文件 {word: query_str}. eval_query_mode != self 必须给.")
    # G2: 物理目标. target = 单 float L2 范数, 切断"词→嵌入向量"语义同构
    ap.add_argument("--target_metric", default="pca",
                    choices=["pca", "l2norm", "pca_recon_err", "tokenid_binary"],
                    help="pca=默认 (target = PCA-32 of input_embed/lm_head/hidden), "
                         "l2norm=G2 (i) (target = ||emb(w)||_2 单 float, 测物理参数访问), "
                         "pca_recon_err=G2 (ii) (target = ||emb - PCA-8(emb)|| 单 float, "
                         "测主成分外残差), "
                         "tokenid_binary=G2 (iii) (target = token_id 低 16 位 binary 16 floats, "
                         "完全无语义, 关键 disjudicator). 见 [[G族_判决性实验#G2]].")
    ap.add_argument("--recon_k", type=int, default=8,
                    help="target_metric=pca_recon_err 用的 PCA 维度. 算 ||emb - PCA-k(emb)||")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rawdir = os.path.join(args.outdir, "raw")
    adapter_dir = os.path.join(args.outdir, "adapter")
    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    print(f"[env] device={device} model={args.model} train_vocab={args.train_vocab}")
    print(f"[env] outdir={args.outdir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # C4 跨模型 target: load target tokenizer (用于双向交集词集) - target model 延后到 emb 抽取段 load
    target_tokenizer = None
    if args.target_model is not None:
        if args.target_source == "hidden":
            raise SystemExit("--target_model 与 --target_source hidden 不兼容 (hidden 跟模型绑定).")
        print(f"[C4] target_model = {args.target_model}")
        target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    # G1: 非激活查询. 读 query_dict {word: query_str}. eval 阶段用 query 替换 prompt 里的词
    query_for_word = None
    if args.eval_query_mode != "self":
        if not args.query_dict_path or not os.path.exists(args.query_dict_path):
            raise SystemExit(f"--eval_query_mode={args.eval_query_mode} 需要 "
                             f"--query_dict_path 指向已存在的 JSON")
        with open(args.query_dict_path, encoding="utf-8") as f:
            query_for_word = json.load(f)
        print(f"[G1] eval_query_mode={args.eval_query_mode}  "
              f"query_dict_path={args.query_dict_path}  覆盖 {len(query_for_word)} 词")

    # C4-control: 用 --intersect_tokenizer 解耦 "词集交集 tokenizer" 跟 "target 嵌入来源".
    # intersect_tokenizer 用于 systematic_vocab_words / build_train_words 的 extra_tokenizer.
    # 优先级: --intersect_tokenizer (显式) > target_tokenizer (跨模型时自动) > None.
    intersect_tokenizer = target_tokenizer
    if args.intersect_tokenizer is not None:
        print(f"[C4-control] intersect_tokenizer = {args.intersect_tokenizer} (词集双向交集, "
              f"target_model={args.target_model})")
        intersect_tokenizer = AutoTokenizer.from_pretrained(args.intersect_tokenizer)

    # 闸 4 (前): 读磁盘 model 文件 mtime
    before_disk_path, before_disk_mtime = disk_mtime(args.model)
    if before_disk_path:
        print(f"  [闸 4 snap] {before_disk_path} mtime={before_disk_mtime}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
    model = model.to(device)

    # 闸 1 (前): snapshot base 权重 sha256
    print("[闸 1] base 权重 snapshot (训前):")
    snap_before = snapshot_base(model)
    for k, v in snap_before.items():
        print(f"  {k}: {v}")

    # 词集 + 真实 PCA target (always) + 训练 target (按 target_mode)
    # 已知坑: tokenizer.get_vocab() 跨 Python process iteration order 不一致 ->
    # 断点重跑必须复用首次跑的 pca_targets.npz, 否则 adapter (旧词集训练) 跟新 PCA basis 不对齐.
    #
    # 评估设计 (区分真信号 vs 模板能力):
    #   - heldout eval target = gt_test_real (永远用 real, 跨 target_mode fair compare)
    #   - trained eval target = train_target_actual (跟训练一致, 测记忆能力)
    pca_file = os.path.join(rawdir, "pca_targets.npz")
    if os.path.exists(pca_file):
        print(f"[resume] 从 {pca_file} 加载词集 + PCA basis + 训练 target")
        pca_old = np.load(pca_file, allow_pickle=True)
        train_words_str = [str(w) for w in pca_old['train_words']]
        test_words_str = [str(w) for w in pca_old['test_words']]

        def _to_pairs(strs):
            out = []
            for w in strs:
                ids = tokenizer.encode(" " + w, add_special_tokens=False)
                out.append((w, ids[0] if len(ids) == 1 else -1))
            return out

        train_words = _to_pairs(train_words_str)
        test_words = _to_pairs(test_words_str)
        mu = pca_old['mu']
        comps = pca_old['comps']
        # backward compat: 老 npz 字段 gt_train/gt_test 等价于 gt_train_real/gt_test_real
        if 'gt_train_real' in pca_old.files:
            gt_train_real = {str(w): v for w, v in zip(pca_old['train_words'], pca_old['gt_train_real'])}
            gt_test_real  = {str(w): v for w, v in zip(pca_old['test_words'],  pca_old['gt_test_real'])}
            train_target  = {str(w): v for w, v in zip(pca_old['train_words'], pca_old['train_target_actual'])}
            saved_mode = str(pca_old['target_mode'])
            if saved_mode != args.target_mode:
                raise SystemExit(
                    f"[fatal] outdir 已存 pca_targets.npz with target_mode='{saved_mode}', "
                    f"但 --target_mode='{args.target_mode}'. 换 outdir 或 --target_mode."
                )
            # target_source 校验 (C2 加, backward compat: 老 npz 无字段 = input_embed)
            saved_source = str(pca_old['target_source']) if 'target_source' in pca_old.files else 'input_embed'
            if saved_source != args.target_source:
                raise SystemExit(
                    f"[fatal] outdir 已存 pca_targets.npz with target_source='{saved_source}', "
                    f"但 --target_source='{args.target_source}'. 换 outdir."
                )
            # target_model 校验 (C4 加, backward compat: 老 npz 无字段 = None)
            saved_tgt_model = str(pca_old['target_model']) if 'target_model' in pca_old.files else ""
            cur_tgt_model = args.target_model if args.target_model else ""
            if saved_tgt_model != cur_tgt_model:
                raise SystemExit(
                    f"[fatal] outdir 已存 pca_targets.npz with target_model='{saved_tgt_model}', "
                    f"但 --target_model='{cur_tgt_model}'. 换 outdir."
                )
            # target_layer / target_prompt 校验 (C3 加, backward compat: 老 npz 无 = sentinel)
            if args.target_source == "hidden":
                saved_layer = int(pca_old['target_layer']) if 'target_layer' in pca_old.files else None
                saved_prompt = str(pca_old['target_prompt']) if 'target_prompt' in pca_old.files else None
                if saved_layer is None or saved_prompt is None:
                    raise SystemExit(
                        "[fatal] outdir 是旧版 pca_targets.npz (无 target_layer/target_prompt 字段), "
                        "但 --target_source=hidden 需要这两字段. 换 outdir.")
                if saved_layer != args.target_layer:
                    raise SystemExit(
                        f"[fatal] outdir 已存 pca_targets.npz with target_layer={saved_layer}, "
                        f"但 --target_layer={args.target_layer}. 换 outdir.")
                if saved_prompt != args.target_prompt:
                    raise SystemExit(
                        f"[fatal] outdir 已存 pca_targets.npz with target_prompt='{saved_prompt}', "
                        f"但 --target_prompt='{args.target_prompt}'. 换 outdir.")
        else:
            # legacy: 老 c1 跑只存了 gt_train/gt_test (real); target_mode='real'
            gt_train_real = {str(w): v for w, v in zip(pca_old['train_words'], pca_old['gt_train'])}
            gt_test_real  = {str(w): v for w, v in zip(pca_old['test_words'],  pca_old['gt_test'])}
            train_target  = dict(gt_train_real)  # legacy 一定是 real
            if args.target_mode != "real":
                raise SystemExit(
                    f"[fatal] outdir 是旧版 pca_targets.npz (无 target_mode 字段, 隐含 real), "
                    f"但 --target_mode='{args.target_mode}'. 换 outdir."
                )
    else:
        train_words = build_train_words(tokenizer, args, extra_tokenizer=intersect_tokenizer)
        n_test_use = args.n_test_sanity if args.n_test_sanity else args.n_test
        test_words = systematic_vocab_words(tokenizer, n_test_use, args.seed,
                                            exclude_words=[w for w, _ in train_words],
                                            extra_tokenizer=intersect_tokenizer)
        # target source: C1 用 input_embed, C2 用 lm_head, C3 用 hidden (C3 跟 target_model 不兼容,
        # 已在 argparse 段 raise). C4 = input_embed/lm_head + target_model != None.
        if args.target_source in ("input_embed", "lm_head"):
            if args.target_source == "lm_head":
                tgt_check_model = args.target_model if args.target_model else args.model
                # untied 校验对实际取嵌入的 model
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(tgt_check_model)
                if getattr(cfg, "tie_word_embeddings", True):
                    raise SystemExit(
                        f"--target_source lm_head 要求 untied (tie_word_embeddings=False), "
                        f"但 target 模型 {tgt_check_model} 是 tied.")
            if args.target_model is None:
                # 同模型 target (C1/C2/C2-multiseed)
                if args.target_source == "input_embed":
                    emb_mat = model.get_input_embeddings().weight
                else:
                    emb_mat = model.get_output_embeddings().weight
                    print(f"  [C2] target_source=lm_head (untied, lm_head ≠ input_embed)")
                emb = emb_mat.detach().float().cpu().numpy()
                emb_tr = np.stack([emb[t] for _, t in train_words])
                emb_te = np.stack([emb[t] for _, t in test_words])
            else:
                # C4: 跨模型 target. Load target model 拿 emb mat, 用 target tokenizer 找 token_id
                print(f"  [C4] 跨模型 target: load {args.target_model} 抽 {args.target_source}")
                tgt_model = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=dt)
                with torch.no_grad():
                    if args.target_source == "input_embed":
                        emb_mat_t = tgt_model.get_input_embeddings().weight
                    else:
                        emb_mat_t = tgt_model.get_output_embeddings().weight
                    emb = emb_mat_t.detach().float().cpu().numpy()
                # words 在 target tokenizer 下找 token_id (双向交集已保证单 token)
                def _tgt_id(w):
                    return target_tokenizer.encode(" " + w, add_special_tokens=False)[0]
                emb_tr = np.stack([emb[_tgt_id(w)] for w, _ in train_words])
                emb_te = np.stack([emb[_tgt_id(w)] for w, _ in test_words])
                del tgt_model  # 释放显存, 仅用于抽 PCA basis 输入
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:  # hidden (C3)
            # 跑 base forward 一遍, 拿 hidden_states[L][-1] 当 PCA basis 输入
            # 注意: 这里跟 token_id 无关(target_prompt 决定 context), 但 train/test 词不同 prompt 内容也不同
            emb_tr = extract_hidden_targets(model, tokenizer, device, train_words,
                                            args.target_layer, args.target_prompt)
            emb_te = extract_hidden_targets(model, tokenizer, device, test_words,
                                            args.target_layer, args.target_prompt)
        # G2: target_metric 切断"词→语义向量"同构, 测物理参数访问.
        if args.target_metric == "l2norm":
            # G2 (i): target = ||emb(w)||_2 单 float.
            if args.pca_dim != 1:
                print(f"[G2] target_metric=l2norm 强制 pca_dim={args.pca_dim} -> 1")
                args.pca_dim = 1
            print(f"  [G2] emb -> L2 norm 单 float (range: tr {np.linalg.norm(emb_tr, axis=1).mean():.3f} ± "
                  f"{np.linalg.norm(emb_tr, axis=1).std():.3f}; te {np.linalg.norm(emb_te, axis=1).mean():.3f} ± "
                  f"{np.linalg.norm(emb_te, axis=1).std():.3f})")
            emb_tr = np.linalg.norm(emb_tr, axis=1, keepdims=True).astype(np.float32)
            emb_te = np.linalg.norm(emb_te, axis=1, keepdims=True).astype(np.float32)
            mu = np.array([0.0], dtype=np.float32)
            comps = np.array([[1.0]], dtype=np.float32)
        elif args.target_metric == "pca_recon_err":
            # G2 (ii): target = ||emb - PCA-k(emb)|| 单 float, 测"嵌入向量在主成分外的残差长度".
            # basis 仅 train 拟合 (no leakage); 重构 = (e-mu) @ V.T @ V + mu; err = ||e - recon||.
            if args.pca_dim != 1:
                print(f"[G2] target_metric=pca_recon_err 强制 pca_dim={args.pca_dim} -> 1")
                args.pca_dim = 1
            k = args.recon_k
            mu_k, V_k = fit_pca(emb_tr, k)
            def _recon_err(emb):
                # (e - mu) @ V.T 是 k 维投影坐标, 再 @ V 投回 d 维, + mu
                proj = (emb - mu_k) @ V_k.T
                recon = proj @ V_k + mu_k
                return np.linalg.norm(emb - recon, axis=1, keepdims=True).astype(np.float32)
            err_tr = _recon_err(emb_tr)
            err_te = _recon_err(emb_te)
            print(f"  [G2] emb -> PCA-{k} recon err 单 float (tr {err_tr.mean():.3f} ± {err_tr.std():.3f}; "
                  f"te {err_te.mean():.3f} ± {err_te.std():.3f})")
            emb_tr, emb_te = err_tr, err_te
            mu = np.array([0.0], dtype=np.float32)
            comps = np.array([[1.0]], dtype=np.float32)
        elif args.target_metric == "tokenid_binary":
            # G2 (iii): target = token_id 低 16 位 binary 16 floats, 完全无语义.
            # 关键 disjudicator: 若 HE ≈ 0 → 物理参数确实不可访问, 漏洞 3 实锤.
            n_bits = 16
            if args.pca_dim != n_bits:
                print(f"[G2] target_metric=tokenid_binary 强制 pca_dim={args.pca_dim} -> {n_bits}")
                args.pca_dim = n_bits
            def _bits(token_ids):
                out = np.zeros((len(token_ids), n_bits), dtype=np.float32)
                for i, tid in enumerate(token_ids):
                    for b in range(n_bits):
                        out[i, b] = float((int(tid) >> b) & 1)
                return out
            train_tids = [t for _, t in train_words]
            test_tids = [t for _, t in test_words]
            emb_tr = _bits(train_tids)
            emb_te = _bits(test_tids)
            print(f"  [G2] token_id -> 低 {n_bits} 位 binary (tr 平均 1-bit 数 {emb_tr.sum(1).mean():.2f}; "
                  f"te {emb_te.sum(1).mean():.2f})")
            mu = np.zeros(n_bits, dtype=np.float32)
            comps = np.eye(n_bits, dtype=np.float32)  # identity, project 退化为恒等
        else:
            mu, comps = fit_pca(emb_tr, args.pca_dim)
        gt_train_real = {w: project(emb_tr[i], mu, comps)
                         for i, (w, _) in enumerate(train_words)}
        gt_test_real  = {w: project(emb_te[i], mu, comps)
                         for i, (w, _) in enumerate(test_words)}
        # 训练 target: 按 mode 选
        if args.target_mode == "real":
            train_target = dict(gt_train_real)
        elif args.target_mode == "random":
            # deterministic random Gaussian per word; 量级匹配 real PCA basis 的 std
            rng_t = np.random.default_rng(args.seed + 9999)
            rand_mat = rng_t.standard_normal(size=(len(train_words), args.pca_dim)).astype(np.float32)
            real_std = float(np.std(np.stack([gt_train_real[w] for w, _ in train_words])))
            rand_mat *= real_std
            train_target = {w: rand_mat[i] for i, (w, _) in enumerate(train_words)}
        else:
            raise ValueError(args.target_mode)
        np.savez(pca_file, mu=mu, comps=comps,
                 target_mode=np.array(args.target_mode),
                 target_source=np.array(args.target_source),
                 target_layer=np.array(args.target_layer),
                 target_prompt=np.array(args.target_prompt),
                 target_model=np.array(args.target_model if args.target_model else ""),
                 train_words=np.array([w for w, _ in train_words]),
                 test_words=np.array([w for w, _ in test_words]),
                 gt_train_real=np.stack([gt_train_real[w] for w, _ in train_words]),
                 gt_test_real=np.stack([gt_test_real[w] for w, _ in test_words]),
                 train_target_actual=np.stack([train_target[w] for w, _ in train_words]))
    print(f"[words] train={len(train_words)} test(held-out)={len(test_words)} "
          f"(disjoint, 子串已查)  target_mode={args.target_mode}"
          + (f" target_model={args.target_model}" if args.target_model else ""))

    results = {"model": args.model, "train_vocab": args.train_vocab,
               "target_mode": args.target_mode, "target_source": args.target_source,
               "target_layer": int(args.target_layer) if args.target_source == "hidden" else None,
               "target_prompt": args.target_prompt if args.target_source == "hidden" else None,
               "target_model": args.target_model,  # C4: None=同 source, str=跨模型 target
               "target_metric": args.target_metric,  # G2: pca | l2norm
               "eval_query_mode": args.eval_query_mode,  # G1: self | synonym | definition
               "query_dict_path": args.query_dict_path,
               "seed": args.seed,
               "pca_dim": args.pca_dim, "n_train": len(train_words),
               "n_test": len(test_words), "max_new": args.max_new,
               "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None)}

    # ---- base 零样本 eval (一次; 如果已有跳过) ----
    base_gen_file = os.path.join(rawdir, "gen_base_heldout.jsonl")
    if not os.path.exists(base_gen_file):
        print("\n[eval] base (zero-shot) held-out ...", flush=True)
        ro_te, raw_te = readout(model, tokenizer, device, test_words, args.pca_dim,
                                args.decimals, args.max_new, args.eval_bs,
                                query_for_word=query_for_word)
        m_base_te = eval_metrics(ro_te, gt_test_real, test_words, seed=args.seed)
        with open(base_gen_file, "w", encoding="utf-8") as f:
            for r in raw_te:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results["base_heldout"] = {k: v for k, v in m_base_te.items() if k != "null"}
        print(f"  base held-out: parse={m_base_te.get('parse_rate'):.2f} "
              f"pearson={m_base_te.get('per_word_pearson_mean')} "
              f"rsa={m_base_te.get('rsa')} p={m_base_te.get('rsa_p')}")
    else:
        print(f"\n[eval] base held-out 已跑过 ({base_gen_file}), 从 raw 重算 metrics 补 results")
        ro_te = load_readout_jsonl(base_gen_file)
        m_base_te = eval_metrics(ro_te, gt_test_real, test_words, seed=args.seed)
        results["base_heldout"] = {k: v for k, v in m_base_te.items() if k != "null"}
        print(f"  base held-out: parse={m_base_te.get('parse_rate'):.2f} "
              f"pearson={m_base_te.get('per_word_pearson_mean')} "
              f"rsa={m_base_te.get('rsa')} p={m_base_te.get('rsa_p')}")

    # ---- LoRA setup / 训练 / 续训 / eval-only ----
    final_adapter = os.path.join(adapter_dir, "final")

    if args.eval_only:
        if not os.path.exists(final_adapter):
            raise SystemExit(f"--eval_only 但 {final_adapter} 不存在; 先训练")
        print(f"\n[eval_only] load final adapter <- {final_adapter}")
        peft_model = PeftModel.from_pretrained(model, final_adapter)
    else:
        latest_ckpt, latest_ep = find_latest_ckpt(adapter_dir)
        if os.path.exists(final_adapter):
            print(f"\n[train] final adapter 已存在, 复用; 跳过训练")
            peft_model = PeftModel.from_pretrained(model, final_adapter,
                                                   is_trainable=False)
        elif latest_ep >= args.epochs:
            print(f"\n[train] 最新 ckpt ep{latest_ep} >= epochs={args.epochs}, "
                  f"标 final 并跳过训练")
            peft_model = PeftModel.from_pretrained(model, latest_ckpt,
                                                   is_trainable=False)
            peft_model.save_pretrained(final_adapter)
        elif latest_ckpt is not None:
            print(f"\n[train] 从 {latest_ckpt} 续训 (epoch {latest_ep+1}..{args.epochs})")
            peft_model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
            assert_base_frozen(peft_model)
            examples = [make_example(tokenizer, w, train_target[w], args.pca_dim,
                                     args.decimals, args.max_len) for w, _ in train_words]
            # bit-exact resume: load AdamW + RNG state (附录 J 加, 老 ckpt 无则 warning)
            resume_state = None
            opt_path = os.path.join(latest_ckpt, "_opt_state.pt")
            rng_path = os.path.join(latest_ckpt, "_rng_state.pt")
            if os.path.exists(opt_path) and os.path.exists(rng_path):
                resume_state = {"opt": torch.load(opt_path, map_location=device, weights_only=False)}
                resume_state.update(torch.load(rng_path, weights_only=False))
                print(f"  [resume] _opt_state.pt + _rng_state.pt 已加载 (bit-exact 续训)")
            else:
                print(f"  [resume] WARNING: {latest_ckpt} 无 _opt_state.pt/_rng_state.pt, "
                      f"续训非 bit-exact (老 ckpt 在加附录 J 修补前存的). 见 §附录 J 重跑清单.")
            train_lora(peft_model, tokenizer, device, examples, args, adapter_dir,
                       start_ep=latest_ep, resume_state=resume_state)
        else:
            print("\n[train] 从零开始 fresh LoRA")
            mtype = getattr(model.config, "model_type", None)
            if mtype not in LORA_TARGET_MODULES:
                raise SystemExit(
                    f"未知 model_type='{mtype}'. 加到 LORA_TARGET_MODULES 字典再跑."
                )
            target_modules = LORA_TARGET_MODULES[mtype]
            print(f"  [LoRA] model_type={mtype}, target_modules={target_modules}")
            lcfg = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules)
            peft_model = get_peft_model(model, lcfg)
            print("[闸 2] 验证 base 完全冻结:")
            assert_base_frozen(peft_model)
            n_train_p = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            emb_grad = peft_model.get_input_embeddings().weight.requires_grad
            print(f"  trainable params={n_train_p:,}  embed.requires_grad={emb_grad}")
            results["lora_trainable_params"] = int(n_train_p)
            results["embed_frozen"] = (not emb_grad)
            examples = [make_example(tokenizer, w, train_target[w], args.pca_dim,
                                     args.decimals, args.max_len) for w, _ in train_words]
            print(f"[train] {len(examples)} examples, {args.epochs} ep, lr={args.lr}",
                  flush=True)
            train_lora(peft_model, tokenizer, device, examples, args, adapter_dir,
                       start_ep=0)

    # 闸 1 (后) + 闸 4 (后): 验证 base 未污染
    # 注: PEFT model 内部 base 还是同一个对象, snapshot_base 看的就是原 base
    print("\n[闸 1] base 权重 snapshot (训后):")
    snap_after = snapshot_base(model)
    verify_snapshot(snap_before, snap_after)
    verify_disk_mtime(args.model, before_disk_path, before_disk_mtime)

    # ---- FT eval: held-out (判据) + trained (memorization check) ----
    print("\n[eval] FT held-out + trained ...", flush=True)
    ro_te2, raw_te2 = readout(peft_model, tokenizer, device, test_words, args.pca_dim,
                              args.decimals, args.max_new, args.eval_bs,
                              query_for_word=query_for_word)
    m_ft_te = eval_metrics(ro_te2, gt_test_real, test_words, seed=args.seed)  # 真信号 fair compare
    # trained eval 仍用原词 prompt: 测的是 LoRA 记忆训练分布, 不受 query_mode 影响
    ro_tr2, raw_tr2 = readout(peft_model, tokenizer, device, train_words, args.pca_dim,
                              args.decimals, args.max_new, args.eval_bs)
    m_ft_tr = eval_metrics(ro_tr2, train_target, train_words, seed=args.seed)  # 跟训练一致, 测记忆
    for name, raw in [("gen_ft_heldout.jsonl", raw_te2),
                      ("gen_ft_trained.jsonl", raw_tr2)]:
        with open(os.path.join(rawdir, name), "w", encoding="utf-8") as f:
            for r in raw:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results["ft_heldout"] = {k: v for k, v in m_ft_te.items() if k != "null"}
    results["ft_trained"] = {k: v for k, v in m_ft_tr.items() if k != "null"}
    print(f"  FT held-out: parse={m_ft_te.get('parse_rate'):.2f} "
          f"pearson={m_ft_te.get('per_word_pearson_mean')} "
          f"rsa={m_ft_te.get('rsa')} p={m_ft_te.get('rsa_p')}")
    print(f"  FT trained : parse={m_ft_tr.get('parse_rate'):.2f} "
          f"pearson={m_ft_tr.get('per_word_pearson_mean')} "
          f"rsa={m_ft_tr.get('rsa')} (记忆 check, 应高)")

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] -> {args.outdir}/summary.json + raw/ + adapter/")
    print("[判据] FT held-out RSA / pearson 明显 > base held-out -> 学到可泛化读出")


if __name__ == "__main__":
    main()
