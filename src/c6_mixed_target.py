"""C6. 混合目标 LoRA 读出 (mixed-target, 2026-06-16)。

一个 LoRA adapter 同时读出**两个正交参数空间**:
  - 一部分训练词 target = input_embedding 的 PCA-32 (C1 目标)
  - 另一部分训练词 target = lm_head 行的 PCA-32  (C2 目标)
比例由 `--ratio` 控制 (= input_embedding 占比, 如 0.7).

## 科学问题
C2 已证 input_embed ⊥ lm_head (per-token cos≈0, 两个正交参数空间), 而 C1/C2 是
**两个独立 adapter**. 本实验问: **一个** adapter 能否**同时**读出这两个正交空间?
若能 → 支持"通用读自己参数"机制 (而非只照搬眼前激活, 因 input_embed 在浅层残差流
里, lm_head 不在前向中直接存在).

## 关键设计点 (与 C1 的区别)
同一个词在两种目标下要吐**不同**的数, 所以 prompt **必须带"目标类型"信号** (input
还是 output 嵌入), 否则同词两套标签就是噪声. 三种信号设计 `--prompt_style`:
  - tag:    在 C1 文案基础上加一行 `Target: input embedding` / `Target: output embedding`
  - verbal: 文案改成 "Read out the **input embedding** vector ..." / "**output
            (unembedding)** vector ..."
  - symbol: prompt 前缀 `[IN]` / `[OUT]`
训练 (make_example_typed) 与 eval (readout_typed) 用**同一个** typed prompt.

## 双 PCA basis (防泄漏 + 跨 ratio 可比)
ie_basis = fit_pca(所有 train 词的 input_embeddings, d=32)
lh_basis = fit_pca(所有 train 词的 lm_head 行,       d=32)
两个 basis 都**只在 train 词上拟合**, 且**与 ratio 无关** (用全部 train 词, 不是子集),
保证不同 ratio 跑出来的 held-out 数字可比. test 词用对应 basis project.

## per-example 目标分配
按 seed 确定性地把 train 词切成 ie 子集 (round(ratio*n_train)) 和 lh 子集, 各自用
对应 basis 的 PCA-32 作 target + 对应 typed prompt 造训练样本, 混在一起训练.

## 双 eval (同一批 held-out test 词, 评两次)
  - type=input  prompt → 与 ie-PCA gt 比 → `ie_heldout`
  - type=output prompt → 与 lh-PCA gt 比 → `lh_heldout`

## 标签互换控制 (必做)
  - type=input  prompt 但与 **lh**-PCA gt 比 → `swap_input_vs_lh`
  - type=output prompt 但与 **ie**-PCA gt 比 → `swap_output_vs_ie`
若模型真用类型信号, swap RSA 应明显低于对应正确配对.

## 复用 c1_lora (绝不修改它)
import: systematic_vocab_words / build_train_words / fit_pca / project / fmt_vec /
collate / train_lora / eval_metrics / load_readout_jsonl / 4 道防污染闸
(snapshot_base/verify_snapshot/assert_base_frozen/disk_mtime/verify_disk_mtime) /
StopOnDoubleNewline / find_latest_ckpt / LORA_TARGET_MODULES. 模型加载 + LoRA 配置
逻辑照 c1_lora.main() 复刻.

## 约束
- 只 untied 模型 (开头 assert tie_word_embeddings is False).
- 4 道防污染闸照 C1 跑 (base 权重不改).
- raw 落盘: gen_*.jsonl + pca_targets_c6.npz (两 basis mu/comps + train/test words +
  assignment).
- 断点: 有 adapter/final + 完整 summary 则 skip.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from poc import NUM_RE, set_seed

# 复用 c1_lora (绝不修改 c1_lora.py)
from c1_lora import (
    LORA_TARGET_MODULES,
    systematic_vocab_words,
    build_train_words,
    fit_pca,
    project,
    fmt_vec,
    collate,
    train_lora,
    eval_metrics,
    load_readout_jsonl,
    find_latest_ckpt,
    StopOnDoubleNewline,
    # 4 道防污染闸
    snapshot_base,
    verify_snapshot,
    assert_base_frozen,
    disk_mtime,
    verify_disk_mtime,
)
from transformers import StoppingCriteriaList

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================ typed prompt (新增) ============================
def build_prompt_typed(tokenizer, word, d, ttype, style):
    """带"目标类型"信号的 prompt. ttype∈{input,output}, style∈{tag,verbal,symbol}.

    训练 (make_example_typed) 与 eval (readout_typed) 用同一个函数, 保证训/测同分布.
    """
    if ttype not in ("input", "output"):
        raise ValueError(f"ttype must be input/output, got {ttype}")

    if style == "tag":
        # C1 文案 + 一行 Target 信号
        line = "Target: input embedding" if ttype == "input" else "Target: output embedding"
        content = (f"Read out the internal embedding vector of a word as {d} numbers, "
                   f"comma-separated, 2 decimals each.\n{line}\nWord: {word}")
    elif style == "verbal":
        if ttype == "input":
            what = "input embedding"
        else:
            what = "output (unembedding)"
        content = (f"Read out the {what} vector of the word as {d} numbers, "
                   f"comma-separated, 2 decimals each. Word: {word}")
    elif style == "symbol":
        prefix = "[IN]" if ttype == "input" else "[OUT]"
        content = (f"{prefix} Read out the internal embedding vector of a word as {d} "
                   f"numbers, comma-separated, 2 decimals each. Word: {word}")
    else:
        raise ValueError(f"unknown prompt_style: {style}")

    if getattr(tokenizer, "chat_template", None):
        msg = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    # base 模型 fallback (无 chat_template): 朴素 Q/A 模板
    return f"Q: {content}\nA: "


def make_example_typed(tokenizer, word, target_vec, d, decimals, max_len, ttype, style):
    """typed 版 make_example: prompt 带类型信号, completion = target 向量数字串 + eos.
    completion-only loss (prompt token labels = -100), 与 c1_lora.make_example 同结构.
    """
    prompt = build_prompt_typed(tokenizer, word, d, ttype, style)
    completion = fmt_vec(target_vec, decimals) + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return ids, labels


@torch.no_grad()
def readout_typed(model, tokenizer, device, words, d, decimals, max_new, batch_size,
                  ttype, style):
    """typed 版 readout: 用 typed prompt 生成读出. 与 c1_lora.readout 同结构 (含
    StopOnDoubleNewline 早停), 但 prompt 带类型信号. 返回 (out_dict, raw_list).
    """
    model.eval()
    prompts = [build_prompt_typed(tokenizer, w, d, ttype, style) for w, _ in words]
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
            raw.append({"word": w, "token_id": int(tid), "ttype": ttype,
                        "prompt_style": style, "raw_text": txt,
                        "parsed": (v.tolist() if v is not None else None)})
    return out, raw


def _strip_null(metric_dict):
    """去掉 eval_metrics 返回里的 numpy null 数组 (不可 JSON 序列化)."""
    return {k: v for k, v in metric_dict.items() if k != "null"}


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="results_c6")
    ap.add_argument("--ratio", type=float, default=0.7,
                    help="input_embedding 目标占比 (0..1). 其余词 target = lm_head PCA-32.")
    ap.add_argument("--prompt_style", default="tag", choices=["tag", "verbal", "symbol"],
                    help="目标类型信号设计. tag=加一行 Target:..., verbal=文案改写, "
                         "symbol=前缀 [IN]/[OUT].")
    ap.add_argument("--train_vocab", default="random", choices=["basic", "random", "freq"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_new", type=int, default=250)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--skip_base", action="store_true",
                    help="跳过 base 零样本 eval (默认跑一次 type=input 的 base_heldout).")
    ap.add_argument("--eval_only", action="store_true",
                    help="检测 adapter/final, load 跑 eval (不重训). 无 final 报错.")
    args = ap.parse_args()

    if not (0.0 <= args.ratio <= 1.0):
        raise SystemExit(f"--ratio 必须在 [0,1], got {args.ratio}")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rawdir = os.path.join(args.outdir, "raw")
    adapter_dir = os.path.join(args.outdir, "adapter")
    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    print(f"[env] device={device} model={args.model}")
    print(f"[env] outdir={args.outdir}  ratio={args.ratio}  prompt_style={args.prompt_style}")

    # ---- 断点: 有 final + 完整 summary 则 skip (照 c1 风格) ----
    final_adapter = os.path.join(adapter_dir, "final")
    summary_path = os.path.join(args.outdir, "summary.json")
    if not args.eval_only and os.path.exists(final_adapter) and os.path.exists(summary_path):
        try:
            done = json.load(open(summary_path, encoding="utf-8"))
        except Exception:
            done = {}
        if ("ie_heldout" in done) and ("lh_heldout" in done):
            print(f"[skip] adapter/final + 完整 summary 已存在 -> 跳过 ({summary_path})")
            return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # 闸 4 (前): 读磁盘 model 文件 mtime
    before_disk_path, before_disk_mtime = disk_mtime(args.model)
    if before_disk_path:
        print(f"  [闸 4 snap] {before_disk_path} mtime={before_disk_mtime}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
    model = model.to(device)

    # ---- 约束: 只支持 untied 模型 (lm_head 目标要求 lm_head ≠ input_embed) ----
    if getattr(model.config, "tie_word_embeddings", True) is not False:
        raise SystemExit(
            "[C6] 本实验混合 input_embedding 与 lm_head 两种目标, 要求 untied 模型 "
            "(tie_word_embeddings=False), 否则 lm_head == input_embed 两目标退化为同一个.\n"
            "  仅支持 OLMo-2-0425-1B-Instruct / pythia-1.4b 等 untied 模型.")
    print(f"  [ok] untied 模型 (tie_word_embeddings=False), 两目标空间正交")

    # 闸 1 (前): snapshot base 权重 sha256
    print("[闸 1] base 权重 snapshot (训前):")
    snap_before = snapshot_base(model)
    for k, v in snap_before.items():
        print(f"  {k}: {v}")

    # ============================ 词集 + 双 PCA basis + target 分配 ============================
    # 断点复用: pca_targets_c6.npz freeze 词集 / 两 basis / assignment (跨 process 词集稳定).
    pca_file = os.path.join(rawdir, "pca_targets_c6.npz")
    if os.path.exists(pca_file):
        print(f"[resume] 从 {pca_file} 加载词集 + 双 PCA basis + target 分配")
        z = np.load(pca_file, allow_pickle=True)
        # 校验关键参数一致 (防 outdir 复用错配)
        for key, cur in [("ratio", args.ratio), ("prompt_style", args.prompt_style),
                         ("seed", args.seed)]:
            saved = z[key].item() if z[key].shape == () else z[key]
            if key == "ratio":
                if abs(float(saved) - float(cur)) > 1e-9:
                    raise SystemExit(f"[fatal] npz ratio={saved} 但 --ratio={cur}. 换 outdir.")
            elif str(saved) != str(cur):
                raise SystemExit(f"[fatal] npz {key}={saved} 但 --{key}={cur}. 换 outdir.")

        train_words_str = [str(w) for w in z['train_words']]
        test_words_str = [str(w) for w in z['test_words']]

        def _to_pairs(strs):
            out = []
            for w in strs:
                ids = tokenizer.encode(" " + w, add_special_tokens=False)
                out.append((w, ids[0] if len(ids) == 1 else -1))
            return out

        train_words = _to_pairs(train_words_str)
        test_words = _to_pairs(test_words_str)
        ie_mu, ie_comps = z['ie_mu'], z['ie_comps']
        lh_mu, lh_comps = z['lh_mu'], z['lh_comps']
        assign = list(z['assignment'])  # per train word: "input" / "output"
        gt_ie_test = {str(w): v for w, v in zip(z['test_words'], z['gt_ie_test'])}
        gt_lh_test = {str(w): v for w, v in zip(z['test_words'], z['gt_lh_test'])}
        train_target = {str(w): v for w, v in zip(z['train_words'], z['train_target_actual'])}
    else:
        train_words = build_train_words(tokenizer, args)
        test_words = systematic_vocab_words(tokenizer, args.n_test, args.seed,
                                            exclude_words=[w for w, _ in train_words])
        # 抽两个嵌入矩阵的行
        ie_mat = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        lh_mat = model.get_output_embeddings().weight.detach().float().cpu().numpy()
        ie_tr = np.stack([ie_mat[t] for _, t in train_words])
        ie_te = np.stack([ie_mat[t] for _, t in test_words])
        lh_tr = np.stack([lh_mat[t] for _, t in train_words])
        lh_te = np.stack([lh_mat[t] for _, t in test_words])
        # sanity: 两矩阵确实不同 (per-token cos 应 ≈ 0)
        cs = float(np.mean([
            np.dot(ie_tr[i], lh_tr[i]) /
            (np.linalg.norm(ie_tr[i]) * np.linalg.norm(lh_tr[i]) + 1e-12)
            for i in range(len(train_words))]))
        print(f"  [sanity] per-token cos(input_embed, lm_head) 均值 = {cs:.4f} (应 ≈ 0)")

        # 双 PCA basis: 都用**全部** train 词拟合 (与 ratio 无关 → 跨 ratio 可比, 无泄漏)
        ie_mu, ie_comps = fit_pca(ie_tr, args.pca_dim)
        lh_mu, lh_comps = fit_pca(lh_tr, args.pca_dim)

        # held-out ground-truth: test 词用对应 basis project
        gt_ie_test = {w: project(ie_te[i], ie_mu, ie_comps)
                      for i, (w, _) in enumerate(test_words)}
        gt_lh_test = {w: project(lh_te[i], lh_mu, lh_comps)
                      for i, (w, _) in enumerate(test_words)}

        # per-example 目标分配: 确定性切 ie 子集 (round(ratio*n_train)) + lh 子集
        n_tr = len(train_words)
        n_ie = int(round(args.ratio * n_tr))
        rng_assign = np.random.default_rng(args.seed + 7777)
        perm = rng_assign.permutation(n_tr)
        ie_idx = set(perm[:n_ie].tolist())
        assign = ["input" if i in ie_idx else "output" for i in range(n_tr)]
        # train target: ie 子集 → ie-PCA, lh 子集 → lh-PCA
        ie_tr_proj = {w: project(ie_tr[i], ie_mu, ie_comps)
                      for i, (w, _) in enumerate(train_words)}
        lh_tr_proj = {w: project(lh_tr[i], lh_mu, lh_comps)
                      for i, (w, _) in enumerate(train_words)}
        train_target = {}
        for i, (w, _) in enumerate(train_words):
            train_target[w] = ie_tr_proj[w] if assign[i] == "input" else lh_tr_proj[w]

        np.savez(pca_file,
                 ratio=np.array(args.ratio), prompt_style=np.array(args.prompt_style),
                 seed=np.array(args.seed), pca_dim=np.array(args.pca_dim),
                 ie_mu=ie_mu, ie_comps=ie_comps, lh_mu=lh_mu, lh_comps=lh_comps,
                 train_words=np.array([w for w, _ in train_words]),
                 test_words=np.array([w for w, _ in test_words]),
                 assignment=np.array(assign),
                 gt_ie_test=np.stack([gt_ie_test[w] for w, _ in test_words]),
                 gt_lh_test=np.stack([gt_lh_test[w] for w, _ in test_words]),
                 train_target_actual=np.stack([train_target[w] for w, _ in train_words]))

    n_train_ie = sum(1 for a in assign if a == "input")
    n_train_lh = sum(1 for a in assign if a == "output")
    print(f"[words] train={len(train_words)} (ie={n_train_ie} lh={n_train_lh}) "
          f"test(held-out)={len(test_words)}  ratio={args.ratio} "
          f"prompt_style={args.prompt_style}")

    results = {
        "model": args.model, "experiment": "C6_mixed_target",
        "ratio": args.ratio, "prompt_style": args.prompt_style, "seed": args.seed,
        "train_vocab": args.train_vocab, "pca_dim": args.pca_dim,
        "n_train": len(train_words), "n_test": len(test_words),
        "n_train_ie": n_train_ie, "n_train_lh": n_train_lh,
        "max_new": args.max_new,
        "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
    }

    # ---- base 零样本 eval (可选, type=input → ie gt) ----
    if not args.skip_base:
        base_gen_file = os.path.join(rawdir, "gen_base_heldout.jsonl")
        if not os.path.exists(base_gen_file):
            print("\n[eval] base (zero-shot) held-out  type=input ...", flush=True)
            ro_b, raw_b = readout_typed(model, tokenizer, device, test_words, args.pca_dim,
                                        args.decimals, args.max_new, args.eval_bs,
                                        ttype="input", style=args.prompt_style)
            with open(base_gen_file, "w", encoding="utf-8") as f:
                for r in raw_b:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            print(f"\n[eval] base held-out 已跑过 ({base_gen_file}), 从 raw 重算 metrics")
            ro_b = load_readout_jsonl(base_gen_file)
        m_base = eval_metrics(ro_b, gt_ie_test, test_words, seed=args.seed)
        results["base_heldout"] = _strip_null(m_base)
        print(f"  base held-out (input→ie): parse={m_base.get('parse_rate'):.2f} "
              f"rsa={m_base.get('rsa')} (parse 应 ≈ 0)")

    # ============================ LoRA setup / 训练 / 续训 / eval-only ============================
    if args.eval_only:
        if not os.path.exists(final_adapter):
            raise SystemExit(f"--eval_only 但 {final_adapter} 不存在; 先训练")
        print(f"\n[eval_only] load final adapter <- {final_adapter}")
        peft_model = PeftModel.from_pretrained(model, final_adapter)
    else:
        # 构造混合训练样本: 每词用其 assignment 的 ttype + 对应 target
        examples = [
            make_example_typed(tokenizer, w, train_target[w], args.pca_dim,
                               args.decimals, args.max_len, assign[i], args.prompt_style)
            for i, (w, _) in enumerate(train_words)
        ]
        latest_ckpt, latest_ep = find_latest_ckpt(adapter_dir)
        if os.path.exists(final_adapter):
            print(f"\n[train] final adapter 已存在, 复用; 跳过训练")
            peft_model = PeftModel.from_pretrained(model, final_adapter, is_trainable=False)
        elif latest_ep >= args.epochs:
            print(f"\n[train] 最新 ckpt ep{latest_ep} >= epochs={args.epochs}, 标 final 并跳过")
            peft_model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=False)
            peft_model.save_pretrained(final_adapter)
        elif latest_ckpt is not None:
            print(f"\n[train] 从 {latest_ckpt} 续训 (epoch {latest_ep+1}..{args.epochs})")
            peft_model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
            assert_base_frozen(peft_model)
            resume_state = None
            opt_path = os.path.join(latest_ckpt, "_opt_state.pt")
            rng_path = os.path.join(latest_ckpt, "_rng_state.pt")
            if os.path.exists(opt_path) and os.path.exists(rng_path):
                resume_state = {"opt": torch.load(opt_path, map_location=device,
                                                  weights_only=False)}
                resume_state.update(torch.load(rng_path, weights_only=False))
                print(f"  [resume] _opt_state.pt + _rng_state.pt 已加载 (bit-exact 续训)")
            else:
                print(f"  [resume] WARNING: {latest_ckpt} 无 opt/rng state, 续训非 bit-exact.")
            train_lora(peft_model, tokenizer, device, examples, args, adapter_dir,
                       start_ep=latest_ep, resume_state=resume_state)
        else:
            print("\n[train] 从零开始 fresh LoRA")
            mtype = getattr(model.config, "model_type", None)
            if mtype not in LORA_TARGET_MODULES:
                raise SystemExit(f"未知 model_type='{mtype}'. 加到 LORA_TARGET_MODULES 再跑.")
            target_modules = LORA_TARGET_MODULES[mtype]
            print(f"  [LoRA] model_type={mtype}, target_modules={target_modules}")
            lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=target_modules)
            peft_model = get_peft_model(model, lcfg)
            print("[闸 2] 验证 base 完全冻结:")
            assert_base_frozen(peft_model)
            n_train_p = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            emb_grad = peft_model.get_input_embeddings().weight.requires_grad
            print(f"  trainable params={n_train_p:,}  embed.requires_grad={emb_grad}")
            results["lora_trainable_params"] = int(n_train_p)
            results["embed_frozen"] = (not emb_grad)
            print(f"[train] {len(examples)} examples (混合 ie/lh), {args.epochs} ep, "
                  f"lr={args.lr}", flush=True)
            train_lora(peft_model, tokenizer, device, examples, args, adapter_dir, start_ep=0)

    # 闸 1 (后) + 闸 4 (后): 验证 base 未污染
    print("\n[闸 1] base 权重 snapshot (训后):")
    snap_after = snapshot_base(model)
    verify_snapshot(snap_before, snap_after)
    verify_disk_mtime(args.model, before_disk_path, before_disk_mtime)

    # ============================ 双 eval + 互换控制 ============================
    print("\n[eval] FT held-out: type=input→ie, type=output→lh + 互换控制 ...", flush=True)

    # type=input prompt → 一次生成, 既算 ie_heldout (正确配对) 也算 swap_input_vs_lh
    ro_in, raw_in = readout_typed(peft_model, tokenizer, device, test_words, args.pca_dim,
                                  args.decimals, args.max_new, args.eval_bs,
                                  ttype="input", style=args.prompt_style)
    # type=output prompt → 一次生成, 既算 lh_heldout 也算 swap_output_vs_ie
    ro_out, raw_out = readout_typed(peft_model, tokenizer, device, test_words, args.pca_dim,
                                    args.decimals, args.max_new, args.eval_bs,
                                    ttype="output", style=args.prompt_style)

    for name, raw in [("gen_ft_heldout_input.jsonl", raw_in),
                      ("gen_ft_heldout_output.jsonl", raw_out)]:
        with open(os.path.join(rawdir, name), "w", encoding="utf-8") as f:
            for r in raw:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 正确配对
    m_ie = eval_metrics(ro_in, gt_ie_test, test_words, seed=args.seed)
    m_lh = eval_metrics(ro_out, gt_lh_test, test_words, seed=args.seed)
    # 互换控制 (同一批生成, 换 gt)
    m_swap_in_lh = eval_metrics(ro_in, gt_lh_test, test_words, seed=args.seed)
    m_swap_out_ie = eval_metrics(ro_out, gt_ie_test, test_words, seed=args.seed)

    results["ie_heldout"] = _strip_null(m_ie)
    results["lh_heldout"] = _strip_null(m_lh)
    results["swap_input_vs_lh"] = {"rsa": m_swap_in_lh.get("rsa"),
                                   "rsa_p": m_swap_in_lh.get("rsa_p"),
                                   "n_kept": m_swap_in_lh.get("n_kept"),
                                   "parse_rate": m_swap_in_lh.get("parse_rate")}
    results["swap_output_vs_ie"] = {"rsa": m_swap_out_ie.get("rsa"),
                                    "rsa_p": m_swap_out_ie.get("rsa_p"),
                                    "n_kept": m_swap_out_ie.get("n_kept"),
                                    "parse_rate": m_swap_out_ie.get("parse_rate")}

    print(f"  ie_heldout (input→ie) : parse={m_ie.get('parse_rate'):.2f} "
          f"rsa={m_ie.get('rsa')} p={m_ie.get('rsa_p')}")
    print(f"  lh_heldout (output→lh): parse={m_lh.get('parse_rate'):.2f} "
          f"rsa={m_lh.get('rsa')} p={m_lh.get('rsa_p')}")
    print(f"  swap_input_vs_lh  (input→lh) : rsa={m_swap_in_lh.get('rsa')}  "
          f"(应 < ie_heldout 若模型用类型信号)")
    print(f"  swap_output_vs_ie (output→ie): rsa={m_swap_out_ie.get('rsa')}  "
          f"(应 < lh_heldout 若模型用类型信号)")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] -> {summary_path} + raw/ + adapter/")
    print("[判据] ie_heldout 与 lh_heldout 同时显著 > swap 对应 + base "
          "-> 一个 adapter 能同时读两个正交参数空间")


if __name__ == "__main__":
    main()
