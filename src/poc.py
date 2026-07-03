"""
poc.py — 探测 LLM 编造的"假嵌入"与其真实 token embedding 的结构关联 (H2/RSA)。

v3 (干净复核):
  - 用精挑的常见英文词 (再过单 token 过滤)，剔除 BPE 子词碎片。
  - max_new_tokens 拉大，确保 n_fab 个数字吐得完，解析率恢复 -> k 次平均真正生效。
  - 多种子重复，看显著性是否稳定复现 (判断真信号 vs 侥幸)。
  - 批量生成 + attention_mask；顶部强制 UTF-8 输出。

流程：真实向量 = token 嵌入矩阵对应行；编造向量 = 让模型瞎编 n_fab 个数字 (k 次取均值)；
      RSA = 词间真实相似度结构 vs 编造相似度结构的 Spearman；置换检验给 p 值。
测的是 H2 (结构版)。
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

# 精挑常见英文词 (跨语义类别)；下方会再过单 token 过滤
WORD_BANK = """
cat dog horse cow pig sheep lion tiger bear wolf fox deer rabbit mouse bird eagle fish shark
whale dolphin snake frog bee ant spider duck goose chicken
water fire earth wind rain snow ice storm cloud sky sun moon star ocean sea river lake
mountain hill valley forest tree flower grass leaf rock stone sand soil
head face eye ear nose mouth hand foot arm leg heart brain blood bone skin hair tooth finger
man woman child baby boy girl king queen doctor teacher student soldier farmer artist writer
leader friend enemy
love hate fear hope joy anger peace war truth power money time space life death dream idea
mind soul faith
book pen paper table chair door window house car train ship plane road bridge wall floor
roof bed clock phone computer machine engine wheel knife glass cup plate
bread milk meat rice fruit apple cheese sugar salt wine beer coffee
big small fast slow hot cold dark light hard soft new old young rich poor strong weak happy
sad good bad high low long short deep wide clean dirty
run walk jump fall rise sit stand sleep eat drink speak listen read write think know learn
teach build break open close give take buy sell win lose
red blue green yellow black white brown pink
city town village country world north south east west home school market church
music art science history trade law order garden field river music color number letter word
language story song dance game sport team player coach
""".split()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_single_token(tokenizer, words):
    """保留 ' '+word 恰好 1 个 token 的真词，去重，返回 [(word, token_id), ...]。"""
    kept, seen = [], set()
    for w in words:
        if w in seen:
            continue
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            seen.add(w)
            kept.append((w, ids[0]))
    return kept


def build_prompt(tokenizer, word, n_fab):
    msg = [{
        "role": "user",
        "content": (
            f"Pretend you can read the internal embedding vector of a word and write it out. "
            f"Output EXACTLY {n_fab} numbers between -1 and 1, comma-separated, and nothing else. "
            f'Word: "{word}"'
        ),
    }]
    return tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)


def generate_batch(model, tokenizer, device, prompts, max_new_tokens, temperature=0.7, top_p=0.9):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)  # temperature<=0 -> 贪婪/argmax 读出 (接近 D 族 logits 读出极限)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
    return tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


def parse_nums(text, n_fab):
    nums = [float(x) for x in NUM_RE.findall(text)]
    return np.array(nums[:n_fab], dtype=np.float64) if len(nums) >= n_fab else None


def cosine_sim_matrix(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    unit = mat / norm
    return unit @ unit.T


def upper_tri(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def rsa(real_mat, fab_mat):
    rho, _ = spearmanr(upper_tri(cosine_sim_matrix(real_mat)),
                       upper_tri(cosine_sim_matrix(fab_mat)))
    return rho


def permutation_test(real_mat, fab_mat, n_perm, rng):
    observed = rsa(real_mat, fab_mat)
    n = fab_mat.shape[0]
    null = np.array([rsa(real_mat, fab_mat[rng.permutation(n)]) for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, null, p


def fabricate_all(model, tokenizer, device, words, args):
    """每词 k 次批量编造，取均值；返回 (fab_mat, kept_word_indices, avg_samples, raw_records)。"""
    flat_idx, flat_prompts = [], []
    for wi, (w, _) in enumerate(words):
        p = build_prompt(tokenizer, w, args.n_fab)
        for _ in range(args.k):
            flat_idx.append(wi)
            flat_prompts.append(p)

    parsed = [[] for _ in range(len(words))]
    raw = []  # 原始数据: 每次生成的文本 + 解析结果
    for b in range(0, len(flat_prompts), args.batch_size):
        texts = generate_batch(model, tokenizer, device,
                               flat_prompts[b:b + args.batch_size], args.max_new_tokens,
                               args.temperature, args.top_p)
        for wi, txt in zip(flat_idx[b:b + args.batch_size], texts):
            v = parse_nums(txt, args.n_fab)
            raw.append({"word": words[wi][0], "token_id": int(words[wi][1]),
                        "raw_text": txt, "parsed": (v.tolist() if v is not None else None)})
            if v is not None:
                parsed[wi].append(v)

    fab_rows, kept_idx, counts = [], [], []
    for wi in range(len(words)):
        if parsed[wi]:
            fab_rows.append(np.mean(parsed[wi], axis=0))
            kept_idx.append(wi)
            counts.append(len(parsed[wi]))
    if not fab_rows:
        # 全部解析失败 (如 gemma 不吐数字); 返回空矩阵让 main 落 raw + 跳指标, 不抛错
        return (np.zeros((0, args.n_fab), dtype=np.float64),
                kept_idx, 0.0, raw)
    return np.stack(fab_rows), kept_idx, float(np.mean(counts)), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n_fab", type=int, default=16)
    ap.add_argument("--k", type=int, default=5, help="每词重复编造次数 (取平均)")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7, help="<=0 则贪婪(argmax)读出, 不采样")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  model={args.model}")
    print("[load] tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = (model.half() if device == "cuda" else model).to(device).eval()

    words = filter_single_token(tokenizer, WORD_BANK)
    print(f"[words] {len(words)} clean single-token words "
          f"(e.g. {[w for w, _ in words[:12]]})")

    emb = model.get_input_embeddings().weight.detach()
    real_mat = np.stack([emb[tid].float().cpu().numpy() for _, tid in words])
    print(f"[real] embedding matrix shape = {real_mat.shape}")

    rawdir = os.path.join(args.outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    # 原始数据: 真实嵌入矩阵 + 词表 (一次)
    np.savez(os.path.join(rawdir, "real_embeddings.npz"), real=real_mat,
             words=np.array([w for w, _ in words]),
             token_ids=np.array([t for _, t in words]))

    rows = []
    print(f"\n[run] {args.n_seeds} seeds x (k={args.k} samples/word, max_new_tokens={args.max_new_tokens})")
    for seed in range(args.n_seeds):
        set_seed(seed)
        fab_mat, kept_idx, avg_s, raw = fabricate_all(model, tokenizer, device, words, args)
        real_used = real_mat[kept_idx]
        n_kept = len(kept_idx)
        # 先落 raw, 即便指标算不出来也保存原始生成 (供 gemma 这种全失败诊断)
        with open(os.path.join(rawdir, f"generations_seed{seed}.jsonl"), "w", encoding="utf-8") as fjs:
            for rec in raw:
                fjs.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n_kept < 2:
            # 解析成功太少 -> 指标无意义, 跳过本 seed 的 RSA/置换, 但记入 rows 作 NaN
            print(f"  seed={seed}: kept={n_kept:3d} avg_samp={avg_s:.2f}/{args.k} "
                  f"[skip-metric] 解析成功 <2, 无法算指标, 仅落 raw")
            rows.append((seed, n_kept, avg_s, float("nan"),
                         float("nan"), float("nan"), float("nan")))
            np.savez(os.path.join(rawdir, f"vectors_seed{seed}.npz"),
                     fab=fab_mat, real=real_used,
                     words=np.array([words[i][0] for i in kept_idx]),
                     null=np.array([], dtype=np.float64),
                     observed=float("nan"))
            continue
        mean_off = upper_tri(cosine_sim_matrix(fab_mat)).mean()
        rng = np.random.default_rng(seed)
        observed, null, p = permutation_test(real_used, fab_mat, args.n_perm, rng)
        z = (observed - null.mean()) / (null.std() + 1e-12)
        rows.append((seed, n_kept, avg_s, mean_off, observed, z, p))
        print(f"  seed={seed}: kept={n_kept:3d} avg_samp={avg_s:.2f}/{args.k} "
              f"fabcos={mean_off:.3f}  RSA={observed:+.4f} z={z:+.2f} p={p:.4f}")
        np.savez(os.path.join(rawdir, f"vectors_seed{seed}.npz"), fab=fab_mat, real=real_used,
                 words=np.array([words[i][0] for i in kept_idx]), null=null, observed=observed)

    # 汇总
    arr = np.array([(r[4], r[6]) for r in rows])  # (RSA, p)
    n_sig = int(np.sum(arr[:, 1] < 0.05))
    print("\n==================== SUMMARY (multi-seed) ====================")
    print(f"  seeds                  = {args.n_seeds}")
    print(f"  mean RSA               = {arr[:,0].mean():+.4f}  (sd {arr[:,0].std():.4f})")
    print(f"  RSA range              = [{arr[:,0].min():+.4f}, {arr[:,0].max():+.4f}]")
    print(f"  seeds with p<0.05      = {n_sig}/{args.n_seeds}")
    print(f"  p values               = {[f'{x:.4f}' for x in arr[:,1]]}")
    stable = "信号稳定复现" if n_sig == args.n_seeds else (
        "部分复现 (需更多功率)" if n_sig > 0 else "未复现 (无信号)")
    print(f"  --> {stable}")
    print("=============================================================")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "n_fab": args.n_fab, "k": args.k,
            "temperature": args.temperature, "top_p": args.top_p,
            "n_words_bank": len(words), "seeds": args.n_seeds,
            "mean_rsa": float(arr[:, 0].mean()), "rsa_sd": float(arr[:, 0].std()),
            "p_values": [float(x) for x in arr[:, 1]], "n_sig": n_sig,
            "per_seed": [{"seed": r[0], "n_kept": r[1], "avg_samples": r[2],
                          "fab_cos": r[3], "rsa": r[4], "z": r[5], "p": r[6]} for r in rows],
        }, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {args.outdir}/summary.json + raw/ (generations.jsonl, vectors, real_embeddings)")


if __name__ == "__main__":
    main()
