"""受控对比: system prompt 的 persona 是否改变 编造向量 与 真实嵌入 的结构相关 (RSA)。

防 p-hack 设计:
  - 固定一小组 prompt: default(中性) / innate(你天生懂embedding) / placebo(自信但无关的诗人)
  - 每个 prompt 跑多种子 + 置换检验
  - 同一种子下, 三个 prompt 在「相同词子集(交集)」上比, 保证可比
  - 看点: innate 抬高 RSA 而 placebo 没有 -> 才算证据; 两个都抬 -> 只是输出风格变了
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poc import (WORD_BANK, cosine_sim_matrix, filter_single_token, generate_batch,
                 parse_nums, permutation_test, set_seed, upper_tri)

CONDITIONS = {
    "default": None,  # None -> 用 Qwen 模板默认 system
    "innate": "You are familiar with every embedding vector inherently.",
    "placebo": "You are a brilliant Renaissance poet.",
}


def build_prompt_sys(tokenizer, word, n_fab, system):
    user = (f"Pretend you can read the internal embedding vector of a word and write it out. "
            f"Output EXACTLY {n_fab} numbers between -1 and 1, comma-separated, and nothing else. "
            f'Word: "{word}"')
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def fabricate_dict(model, tokenizer, device, words, system, args):
    """返回 {word_index: 平均编造向量} (只含至少成功解析 1 次的词)。"""
    flat_idx, flat_prompts = [], []
    for wi, (w, _) in enumerate(words):
        p = build_prompt_sys(tokenizer, w, args.n_fab, system)
        for _ in range(args.k):
            flat_idx.append(wi)
            flat_prompts.append(p)
    parsed = {}
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
                parsed.setdefault(wi, []).append(v)
    return {wi: np.mean(vs, axis=0) for wi, vs in parsed.items()}, raw


def rsa_on(real_mat, fab_dict, common):
    fab = np.stack([fab_dict[wi] for wi in common])
    real = real_mat[common]
    rng = np.random.default_rng(0)
    return permutation_test(real, fab, 2000, rng)  # (observed, null, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n_fab", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7, help="<=0 则贪婪(argmax)读出, 不采样")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--outdir", default="results_persona")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = (model.half() if device == "cuda" else model).to(device).eval()

    words = filter_single_token(tokenizer, WORD_BANK)
    emb = model.get_input_embeddings().weight.detach()
    real_mat = np.stack([emb[tid].float().cpu().numpy() for _, tid in words])
    print(f"[words] {len(words)} single-token words; real dim={real_mat.shape[1]}")
    print(f"[design] conditions={list(CONDITIONS)}  seeds={args.n_seeds}  k={args.k}\n")

    # ---- 原始数据落盘 (raw): 真实嵌入矩阵 + 词表 (一次) ----
    rawdir = os.path.join(args.outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    np.savez(os.path.join(rawdir, "real_embeddings.npz"), real=real_mat,
             words=np.array([w for w, _ in words]),
             token_ids=np.array([t for _, t in words]))

    # results[cond] = list of (rsa, p, n_common) per seed
    results = {c: [] for c in CONDITIONS}
    for seed in range(args.n_seeds):
        set_seed(seed)
        fab, raws = {}, {}
        for cond, sysp in CONDITIONS.items():
            fab[cond], raws[cond] = fabricate_dict(model, tokenizer, device, words, sysp, args)
            # 原始数据: 每条件每次生成文本(JSONL)
            with open(os.path.join(rawdir, f"generations_seed{seed}_{cond}.jsonl"), "w", encoding="utf-8") as fjs:
                for rec in raws[cond]:
                    fjs.write(json.dumps(rec, ensure_ascii=False) + "\n")
        common = sorted(set.intersection(*[set(fab[c].keys()) for c in CONDITIONS]))
        print(f"seed={seed}: common kept words across conditions = {len(common)}")
        for cond in CONDITIONS:
            obs, null, p = rsa_on(real_mat, fab[cond], common)
            z = (obs - null.mean()) / (null.std() + 1e-12)
            results[cond].append((obs, p, len(common)))
            # 原始数据: 本 seed×cond 的 fab/real(共同词) + null 分布
            np.savez(os.path.join(rawdir, f"vectors_seed{seed}_{cond}.npz"),
                     fab=np.stack([fab[cond][wi] for wi in common]),
                     real=real_mat[common], words=np.array([words[wi][0] for wi in common]),
                     null=null, observed=obs)
            print(f"   {cond:8s}: RSA={obs:+.4f}  z={z:+.2f}  p={p:.4f}")
        print()

    print("==================== SUMMARY ====================")
    print(f"{'cond':10s} {'mean_RSA':>9s} {'sd':>7s} {'sig/seeds':>10s}  p-values")
    for cond in CONDITIONS:
        rsas = np.array([r[0] for r in results[cond]])
        ps = [r[1] for r in results[cond]]
        nsig = sum(p < 0.05 for p in ps)
        print(f"{cond:10s} {rsas.mean():+9.4f} {rsas.std():7.4f} {nsig:>6d}/{args.n_seeds}   "
              f"{['%.4f' % p for p in ps]}")
    print("=================================================")

    # ---- summary.json 落盘 ----
    summary = {"model": args.model, "n_fab": args.n_fab, "k": args.k,
               "temperature": args.temperature, "top_p": args.top_p,
               "n_seeds": args.n_seeds, "conditions": {}}
    for cond in CONDITIONS:
        rsas = np.array([r[0] for r in results[cond]])
        ps = [r[1] for r in results[cond]]
        summary["conditions"][cond] = {
            "system": CONDITIONS[cond], "mean_rsa": float(rsas.mean()),
            "rsa_sd": float(rsas.std()), "n_sig": int(sum(p < 0.05 for p in ps)),
            "per_seed": [{"seed": s, "rsa": float(r[0]), "p": float(r[1]), "n_common": r[2]}
                         for s, r in enumerate(results[cond])]}
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {args.outdir}/summary.json + raw/ (generations_*, vectors_*, real_embeddings)")

    print("判读: innate 稳定 > default 且 placebo 没抬 -> persona 真有结构效应;")
    print("      innate≈placebo 都变 -> 只是输出风格变, 非自我认知; 都≈0 -> persona 无效。")


if __name__ == "__main__":
    main()
