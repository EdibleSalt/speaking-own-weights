"""G3. 跨模型真实嵌入 RDM 静态相似度诊断

回应 [[研究纲要]] G 族漏洞 2 (C4 cross > self 反例 / Pythia 学 OLMo 比学自己易).
若 RSA(RDM_OLMo, RDM_pythia) 静态就高 → C4 反例在数学预期内, 不是异常.

完全静态分析: 抽两模型 input_embed / lm_head 矩阵的双向交集词行 → cosine RDM →
Spearman RSA. 无 LoRA, 无 inference. < 5 min.

用法 (run_pending phase G 或独立):
    .venv/Scripts/python.exe src/g3_cross_rdm.py \
        --model_a models/OLMo-2-0425-1B-Instruct \
        --model_b models/pythia-1.4b \
        --outdir results/data/G3_cross_rdm/olmo_vs_pythia_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def bidirectional_vocab_words(tokA, tokB, n_want, seed):
    """抽 A & B 都是单 token 的双向交集 (复用 c1_lora.systematic_vocab_words 精简版).
    返回: [(word, tokA_id, tokB_id), ...]
    """
    vocab = tokA.get_vocab()
    cand = []
    for tokstr, tid in vocab.items():
        s = tokA.convert_tokens_to_string([tokstr])
        if not s.startswith(" "):
            continue
        w = s[1:]
        if not (len(w) >= 3 and w.isascii() and w.isalpha() and w.islower()):
            continue
        a_ids = tokA.encode(" " + w, add_special_tokens=False)
        if len(a_ids) != 1 or a_ids[0] != tid:
            continue
        b_ids = tokB.encode(" " + w, add_special_tokens=False)
        if len(b_ids) != 1:
            continue
        cand.append((w, tid, b_ids[0]))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cand))[:n_want]
    return [cand[i] for i in sorted(idx)]


def get_emb_matrix(model_path, kind):
    """kind: 'input_embed' | 'lm_head'. fp32, CPU. load 完即 del 释放显存."""
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    if kind == "input_embed":
        W = m.get_input_embeddings().weight.detach().cpu().numpy().copy()
    elif kind == "lm_head":
        W = m.get_output_embeddings().weight.detach().cpu().numpy().copy()
    else:
        raise ValueError(kind)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W


def cosine_rdm(X):
    """X: (n, d) -> (n, n) 1 - cos similarity (squareform of pdist)."""
    return squareform(pdist(X, metric="cosine"))


def upper_tri(D):
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def rdm_rsa(D1, D2):
    """Spearman RSA of upper-triangular vectorized RDMs."""
    return float(spearmanr(upper_tri(D1), upper_tri(D2)).statistic)


def per_tok_cos(X, Y):
    """Mean per-token cosine(X[i], Y[i]). 仅适用同模型 ie ↔ lh (维度同)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return float((Xn * Yn).sum(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_a", required=True, help="本地路径或 HF id")
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_words", type=int, default=300,
                    help="双向交集中抽多少词作 RDM (默认 300, ~45K 上三角对)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "raw"), exist_ok=True)

    print(f"[G3] tokenizers...", flush=True)
    tokA = AutoTokenizer.from_pretrained(args.model_a)
    tokB = AutoTokenizer.from_pretrained(args.model_b)
    words = bidirectional_vocab_words(tokA, tokB, args.n_words, args.seed)
    n = len(words)
    print(f"[G3] 双向交集词数 = {n} (要求 ≥ {args.n_words})")
    if n < args.n_words:
        print(f"[G3] WARN: 交集不足 {args.n_words}, 实际 {n}, 继续")

    ws = [w for w, _, _ in words]
    aids = [a for _, a, _ in words]
    bids = [b for _, _, b in words]

    print(f"[G3] loading model_a = {args.model_a} ...", flush=True)
    A_ie_full = get_emb_matrix(args.model_a, "input_embed")
    A_lh_full = get_emb_matrix(args.model_a, "lm_head")
    A_ie = A_ie_full[aids]
    A_lh = A_lh_full[aids]
    del A_ie_full, A_lh_full

    print(f"[G3] loading model_b = {args.model_b} ...", flush=True)
    B_ie_full = get_emb_matrix(args.model_b, "input_embed")
    B_lh_full = get_emb_matrix(args.model_b, "lm_head")
    B_ie = B_ie_full[bids]
    B_lh = B_lh_full[bids]
    del B_ie_full, B_lh_full

    print(f"[G3] embedding shapes: A_ie={A_ie.shape} A_lh={A_lh.shape} "
          f"B_ie={B_ie.shape} B_lh={B_lh.shape}")

    print(f"[G3] computing RDMs (cosine)...", flush=True)
    D = {
        "A_ie": cosine_rdm(A_ie),
        "A_lh": cosine_rdm(A_lh),
        "B_ie": cosine_rdm(B_ie),
        "B_lh": cosine_rdm(B_lh),
    }

    # 6 对 RDM RSA (4 跨模型 + 2 同模型 ie-lh)
    pairs = [
        ("A_ie", "B_ie"),  # 核心: OLMo ie vs pythia ie
        ("A_lh", "B_lh"),  # 核心: OLMo lh vs pythia lh
        ("A_ie", "B_lh"),  # 交叉
        ("A_lh", "B_ie"),  # 交叉
        ("A_ie", "A_lh"),  # 同模型 OLMo ie vs lh (与 C2 内 -0.106 对应)
        ("B_ie", "B_lh"),  # 同模型 pythia ie vs lh
    ]
    rsa = {f"{a}__{b}": rdm_rsa(D[a], D[b]) for a, b in pairs}

    # per-token cosine 同模型内
    per_tok = {
        "A_ie_vs_lh": per_tok_cos(A_ie, A_lh),
        "B_ie_vs_lh": per_tok_cos(B_ie, B_lh),
    }

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "n_words": n,
        "n_words_requested": args.n_words,
        "seed": args.seed,
        "rdm_rsa": rsa,
        "per_token_cosine_within_model": per_tok,
        "embedding_shapes": {
            "A_ie": list(A_ie.shape), "A_lh": list(A_lh.shape),
            "B_ie": list(B_ie.shape), "B_lh": list(B_lh.shape),
        },
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.outdir, "raw", "embeddings.npz"),
             words=np.array(ws, dtype=object),
             A_ie=A_ie, A_lh=A_lh, B_ie=B_ie, B_lh=B_lh,
             A_token_ids=np.array(aids), B_token_ids=np.array(bids))
    np.savez(os.path.join(args.outdir, "raw", "rdms.npz"), **D)

    print(f"\n[G3] === RDM Spearman RSA ===")
    for k, v in rsa.items():
        print(f"  {k:24s}: {v:+.4f}")
    print(f"\n[G3] === per-token cosine (sanity, 同模型 ie↔lh) ===")
    for k, v in per_tok.items():
        print(f"  {k:24s}: {v:+.4f}")
    print(f"\n[G3] summary -> {os.path.relpath(args.outdir)}/summary.json")


if __name__ == "__main__":
    main()
