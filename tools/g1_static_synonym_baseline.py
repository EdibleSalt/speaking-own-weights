"""G1 静态 baseline · 算 emb(word) ↔ emb(synonym) 在模型自身嵌入空间里的几何相似度.

回应 [[G族_判决性实验#G1]] 的 caveat: G1 FT synonym HE 0.25 可能完全由"同义词嵌入
跟原词嵌入静态近邻"解释 (语义流形耦合). 必须报这个静态量, 否则 reviewer 会问
"你的 0.25 是不是仅仅因为同义词嵌入跟原词本来就近".

静态分析, 无 LoRA, 无 inference. 用 c1_lora.py 落盘的 pca_targets.npz 拿 G1 cell 的
held-out 词列表, 配上 WordNet 词典 (materials/G1_queries/{tag}_synonym.json), 算:
  - per-pair cosine(emb(word), emb(synonym_avg))  平均
  - RDM_Spearman(RDM(emb_word), RDM(emb_synonym_avg))  几何结构对齐

synonym 可能多 token → 取 token embeddings mean pooling (跟 input_embed 同矩阵).
没在词典里的 word fallback 自身 (跟 G1 eval 时 fallback 行为一致).

用法:
    python tools/g1_static_synonym_baseline.py \\
        --model models/OLMo-2-0425-1B-Instruct \\
        --pca_targets results/data/G1_non_activation_query/olmo2-1b_self_s0/raw/pca_targets.npz \\
        --query_dict materials/G1_queries/olmo_synonym.json \\
        --outdir results/data/G1_static_baseline/olmo_synonym_s0
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


def encode_to_embedding(tokenizer, emb_matrix, text):
    """编码 text 为 token ids, 取对应 input_embed 行 mean pooling. 返回 (d,) 或 None."""
    ids = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        return None
    return emb_matrix[ids].mean(axis=0)


def cosine_rdm(X):
    """X: (n, d) -> (n, n) 1 - cos sim, squareform of pdist."""
    return squareform(pdist(X, metric="cosine"))


def upper_tri(D):
    n = D.shape[0]
    return D[np.triu_indices(n, k=1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pca_targets", required=True,
                    help="G1 cell raw/pca_targets.npz, 取 test_words 列表")
    ap.add_argument("--query_dict", required=True,
                    help="materials/G1_queries/{tag}_synonym.json, {word: synonym_phrase}")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "raw"), exist_ok=True)

    # 词集 + 词典
    npz = np.load(args.pca_targets, allow_pickle=True)
    test_words = [str(w) for w in npz["test_words"]]
    queries = json.load(open(args.query_dict, encoding="utf-8"))
    covered = [w for w in test_words if w in queries]
    print(f"[G1-static] test_words={len(test_words)}, 词典覆盖={len(covered)} "
          f"({100*len(covered)/len(test_words):.1f}%)")

    # 加载模型 (仅取 input_embed 矩阵 → fp32 numpy)
    print(f"[G1-static] loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    emb_matrix = m.get_input_embeddings().weight.detach().cpu().numpy().copy()
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 对每个 covered word, 抽 emb(word) 和 emb(synonym) (synonym 多 token 取 mean)
    pairs = []
    for w in covered:
        e_w = encode_to_embedding(tokenizer, emb_matrix, w)
        e_s = encode_to_embedding(tokenizer, emb_matrix, queries[w])
        if e_w is None or e_s is None:
            continue
        pairs.append((w, queries[w], e_w, e_s))
    print(f"[G1-static] 有效 pair = {len(pairs)} (token 编码成功)")

    words = [p[0] for p in pairs]
    syns  = [p[1] for p in pairs]
    E_w = np.stack([p[2] for p in pairs])  # (n, d)
    E_s = np.stack([p[3] for p in pairs])

    # per-pair cosine(emb_word, emb_synonym)
    def per_tok_cos(X, Y):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        return (Xn * Yn).sum(axis=1)  # (n,)
    per_pair_cos = per_tok_cos(E_w, E_s)

    # RDM RSA: cosine RDM of E_w vs cosine RDM of E_s
    D_w = cosine_rdm(E_w)
    D_s = cosine_rdm(E_s)
    rdm_rsa = float(spearmanr(upper_tri(D_w), upper_tri(D_s)).statistic)

    summary = {
        "model": args.model,
        "pca_targets": args.pca_targets,
        "query_dict": args.query_dict,
        "n_test_words": len(test_words),
        "n_covered_by_dict": len(covered),
        "n_valid_pairs": len(pairs),
        "per_pair_cosine": {
            "mean":   float(per_pair_cos.mean()),
            "median": float(np.median(per_pair_cos)),
            "std":    float(per_pair_cos.std()),
            "min":    float(per_pair_cos.min()),
            "max":    float(per_pair_cos.max()),
        },
        "rdm_spearman_rsa_word_vs_synonym": rdm_rsa,
        "embedding_dim": int(E_w.shape[1]),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.outdir, "raw", "embeddings.npz"),
             words=np.array(words, dtype=object),
             synonyms=np.array(syns, dtype=object),
             E_word=E_w, E_synonym=E_s,
             per_pair_cosine=per_pair_cos)

    print(f"\n[G1-static] === 结果 ===")
    print(f"  n_valid_pairs                : {len(pairs)}")
    print(f"  per-pair cosine mean         : {per_pair_cos.mean():+.4f}")
    print(f"  per-pair cosine median       : {np.median(per_pair_cos):+.4f}")
    print(f"  RDM Spearman RSA (word ↔ syn): {rdm_rsa:+.4f}")
    print(f"\n[G1-static] summary -> {os.path.relpath(args.outdir)}/summary.json")


if __name__ == "__main__":
    main()
