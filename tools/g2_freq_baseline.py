"""G2 静态 baseline · 算 unigram 词频跟 L2 norm 的相关性.

回应 [[G族_判决性实验#G2]] 的 caveat: OLMo l2norm FT HE 0.776 远超 PCA target, 但
L2 norm 跟 unigram 词频强相关 (instruct 训练后 anisotropy: 高频词 norm 小 / 低频词
norm 大). LoRA 可能仅学了 "词义 → 词频 → norm" 这个间接 prior. 必须报这个 baseline.

静态分析: 用 NLTK Brown corpus (1.16M 词 token) 算 log unigram freq,
然后跟模型 input_embed 行的 L2 norm 算 Pearson / Spearman. 若 ≥ 0.5 →
OLMo 0.776 几乎完全可解释为词频回归; 若 ≈ 0 → 信号是别的来源.

用法:
    python tools/g2_freq_baseline.py \\
        --model models/OLMo-2-0425-1B-Instruct \\
        --pca_targets results/data/G2_physical_target/olmo2-1b_l2norm_s0/raw/pca_targets.npz \\
        --outdir results/data/G2_freq_baseline/olmo_s0
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def brown_log_freq_dict():
    """NLTK Brown corpus 上的 lowercase unigram log freq (smoothed +1).
    返回 dict[str → float], 词不在表里 fallback log(1/total) = ln 兜底.
    """
    from nltk.corpus import brown
    counter = Counter(w.lower() for w in brown.words())
    total = sum(counter.values())
    return counter, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pca_targets", required=True,
                    help="G2 l2norm cell 的 raw/pca_targets.npz, 取 test_words")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "raw"), exist_ok=True)

    # 词集
    npz = np.load(args.pca_targets, allow_pickle=True)
    test_words = [str(w) for w in npz["test_words"]]
    train_words = [str(w) for w in npz["train_words"]] if "train_words" in npz else []
    print(f"[G2-freq] test_words={len(test_words)}, train_words={len(train_words)}")

    # 加载模型 (input_embed → fp32 numpy)
    print(f"[G2-freq] loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    emb_matrix = m.get_input_embeddings().weight.detach().cpu().numpy().copy()
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Brown 词频
    print("[G2-freq] loading Brown corpus unigram counts ...", flush=True)
    counter, total = brown_log_freq_dict()
    print(f"[G2-freq] Brown corpus: {total:,} tokens, {len(counter):,} unique types")

    def cell_for(words):
        """返回 (l2_norms, log_freqs, brown_hits) 三个等长数组."""
        l2 = []
        lf = []
        hit = []
        for w in words:
            ids = tokenizer.encode(" " + w, add_special_tokens=False)
            if not ids:
                continue
            # 单 token 词 (G2 l2norm 用 input_embed[token_id], 跟 c1_lora.py 一致)
            tid = ids[0]
            l2.append(float(np.linalg.norm(emb_matrix[tid])))
            # smoothed log freq
            cnt = counter.get(w, 0)
            lf.append(float(np.log((cnt + 1) / (total + 1))))
            hit.append(1 if cnt > 0 else 0)
        return np.array(l2), np.array(lf), np.array(hit)

    def report(name, words):
        l2, lf, hit = cell_for(words)
        if len(l2) == 0:
            return None
        pr_p = float(pearsonr(lf, l2)[0])
        sr_p = float(spearmanr(lf, l2).statistic)
        out = {
            "n": len(l2),
            "brown_hit_rate": float(hit.mean()),
            "pearson_logfreq_vs_l2norm":  pr_p,
            "spearman_logfreq_vs_l2norm": sr_p,
            "l2norm_stats": {"mean": float(l2.mean()), "std": float(l2.std()),
                             "min": float(l2.min()), "max": float(l2.max())},
            "logfreq_stats": {"mean": float(lf.mean()), "std": float(lf.std()),
                              "min": float(lf.min()), "max": float(lf.max())},
        }
        print(f"\n[G2-freq] === {name} (n={len(l2)}) ===")
        print(f"  Brown hit rate                : {hit.mean()*100:.1f}%")
        print(f"  Pearson(log_freq, l2_norm)    : {pr_p:+.4f}")
        print(f"  Spearman(log_freq, l2_norm)   : {sr_p:+.4f}")
        print(f"  l2_norm range (mean ± std)    : {l2.mean():.3f} ± {l2.std():.3f}")
        print(f"  log_freq range (mean ± std)   : {lf.mean():.3f} ± {lf.std():.3f}")
        return out, l2, lf, hit

    r_te = report("held-out (G2 test_words)", test_words)
    r_tr = report("train (G2 train_words)", train_words) if train_words else None

    summary = {
        "model": args.model,
        "pca_targets": args.pca_targets,
        "brown_corpus": {"total_tokens": total, "unique_types": len(counter)},
        "held_out": r_te[0],
        "train": r_tr[0] if r_tr else None,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.outdir, "raw", "freq_norm.npz"),
             test_words=np.array(test_words, dtype=object),
             test_l2=r_te[1], test_logfreq=r_te[2], test_brown_hit=r_te[3],
             **({"train_words": np.array(train_words, dtype=object),
                 "train_l2": r_tr[1], "train_logfreq": r_tr[2],
                 "train_brown_hit": r_tr[3]} if r_tr else {}))

    print(f"\n[G2-freq] summary -> {os.path.relpath(args.outdir)}/summary.json")


if __name__ == "__main__":
    main()
