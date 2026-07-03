"""诊断 P4 layer 0 RSA 未达 1.0 的根因（C0-base 延伸 ② 的前置分析）。

理论 hypothesis:
  1. (dominant) underdetermined: n_train=151 << d=1280/2048,
     col(X_train) 至多 151 维 → test 在 train 子空间外的方向无法预测;
  2. (secondary) ridge λ 缩水: λ_min=1e0 在弱方向上有显著缩水.

实验:
  - 直接读 raw/hidden_states.npz + real_embeddings.npz, 不重跑模型
  - 对每个 P4 模型, layer 0 上跑:
    * identity baseline: RSA(X[test] vs Y_c[test]) 不经 probe -> 真上界
    * ridge λ ∈ {1e0, 1e-1, 1e-2, 1e-4, 1e-6}
    * OLS via pinv (λ→0 极限)
    * (可选) 不同 n_train 比例: 0.6/0.8/0.95 看子空间限制
  - 输出每个方法的: RSA / mean per-word cosine
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def cosine_sim_matrix(X):
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def upper_tri(M):
    return M[np.triu_indices_from(M, k=1)]


def rsa(P, T):
    rp = upper_tri(cosine_sim_matrix(P))
    rt = upper_tri(cosine_sim_matrix(T))
    return float(spearmanr(rp, rt)[0])


def per_word_cos(P, T):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(Pn * Tn, axis=1)))


def fit_predict(Xtr_s, Ytr_c, Xte_s, lam):
    d = Xtr_s.shape[1]
    if lam == 0:
        W = np.linalg.pinv(Xtr_s) @ Ytr_c  # OLS via pinv
    else:
        A = Xtr_s.T @ Xtr_s + lam * np.eye(d)
        W = np.linalg.solve(A, Xtr_s.T @ Ytr_c)
    return Xte_s @ W


def analyze_model(tag):
    base = os.path.join(os.path.dirname(HERE), "results", "data", "C0_probe_ceiling", tag, "raw")
    H = np.load(os.path.join(base, "hidden_states.npz"))
    E = np.load(os.path.join(base, "real_embeddings.npz"))
    tr = H["train_idx"]
    te = H["test_idx"]
    target = E["real"]
    Ymu = E["Ymu"]
    Y_c = target - Ymu  # 全集中心化

    X = H["layer_0"]  # P4 layer 0
    Xtr_raw = X[tr]
    Xte_raw = X[te]
    Ytr_c = Y_c[tr]
    Yte_c = Y_c[te]

    d = X.shape[1]
    n_tr = len(tr)
    n_te = len(te)

    print(f"\n{'='*72}")
    print(f"{tag}  (d={d}, n_train={n_tr}, n_test={n_te}, "
          f"子空间覆盖率 n_tr/d={n_tr/d:.1%})")
    print(f"{'='*72}")

    # ----- baseline 0: identity (X[test] 中心化 vs Y_c[test]) -----
    # 用 train 的均值中心化 X（与 probe 的 standardize 路径一致）
    Xte_centered = Xte_raw - Xtr_raw.mean(0)
    rsa_id = rsa(Xte_centered, Yte_c)
    cos_id = per_word_cos(Xte_centered, Yte_c)
    print(f"  identity (X_test centered, no probe):  RSA={rsa_id:.4f}  cos={cos_id:.4f}")
    print(f"    -> 真上界: X 自带的 Y 信息. <1 说明 X≠Y (含位置/标准化漂移).")

    # ----- ridge/OLS sweep -----
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
    Xtr_s = (Xtr_raw - mu) / sd
    Xte_s = (Xte_raw - mu) / sd

    print()
    print(f"  {'method':16s} {'RSA':>8s} {'cos':>8s}")
    for lam in [1e0, 1e-1, 1e-2, 1e-4, 1e-6]:
        Pte = fit_predict(Xtr_s, Ytr_c, Xte_s, lam)
        print(f"  ridge λ={lam:<7g} {rsa(Pte, Yte_c):8.4f} {per_word_cos(Pte, Yte_c):8.4f}")
    Pte_ols = fit_predict(Xtr_s, Ytr_c, Xte_s, 0)
    print(f"  {'pinv (OLS)':16s} {rsa(Pte_ols, Yte_c):8.4f} {per_word_cos(Pte_ols, Yte_c):8.4f}")

    # ----- 加大 n_train 测子空间限制 -----
    all_idx = np.concatenate([tr, H["val_idx"]])  # 把 val 并入 train, 50 + 151 = 201
    rng = np.random.default_rng(0)
    rng.shuffle(all_idx)
    print()
    print(f"  {'n_train':>8s} {'RSA(λ=1e-2)':>13s} {'RSA(pinv)':>11s}")
    for frac in [0.6, 0.8, 0.95]:
        n_use = int(frac * len(all_idx))
        tr2 = all_idx[:n_use]
        Xtr2_raw = X[tr2]
        Ytr2_c = Y_c[tr2]
        mu2, sd2 = Xtr2_raw.mean(0), Xtr2_raw.std(0) + 1e-6
        Xtr2_s = (Xtr2_raw - mu2) / sd2
        Xte2_s = (Xte_raw - mu2) / sd2
        Pte2_r = fit_predict(Xtr2_s, Ytr2_c, Xte2_s, 1e-2)
        Pte2_o = fit_predict(Xtr2_s, Ytr2_c, Xte2_s, 0)
        print(f"  {n_use:>8d} {rsa(Pte2_r, Yte_c):>13.4f} {rsa(Pte2_o, Yte_c):>11.4f}")
    print(f"    -> n_train 趋近 d 时 RSA 应趋近 identity 上界.")


def main():
    for tag in ["pythia-1.4b__P4", "gpt2-large__P4"]:
        analyze_model(tag)

    print()
    print("="*72)
    print("综合判断模板:")
    print("- identity ≈ 1 + ridge λ→0 也 ≈ 1: probe 是损失源 (改 OLS / 加 λ→0 可解)")
    print("- identity ≈ 1 + ridge λ→0 < 1   : 子空间限制 (n_train<<d), OLS 帮不大")
    print("- identity < 1                   : X ≠ Y 本身 (位置/层处理引入了非恒等)")
    print("="*72)


if __name__ == "__main__":
    main()
