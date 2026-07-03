# coding: utf-8
"""Ogden 850(train_vocab=basic) 为何泛化更弱的几何检验:
训练词分布是否在 OLMo 输入嵌入空间里单调/集中(对照随机训练词≈全词表)。
结论: Ogden 词集中(meanPairCos 0.078)、不铺开(spread 6.5)、范数小(6.8);
随机训练词≈整张词表(0.053/9.3/9.5 vs 0.034/9.7/9.9); Ogden-PCA 对测试词几何上界仅 0.46
而模型只到 0.19 -> 跌幅来自"窄训练分布外推不到宽测试词", 非读出能力本身。"""
import glob
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
MP = "models/OLMo-2-0425-1B-Instruct"
def load_emb(p):
    with safe_open(glob.glob(p + "/*.safetensors")[0], framework="pt") as f:
        k = [x for x in f.keys() if x.endswith("embed_tokens.weight")][0]
        return f.get_tensor(k).float().numpy()
EMB = load_emb(MP); tok = AutoTokenizer.from_pretrained(MP)
def wid(w):
    i = tok.encode(" " + str(w), add_special_tokens=False); return i[0] if len(i) == 1 else None
def emb_of(ws): return EMB[[wid(w) for w in ws if wid(w) is not None]]
def words(cell, key):
    z = np.load(f"results/data/C1_lora_finetune/{cell}/raw/pca_targets.npz", allow_pickle=True)
    return [str(w) for w in z[key]]
ogden = emb_of(words("olmo2-1b_basic_real_s0", "train_words"))
rand = emb_of(words("olmo2-1b_random_s0", "train_words"))
test = emb_of(words("olmo2-1b_random_s0", "test_words"))
rng = np.random.default_rng(0); vocab = EMB[rng.choice(EMB.shape[0], 3000, replace=False)]
def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
def mpc(X): S = unit(X) @ unit(X).T; return float(S[np.triu_indices(len(X), 1)].mean())
def effdim(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False); ev = s ** 2; return float(ev.sum() ** 2 / (ev ** 2).sum())
def spread(X): return float(np.linalg.norm(X - X.mean(0), axis=1).mean())
def pca(X, k=32):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); return mu, Vt[:k]
def recon_r2(tr, te, k=32):
    mu, V = pca(tr, k); Tc = te - mu; pr = Tc @ V.T @ V; return float(1 - ((Tc - pr) ** 2).sum() / (Tc ** 2).sum())
print(f"{'set':<13}{'meanPairCos':>13}{'effDim':>9}{'spread':>9}{'meanNorm':>10}")
for k, X in [("Ogden-train", ogden), ("random-train", rand), ("test(broad)", test), ("vocab-3000", vocab)]:
    print(f"{k:<13}{mpc(X):>13.3f}{effdim(X):>9.1f}{spread(X):>9.3f}{np.linalg.norm(X, axis=1).mean():>10.3f}")
print(f"\nPCA-32 拟合训练词 -> 测试词重构R²: Ogden={recon_r2(ogden, test):.3f}  "
      f"random={recon_r2(rand, test):.3f}  (上界 test自拟合={recon_r2(test, test):.3f})")
