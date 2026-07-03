# coding: utf-8
"""Bootstrap 95% CI for the partial-RSA self-specific delta (self - cross), per target.
复用 analyze_cross_model_specificity 的缓存(raw gen 向量 + 两模型嵌入矩阵), 不重训、不生成。
self / cross 两 cell 的测试词几乎不重叠(C4 各 cell 抽样不同), 故二者是独立估计:
对每个 cell 在其自身词上算 partial(spoken,target|other) 并各自 bootstrap(resample 自己的词),
再取 self_boot - cross_boot(独立差)得 delta 的 percentile 95% CI。与 0.16 的 bootstrap 同口径。
  python tools/bootstrap_partial_selfnet.py
"""
import json, glob
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from scipy.stats import rankdata

DATA = "results/data/C4_cross_model_target"
OLMO = "models/OLMo-2-0425-1B-Instruct"; PYT = "models/pythia-1.4b"

def load_emb(p):
    with safe_open(glob.glob(p + "/*.safetensors")[0], framework="pt") as f:
        k = [x for x in f.keys() if x.endswith("embed_tokens.weight") or x.endswith("embed_in.weight")][0]
        return f.get_tensor(k).float().numpy()
EMB = {OLMO: load_emb(OLMO), PYT: load_emb(PYT)}
TOK = {OLMO: AutoTokenizer.from_pretrained(OLMO), PYT: AutoTokenizer.from_pretrained(PYT)}

def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
def rdm_idx(Xu, idx):
    Xi = Xu[idx]; S = Xi @ Xi.T; return S[np.triu_indices(len(idx), 1)]
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c); Xc = np.c_[np.ones(len(ra)), rc]
    res = lambda y: y - Xc @ np.linalg.lstsq(Xc, y, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])
def load_pred(folder):
    pred = {}
    for ln in open(f"{DATA}/{folder}/raw/gen_ft_heldout.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        if d.get("parsed") and len(d["parsed"]) == 32: pred[d["word"]] = np.array(d["parsed"], float)
    return pred
def emb_row(tok, E, w):
    i = tok.encode(" " + str(w), add_special_tokens=False); return E[i[0]] if len(i) == 1 else None

def cell_boot(folder, tp, op, NB, rng):
    """partial(spoken,target|other) over this cell's own words; point + bootstrap array."""
    pred = load_pred(folder); ttok, tE = TOK[tp], EMB[tp]; otok, oE = TOK[op], EMB[op]
    ws = [w for w in pred if emb_row(ttok, tE, w) is not None and emb_row(otok, oE, w) is not None]
    P = unit(np.stack([pred[w] for w in ws])); T = unit(np.stack([emb_row(ttok, tE, w) for w in ws]))
    O = unit(np.stack([emb_row(otok, oE, w) for w in ws])); n = len(ws)
    f = lambda idx: partial(rdm_idx(P, idx), rdm_idx(T, idx), rdm_idx(O, idx))
    point = f(np.arange(n))
    boot = np.array([f(rng.integers(0, n, n)) for _ in range(NB)])
    return point, boot, n

# (self_folder, cross_folder, target, other, name)
TARGETS = [("olmo2-1b_self_ie_intersect_pythia_s0", "pythia-1.4b_FT_to_olmo2-1b_ie_s0", OLMO, PYT, "OLMo target"),
           ("pythia-1.4b_self_ie_intersect_olmo_s0", "olmo2-1b_FT_to_pythia-1.4b_ie_s0", PYT, OLMO, "pythia target")]

rng = np.random.default_rng(0); NB = 1000   # 与 §2/analyze_all 的 0.16 CI 同口径(1000 次)
print(f"partial-RSA(spoken,target|other), self - cross delta; bootstrap over words (NB={NB}, seed 0)\n")
deltas = []
for self_f, cross_f, tp, op, name in TARGETS:
    ps, bs, ns = cell_boot(self_f, tp, op, NB, rng)
    pc, bc, nc = cell_boot(cross_f, tp, op, NB, rng)
    d = bs - bc; deltas.append(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {name:<13} self={ps:+.3f}(n={ns}) cross={pc:+.3f}(n={nc})  delta={ps-pc:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={np.mean(d > 0):.3f}")
dm = np.mean(np.stack(deltas), axis=0); lo, hi = np.percentile(dm, [2.5, 97.5])
print(f"\n  two-target mean delta = {dm.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={np.mean(dm > 0):.3f}")
