# coding: utf-8
"""跨模型 2x2 再分析(不重训, 复用现有 raw 生成向量 + 两模型嵌入矩阵):
(A) 多指标 battery: RSA / per-word cos / 识别 top-1,5,MRR —— 看哪种指标对"自我 vs 通用"更敏感。
(B) 偏 RSA 自我特异性: partial(spoken, target | other) —— 扣掉两模型共享几何后,
    说出的向量还能匹配多少"目标自身"的几何; 通用回归预测 ≈0, 真自我访问 >0。
结论(s0): RSA 自/目=0.30(目标主导); 识别率 0.67–0.92(自我抬头);
偏 RSA 自我增量(同目标 self−cross) ≈ +0.05, 两模型一致为正、但小。"""
import json, glob
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from scipy.stats import spearmanr, rankdata

DATA = "results/data/C4_cross_model_target"
OLMO = "models/OLMo-2-0425-1B-Instruct"; PYT = "models/pythia-1.4b"
def load_emb(p):
    with safe_open(glob.glob(p + "/*.safetensors")[0], framework="pt") as f:
        k = [x for x in f.keys() if x.endswith("embed_tokens.weight") or x.endswith("embed_in.weight")][0]
        return f.get_tensor(k).float().numpy()
EMB = {OLMO: load_emb(OLMO), PYT: load_emb(PYT)}
TOK = {OLMO: AutoTokenizer.from_pretrained(OLMO), PYT: AutoTokenizer.from_pretrained(PYT)}
CELLS = [("OLMo->OLMo (self)", "olmo2-1b_self_ie_intersect_pythia_s0", OLMO),
         ("pythia->OLMo(cross)", "pythia-1.4b_FT_to_olmo2-1b_ie_s0", OLMO),
         ("OLMo->pythia(cross)", "olmo2-1b_FT_to_pythia-1.4b_ie_s0", PYT),
         ("pythia->pythia(self)", "pythia-1.4b_self_ie_intersect_olmo_s0", PYT)]
def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
def rdm(X): S = unit(X) @ unit(X).T; return S[np.triu_indices(len(X), 1)]
def spear(a, b): return float(spearmanr(a, b).correlation)
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c); X = np.c_[np.ones(len(ra)), rc]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])
def ident(P, G):
    S = unit(P) @ unit(G).T; n = len(P); t1 = t5 = mrr = 0.0
    for i in range(n):
        r = int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1; t1 += r == 1; t5 += r <= 5; mrr += 1 / r
    return t1 / n, t5 / n, mrr / n
def rows(tok, E, ws):
    out = []
    for w in ws:
        i = tok.encode(" " + str(w), add_special_tokens=False); out.append(E[i[0]] if len(i) == 1 else None)
    return out
def load_cell(folder):
    z = np.load(f"{DATA}/{folder}/raw/pca_targets.npz", allow_pickle=True)
    gt32 = {str(w): v for w, v in zip(z["test_words"], z["gt_test_real"])}
    pred = {}
    for ln in open(f"{DATA}/{folder}/raw/gen_ft_heldout.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        if d.get("parsed") and len(d["parsed"]) == 32: pred[d["word"]] = np.array(d["parsed"], float)
    return gt32, pred

print("=== (A) battery: pred vs PCA-32 target ===")
print(f"{'cell':<21}{'n':>4}{'RSA':>7}{'cos':>7}{'top1':>7}{'top5':>7}{'MRR':>7}")
for name, folder, tp in CELLS:
    gt32, pred = load_cell(folder); ws = [w for w in gt32 if w in pred]
    P = np.stack([pred[w] for w in ws]); G = np.stack([gt32[w] for w in ws])
    cos = float(np.mean([np.dot(unit(P)[i], unit(G)[i]) for i in range(len(ws))]))
    t1, t5, mrr = ident(P, G)
    print(f"{name:<21}{len(ws):>4}{spear(rdm(P), rdm(G)):>7.3f}{cos:>7.3f}{t1:>7.3f}{t5:>7.3f}{mrr:>7.3f}")

print("\n=== (B) partial(spoken,target|other) over raw embeddings ===")
print(f"{'cell':<21}{'n':>4}{'RSA(p,tgt)':>11}{'RSA(tgt,oth)':>13}{'partial':>9}")
PR = {}
for name, folder, tp in CELLS:
    gt32, pred = load_cell(folder); ws0 = [w for w in gt32 if w in pred]
    tgt_tok, tgt_E = TOK[tp], EMB[tp]; op = OLMO if tp == PYT else PYT; oth_tok, oth_E = TOK[op], EMB[op]
    tg = rows(tgt_tok, tgt_E, ws0); ot = rows(oth_tok, oth_E, ws0)
    keep = [i for i in range(len(ws0)) if tg[i] is not None and ot[i] is not None]
    P = np.stack([pred[ws0[i]] for i in keep]); T = np.stack([tg[i] for i in keep]); O = np.stack([ot[i] for i in keep])
    rp, rt, ro = rdm(P), rdm(T), rdm(O); PR[name] = partial(rp, rt, ro)
    print(f"{name:<21}{len(keep):>4}{spear(rp, rt):>11.3f}{spear(rt, ro):>13.3f}{PR[name]:>9.3f}")
print("\n同目标 self vs cross 的 partial(spoken,target|other):")
print(f"  OLMo  : self={PR['OLMo->OLMo (self)']:.3f} cross={PR['pythia->OLMo(cross)']:.3f} diff={PR['OLMo->OLMo (self)']-PR['pythia->OLMo(cross)']:+.3f}")
print(f"  pythia: self={PR['pythia->pythia(self)']:.3f} cross={PR['OLMo->pythia(cross)']:.3f} diff={PR['pythia->pythia(self)']-PR['OLMo->pythia(cross)']:+.3f}")
