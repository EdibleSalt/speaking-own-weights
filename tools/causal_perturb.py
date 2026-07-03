# coding: utf-8
"""C7 方案1+2: 因果扰动单行输入嵌入, 看冻结读出是否跟随(纯推理、不训练)。
  python tools/causal_perturb.py [olmo|pythia]      (环境变量 C7_N 限词数, 默认120; 冒烟测试用 C7_N=3)

对每个测试词 w: δ = E_in[w2] − E_in[w](w2=另一随机测试词); 按 {0,0.5,1,2}×δ 改 E_in[w] 再还原;
两种提示 (a) 含 w / (b) 同义词(w 不出现, 复用 G1 clean 集); 测报出向量(PCA-32)沿 δ_proj 的位移。
跟随系数 k = Δ_along(s=1)/‖δ_proj‖: k≈1 读了被扰动激活, k≈0 无视。复用 c1_lora.readout。
预注册预测: (a) k>0 显著、随幅度增; (b) k 的 CI 含 0(机制上 w 不在提示→该行不被读)。
"""
import sys, os, json
sys.path.insert(0, "src")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from c1_lora import readout

MODELS = {
    "olmo": ("models/OLMo-2-0425-1B-Instruct",
             "results/data/C1_lora_finetune/olmo2-1b_random_s0",
             "results/data/G1_non_activation_query/olmo2-1b_synonym_clean_s0"),
    "pythia": ("models/pythia-1.4b",
               "results/data/C1_lora_finetune/pythia-1.4b_random_s0",
               "results/data/G1_non_activation_query/pythia-1.4b_synonym_clean_s0"),
}
which = sys.argv[1] if len(sys.argv) > 1 else "olmo"
MODEL, C1, G1 = MODELS[which]
OUT = "results/data/C7_causal_perturb"; os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtp = torch.bfloat16 if torch.cuda.is_available() else torch.float32
SCALES = [0.0, 0.5, 1.0, 2.0]
CAP = int(os.environ.get("C7_N", "120"))

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

z = np.load(f"{C1}/raw/pca_targets.npz", allow_pickle=True)
mu, comps = z["mu"], z["comps"]            # comps (32,2048)

syn = {}
for ln in open(f"{G1}/raw/gen_ft_heldout.jsonl", encoding="utf-8"):
    d = json.loads(ln)
    if d.get("query") and d["query"] != d["word"]:
        syn[d["word"]] = d["query"]

def single(w): return len(tok.encode(" " + w, add_special_tokens=False)) == 1
def tid(w):    return tok.encode(" " + w, add_special_tokens=False)[0]

c1train = set(str(w) for w in z["train_words"].tolist())   # 排除 C1 训练词(读出已记忆), 保证泛化
cand = [w for w in syn if single(w) and w not in c1train]   # G1 同义词词(都有 synonym)∩ 单token ∩ 非C1训练
rng = np.random.default_rng(0)
if len(cand) > CAP:
    cand = [cand[i] for i in sorted(rng.choice(len(cand), CAP, replace=False))]
W = [(w, tid(w)) for w in cand]
print(f"[{which}] n_words={len(W)} device={device} (syn_map={len(syn)})", flush=True)

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtp).to(device)
model = PeftModel.from_pretrained(base, f"{C1}/adapter/final").to(device)
model.eval()
Wemb = model.get_input_embeddings().weight       # (V,H), frozen
EMBnp = Wemb.detach().float().cpu().numpy()

drng = np.random.default_rng(1)
deltas = {}
for (w, t) in W:
    j = int(drng.integers(0, len(W)))
    while W[j][1] == t: j = int(drng.integers(0, len(W)))
    deltas[w] = (EMBnp[W[j][1]] - EMBnp[t], W[j][0])   # raw δ (2048,), w2

rows = []
for (w, t) in W:
    dvec, w2 = deltas[w]
    dproj = dvec @ comps.T
    orig = Wemb.data[t].clone()
    dtorch = torch.tensor(dvec, dtype=Wemb.dtype, device=Wemb.device)
    rec = {"word": w, "tid": int(t), "w2": w2, "dproj_norm": float(np.linalg.norm(dproj)), "a": {}, "b": {}}
    for s in SCALES:
        with torch.no_grad():
            Wemb.data[t] = orig + s * dtorch
        ra, _ = readout(model, tok, device, [(w, t)], 32, 2, 200, 1, query_for_word=None)
        rb, _ = readout(model, tok, device, [(w, t)], 32, 2, 200, 1, query_for_word={w: syn[w]})
        rec["a"][str(s)] = (ra[w].tolist() if ra[w] is not None else None)
        rec["b"][str(s)] = (rb[w].tolist() if rb[w] is not None else None)
    with torch.no_grad():
        Wemb.data[t] = orig
    rows.append(rec)
    if len(rows) % 20 == 0: print(f"  {len(rows)}/{len(W)}", flush=True)

def along(rec, cond):
    P0 = rec[cond]["0.0"]
    if P0 is None: return None
    P0 = np.array(P0); dvec, _ = deltas[rec["word"]]; dproj = dvec @ comps.T
    u = dproj / (np.linalg.norm(dproj) + 1e-9)
    return {str(s): (float((np.array(rec[cond][str(s)]) - P0) @ u) if rec[cond][str(s)] is not None else None) for s in SCALES}

brng = np.random.default_rng(2)
summary = {"model": which, "n": len(rows), "scales": SCALES}
for cond in ["a", "b"]:
    per_scale = {str(s): [] for s in SCALES}; ks = []; parse = {str(s): 0 for s in SCALES}
    for rec in rows:
        for s in SCALES:
            if rec[cond][str(s)] is not None: parse[str(s)] += 1
        a = along(rec, cond)
        if a is None: continue
        for s in SCALES:
            if a[str(s)] is not None: per_scale[str(s)].append(a[str(s)])
        if a["1.0"] is not None and rec["dproj_norm"] > 1e-6:
            ks.append(a["1.0"] / rec["dproj_norm"])
    ks = np.array(ks)
    bo = np.array([ks[brng.integers(0, len(ks), len(ks))].mean() for _ in range(1000)]) if len(ks) else np.array([])
    summary[cond] = {
        "mean_along_by_scale": {s: (round(float(np.mean(v)), 4) if v else None) for s, v in per_scale.items()},
        "k_mean": round(float(ks.mean()), 4) if len(ks) else None,
        "k_CI": [round(float(np.percentile(bo, 2.5)), 4), round(float(np.percentile(bo, 97.5)), 4)] if len(bo) else None,
        "k_P_gt0": round(float(np.mean(bo > 0)), 3) if len(bo) else None,
        "n_k": int(len(ks)),
        "parse_rate_by_scale": {s: round(parse[s] / len(rows), 3) for s in parse},
    }
json.dump({"summary": summary, "rows": rows}, open(f"{OUT}/{which}_perturb.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
