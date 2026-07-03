# coding: utf-8
"""路4严格版(一进程跑一格, 避免多模型累积导致的显存/内存死锁):
  python tools/matched_eval.py <cell_idx 0..3>
让该 cell 的 adapter 在同一批 held-out 词 W 上生成, 算匹配识别率(top1/5/MRR), 写 _matched_cell{idx}.json。
W 由 4 个 cell 的 (并test - 并train) 确定性构造, 4 个进程一致。复用 src/c1_lora.py 的 readout/project。"""
import sys, os, json, glob
sys.path.insert(0, "src")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors import safe_open
from c1_lora import readout, project

DATA = "results/data/C4_cross_model_target"
OLMO = "models/OLMo-2-0425-1B-Instruct"; PYT = "models/pythia-1.4b"
CELLS = [("OLMo->OLMo(self)",    "olmo2-1b_self_ie_intersect_pythia_s0"),
         ("OLMo->pythia(cross)", "olmo2-1b_FT_to_pythia-1.4b_ie_s0"),
         ("pythia->OLMo(cross)", "pythia-1.4b_FT_to_olmo2-1b_ie_s0"),
         ("pythia->pythia(self)","pythia-1.4b_self_ie_intersect_olmo_s0")]
idx = int(sys.argv[1]); name, folder = CELLS[idx]
device = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def mktok(p):
    t = AutoTokenizer.from_pretrained(p)
    if t.pad_token is None: t.pad_token = t.eos_token
    t.padding_side = "left"; return t
tok_o, tok_p = mktok(OLMO), mktok(PYT)

def load_npz(f): return np.load(f"{DATA}/{f}/raw/pca_targets.npz", allow_pickle=True)
tr = set(); te = set()
for _, f in CELLS:
    z = load_npz(f); tr |= {str(w) for w in z["train_words"]}; te |= {str(w) for w in z["test_words"]}
def single(t, w): return len(t.encode(" " + w, add_special_tokens=False)) == 1
W = [w for w in sorted(te - tr) if single(tok_o, w) and single(tok_p, w)]
rng = np.random.default_rng(0)
if len(W) > 150: W = [W[i] for i in sorted(rng.choice(len(W), 150, replace=False))]

d = json.load(open(f"{DATA}/{folder}/summary.json", encoding="utf-8"))
def norm(p): return "models/" + p.replace("\\", "/").rstrip("/").split("/")[-1]
reader = norm(d["model"]); target = norm(d["target_model"]) if d.get("target_model") else norm(d["model"])
z = load_npz(folder); mu = z["mu"]; comps = z["comps"]
rtok = tok_o if reader == OLMO else tok_p
ttok = tok_o if target == OLMO else tok_p

def load_emb(p):
    with safe_open(glob.glob(p + "/*.safetensors")[0], framework="pt") as f:
        k = [x for x in f.keys() if x.endswith("embed_tokens.weight") or x.endswith("embed_in.weight")][0]
        return f.get_tensor(k).float().numpy()
temb = load_emb(target)
gt = {w: project(temb[ttok.encode(" " + w, add_special_tokens=False)[0]], mu, comps) for w in W}
del temb

print(f"[{idx}] {name} reader={reader} target={target} |W|={len(W)} device={device}", flush=True)
base = AutoModelForCausalLM.from_pretrained(reader, dtype=dt).to(device)
peft = PeftModel.from_pretrained(base, f"{DATA}/{folder}/adapter/final").to(device)
pairs = [(w, rtok.encode(" " + w, add_special_tokens=False)[0]) for w in W]
ro, raw = readout(peft, rtok, device, pairs, 32, 2, 200, 16)

def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
keep = [w for w in W if ro.get(w) is not None]
P = np.stack([np.asarray(ro[w], float) for w in keep]); G = np.stack([gt[w] for w in keep])
S = unit(P) @ unit(G).T; n = len(keep); t1 = t5 = mrr = 0.0
for i in range(n):
    r = int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1; t1 += r == 1; t5 += r <= 5; mrr += 1 / r
res = dict(name=name, n=n, top1=t1 / n, top5=t5 / n, mrr=mrr / n, Wsize=len(W))
json.dump(res, open(f"{DATA}/_matched_cell{idx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[{idx}] {name}: n={n} top1={t1/n:.3f} top5={t5/n:.3f} MRR={mrr/n:.3f} -> _matched_cell{idx}.json", flush=True)
