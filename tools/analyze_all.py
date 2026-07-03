"""归纳分析器 · 把已跑完的 cells 从 summary.json + raw/ 重算成"可写进记录"的权威数字。

设计目标: **不可崩** —— 每个分析块独立 try/except, 任一块失败只记 warning, 不影响其它块
和隔夜批跑. 全程**不加载任何模型**(只读 summary.json + raw/*.npz/*.jsonl), 纯 CPU 秒级.

产出 (项目根):
  - _analysis_report.md   人读: 权威数字表 + 文档对账 + 多 seed 聚合 + bootstrap CI + 偏相关
  - _analysis.json        机读: 同内容结构化

回应审计:
  - c11/c9/c7: 文档 vs summary.json 对账 (claimed-vs-actual)
  - c7stat-1/4: 用 bootstrap 95% CI 取代贴地板的 p=0.0005
  - c7stat-3 / c13scope-3: 多 seed mean±std (跑完 s1/s2 后自动纳入)
  - c7stat-6 / c5g-9: G2 偏相关 partial_r(pred, l2 | logfreq) —— 真正剔除词频后的残余物理访问信号

用法: .venv\\Scripts\\python.exe tools/analyze_all.py
"""
import os, sys, json, glob, math, traceback
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "results", "data")
RNG = np.random.default_rng(0)

# ---- 文档里"声称"的关键数字 (来自审计, 用于对账; 实测以 summary.json 为准) ----
CLAIMED = {
    "C1_lora_finetune/olmo2-1b_random_s0":              ("HE", 0.452),
    "C1_lora_finetune/olmo2-1b_basic_real_s0":          ("HE", 0.329),
    "C2_hard_target/olmo2-1b_random_real_lmhead_s0":    ("HE", 0.399),
    "C1_lora_finetune/olmo2-1b_random_r64_ep30_n800_s0":("HE", 0.595),
    "G1_non_activation_query/olmo2-1b_synonym_s0":      ("HE", 0.251),
    "G2_physical_target/olmo2-1b_l2norm_s0":            ("HE", 0.776),
}


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def he_of(d):
    """从 summary.json 取 held-out 主指标 (RSA 或 d=1 的 scalar pearson)."""
    if not isinstance(d, dict):
        return None
    fh = d.get("ft_heldout") or {}
    return fh.get("rsa", fh.get("scalar_pearson_r"))


def tr_of(d):
    fh = d.get("ft_trained") or {}
    return fh.get("rsa", fh.get("scalar_pearson_r"))


# ---------- raw 重算 RSA + bootstrap CI (不加载模型) ----------
def _read_pred(cell, split="ft_heldout"):
    """从 gen_{split}.jsonl 读 parsed 预测, 对齐到 pca_targets.npz 的 test/train words.
    返回 (P[n,d], G[n,d], words[n]) 仅含解析成功的词. 失败返回 None."""
    raw = os.path.join(cell, "raw")
    npz = os.path.join(raw, "pca_targets.npz")
    jl = os.path.join(raw, f"gen_{split}.jsonl")
    if not (os.path.exists(npz) and os.path.exists(jl)):
        return None
    z = np.load(npz, allow_pickle=True)
    is_test = "heldout" in split
    words = list(z["test_words"] if is_test else z["train_words"])
    G = z["gt_test_real"] if is_test else z["gt_train_real"]
    widx = {str(w): i for i, w in enumerate(words)}
    d = G.shape[1]
    pred = {}
    with open(jl, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            v = o.get("parsed")
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            if v.shape[0] >= d:
                pred[str(o["word"])] = v[:d]
    rows_p, rows_g = [], []
    for w, i in widx.items():
        if w in pred:
            rows_p.append(pred[w]); rows_g.append(G[i])
    if len(rows_p) < 5:
        return None
    return np.array(rows_p), np.array(rows_g), len(rows_p)


def _rsa(P, G):
    from scipy.stats import spearmanr, pearsonr
    if P.shape[1] == 1:
        return float(pearsonr(P[:, 0], G[:, 0])[0])
    def utri_cos(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        iu = np.triu_indices(S.shape[0], k=1)
        return S[iu]
    return float(spearmanr(utri_cos(P), utri_cos(G))[0])


def _bootstrap_ci(P, G, nboot=1000):
    n = P.shape[0]
    vals = []
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        try:
            vals.append(_rsa(P[idx], G[idx]))
        except Exception:
            pass
    if not vals:
        return None
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())


# =================== 主流程 ===================
def main():
    report = {"blocks": {}}
    md = ["# 归纳分析报告 (analyze_all.py)\n",
          "> 全部数字以 summary.json / raw 实测为准; 不加载模型, 纯重算.\n"]

    # ---------- BLOCK 1: 全量权威数字表 + 文档对账 ----------
    try:
        rows = []
        for sj in glob.glob(os.path.join(DATA, "**", "summary.json"), recursive=True):
            cell = os.path.dirname(sj)
            rel = os.path.relpath(cell, DATA).replace("\\", "/")
            d = _load(sj)
            if d is None:
                continue
            he, tr = he_of(d), tr_of(d)
            rp = (d.get("ft_heldout") or {}).get("rsa_p")
            rows.append({"cell": rel, "HE": he, "TR": tr, "rsa_p": rp,
                         "base_HE": (d.get("base_heldout") or {}).get("rsa")})
        rows.sort(key=lambda r: r["cell"])
        report["blocks"]["authoritative"] = rows
        md.append(f"\n## 1. 权威数字表 ({len(rows)} cells)\n")
        md.append("| cell | HE | TR | base_HE | rsa_p |")
        md.append("|---|---|---|---|---|")
        for r in rows:
            def f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "—"
            md.append(f"| {r['cell']} | {f(r['HE'])} | {f(r['TR'])} | {f(r['base_HE'])} | {f(r['rsa_p'])} |")

        # 对账
        md.append("\n## 1b. 文档 vs 实测对账 (claimed-vs-actual)\n")
        md.append("| cell | 文档声称 | 实测 | 差 | 状态 |")
        md.append("|---|---|---|---|---|")
        diffs = []
        amap = {r["cell"]: r for r in rows}
        for cell, (k, claimed) in CLAIMED.items():
            actual = amap.get(cell, {}).get("HE")
            if actual is None:
                md.append(f"| {cell} | {claimed} | 缺失 | — | ⚠️ 缺 |"); continue
            dd = actual - claimed
            flag = "✅一致" if abs(dd) < 0.01 else ("❗反转/大偏" if abs(dd) > 0.08 else "⚠️偏")
            md.append(f"| {cell} | {claimed} | {actual:.4f} | {dd:+.4f} | {flag} |")
            diffs.append({"cell": cell, "claimed": claimed, "actual": actual, "diff": dd})
        report["blocks"]["reconcile"] = diffs
    except Exception:
        md.append("\n## 1. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 2: 多 seed 聚合 mean±std ----------
    try:
        import re
        groups = {}
        for r in report["blocks"].get("authoritative", []):
            m = re.match(r"^(.*)_s(\d+)$", r["cell"])
            if not m or r["HE"] is None:
                continue
            groups.setdefault(m.group(1), []).append((int(m.group(2)), r["HE"]))
        multi = {g: v for g, v in groups.items() if len(v) >= 2}
        md.append(f"\n## 2. 多 seed 聚合 (≥2 seed 的组, 共 {len(multi)})\n")
        md.append("| 组 | seeds | HE 值 | mean | std | CV% |")
        md.append("|---|---|---|---|---|---|")
        agg = []
        for g, v in sorted(multi.items()):
            v.sort()
            hes = [x[1] for x in v]
            mean = float(np.mean(hes)); std = float(np.std(hes, ddof=1)) if len(hes) > 1 else 0.0
            cv = 100 * std / abs(mean) if mean else float("nan")
            seeds = ",".join(str(x[0]) for x in v)
            md.append(f"| {g} | {seeds} | {', '.join(f'{h:.3f}' for h in hes)} | {mean:.4f} | {std:.4f} | {cv:.1f} |")
            agg.append({"group": g, "seeds": seeds, "mean": mean, "std": std, "cv_pct": cv, "values": hes})
        report["blocks"]["multiseed"] = agg
    except Exception:
        md.append("\n## 2. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 3: bootstrap 95% CI (取代地板 p) ----------
    try:
        KEY = glob.glob(os.path.join(DATA, "C1_lora_finetune", "*", "")) + \
              glob.glob(os.path.join(DATA, "C2_hard_target", "*", "")) + \
              glob.glob(os.path.join(DATA, "C3_deep_hidden", "*", "")) + \
              glob.glob(os.path.join(DATA, "C4_cross_model_target", "*", "")) + \
              glob.glob(os.path.join(DATA, "G1_non_activation_query", "*", "")) + \
              glob.glob(os.path.join(DATA, "G2_physical_target", "*", ""))
        md.append(f"\n## 3. Bootstrap 95% CI on HE (从 raw 重算, nboot=1000)\n")
        md.append("| cell | n_kept | HE(报告) | HE(重算) | 95% CI |")
        md.append("|---|---|---|---|---|")
        ci_rows = []
        for cell in sorted(KEY):
            cell = cell.rstrip("\\/")
            rel = os.path.relpath(cell, DATA).replace("\\", "/")
            pr = _read_pred(cell, "ft_heldout")
            d = _load(os.path.join(cell, "summary.json"))
            reported = he_of(d) if d else None
            if pr is None:
                continue
            P, G, n = pr
            recomp = _rsa(P, G)
            ci = _bootstrap_ci(P, G)
            cistr = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            rep = f"{reported:.4f}" if isinstance(reported, (int, float)) else "—"
            md.append(f"| {rel} | {n} | {rep} | {recomp:.4f} | {cistr} |")
            ci_rows.append({"cell": rel, "n": n, "reported": reported, "recomputed": recomp,
                            "ci_lo": ci[0] if ci else None, "ci_hi": ci[1] if ci else None})
        report["blocks"]["bootstrap_ci"] = ci_rows
    except Exception:
        md.append("\n## 3. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 4: G2 偏相关 partial_r(pred, l2 | logfreq) ----------
    try:
        from scipy.stats import pearsonr
        freq = None
        try:
            from nltk.corpus import brown
            from collections import Counter
            c = Counter(w.lower() for w in brown.words())
            freq = c
        except Exception as e:
            md.append(f"\n## 4. G2 偏相关\n> ⚠️ 词频源(nltk brown)不可用, 跳过: {e}\n")
        if freq is not None:
            md.append("\n## 4. G2 偏相关 partial_r(LoRA_pred, L2norm | log_freq)\n")
            md.append("> 审计 c7stat-6: 0.776−0.740≈0.04 不是偏相关. 这才是剔除词频后的残余物理访问信号.\n")
            md.append("| cell | n | r(pred,l2) | r(pred,logf) | r(l2,logf) | **partial(pred,l2|logf)** |")
            md.append("|---|---|---|---|---|---|")
            g2rows = []
            for cell in sorted(glob.glob(os.path.join(DATA, "G2_physical_target", "*_l2norm_s*", ""))):
                cell = cell.rstrip("\\/")
                rel = os.path.relpath(cell, DATA).replace("\\", "/")
                pr = _read_pred(cell, "ft_heldout")
                if pr is None:
                    continue
                P, G, n = pr
                z = np.load(os.path.join(cell, "raw", "pca_targets.npz"), allow_pickle=True)
                words = [str(w) for w in z["test_words"]]
                # 重建对齐的 word 顺序: _read_pred 丢了 word 列表, 这里重新对齐
                # 直接用 raw 再读一遍带 word 的 pred
                jl = os.path.join(cell, "raw", "gen_ft_heldout.jsonl")
                pred = {}
                for ln in open(jl, encoding="utf-8"):
                    o = json.loads(ln)
                    if o.get("parsed") is not None:
                        pred[str(o["word"])] = float(np.asarray(o["parsed"], float)[0])
                widx = {str(w): i for i, w in enumerate(z["test_words"])}
                xs, l2s, lfs = [], [], []
                for w, p in pred.items():
                    if w in widx and w in freq and freq[w] > 0:
                        xs.append(p); l2s.append(float(z["gt_test_real"][widx[w], 0]))
                        lfs.append(math.log(freq[w]))
                if len(xs) < 8:
                    md.append(f"| {rel} | {len(xs)} | (词频命中太少) |||| —|"); continue
                xs, l2s, lfs = np.array(xs), np.array(l2s), np.array(lfs)
                def r(a, b): return float(pearsonr(a, b)[0])
                r_pl, r_pf, r_lf = r(xs, l2s), r(xs, lfs), r(l2s, lfs)
                denom = math.sqrt(max(1e-12, (1 - r_pf**2) * (1 - r_lf**2)))
                partial = (r_pl - r_pf * r_lf) / denom
                md.append(f"| {rel} | {len(xs)} | {r_pl:.3f} | {r_pf:.3f} | {r_lf:.3f} | **{partial:.3f}** |")
                g2rows.append({"cell": rel, "n": len(xs), "r_pred_l2": r_pl,
                               "r_pred_logf": r_pf, "r_l2_logf": r_lf, "partial": partial})
            report["blocks"]["g2_partial"] = g2rows
    except Exception:
        md.append("\n## 4. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 5: G1 synonym/definition 泄漏分级 + 干净子集 HE ----------
    # 审计 c1pipe-4 / c5g-6: 0.251 含 ~46/120 fallback(原词) + ~25/120 词干变体, 激活路径没切.
    # 直接从 raw 把 held-out 词分成 fallback / 形态变体 / 真正切断 三档, 各档重算 HE. 无需 rerun.
    try:
        def _morph(a, b):
            a, b = a.strip().lower(), b.strip().lower()
            if a == b:
                return "fallback"
            if a in b or b in a:
                return "morph"
            if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
                return "morph"
            return "clean"
        md.append("\n## 5. G1 泄漏分级 + 干净子集 HE (审计 c1pipe-4)\n")
        md.append("> headline 0.251 的样本里有多少真正切断了激活路径? 按 query 分档重算 HE.\n")
        md.append("| cell | n_all HE | fallback | morph | **clean** | clean HE | clean 95%CI |")
        md.append("|---|---|---|---|---|---|---|")
        g1rows = []
        for cell in sorted(glob.glob(os.path.join(DATA, "G1_non_activation_query", "*synonym*_s*", "")) +
                           glob.glob(os.path.join(DATA, "G1_non_activation_query", "*definition*_s*", ""))):
            cell = cell.rstrip("\\/")
            rel = os.path.relpath(cell, DATA).replace("\\", "/")
            raw = os.path.join(cell, "raw")
            npz = os.path.join(raw, "pca_targets.npz")
            jl = os.path.join(raw, "gen_ft_heldout.jsonl")
            if not (os.path.exists(npz) and os.path.exists(jl)):
                continue
            z = np.load(npz, allow_pickle=True)
            widx = {str(w): i for i, w in enumerate(z["test_words"])}
            G = z["gt_test_real"]
            buckets = {"fallback": [], "morph": [], "clean": []}
            all_p, all_g = [], []
            is_def = "definition" in rel
            for ln in open(jl, encoding="utf-8"):
                o = json.loads(ln)
                w = str(o["word"]); q = str(o.get("query", w)); v = o.get("parsed")
                if v is None or w not in widx:
                    continue
                v = np.asarray(v, float)[:G.shape[1]]
                all_p.append(v); all_g.append(G[widx[w]])
                if is_def:
                    cat = "fallback" if q.strip().lower() == w.lower() else (
                        "morph" if w.lower() in q.lower() else "clean")
                else:
                    cat = _morph(w, q)
                buckets[cat].append((v, G[widx[w]]))
            def he_sub(lst):
                if len(lst) < 5:
                    return None
                P = np.array([x[0] for x in lst]); Gm = np.array([x[1] for x in lst])
                return _rsa(P, Gm)
            he_all = _rsa(np.array(all_p), np.array(all_g)) if len(all_p) >= 5 else None
            he_clean = he_sub(buckets["clean"])
            ci_clean = None
            if len(buckets["clean"]) >= 8:
                Pc = np.array([x[0] for x in buckets["clean"]])
                Gc = np.array([x[1] for x in buckets["clean"]])
                ci_clean = _bootstrap_ci(Pc, Gc)
            cstr = f"[{ci_clean[0]:.3f}, {ci_clean[1]:.3f}]" if ci_clean else "—"
            md.append(f"| {rel} | {he_all if he_all is None else round(he_all,4)} | "
                      f"{len(buckets['fallback'])} | {len(buckets['morph'])} | "
                      f"{len(buckets['clean'])} | {he_clean if he_clean is None else round(he_clean,4)} | {cstr} |")
            g1rows.append({"cell": rel, "he_all": he_all,
                           "n_fallback": len(buckets["fallback"]), "n_morph": len(buckets["morph"]),
                           "n_clean": len(buckets["clean"]), "he_clean": he_clean,
                           "he_clean_ci": (list(ci_clean[:2]) if ci_clean else None)})
        report["blocks"]["g1_leakage"] = g1rows
    except Exception:
        md.append("\n## 5. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 6: identification accuracy (读出向量 → 最近邻还原 token) ----------
    # 比 RSA 更直观的"它真编码了那个具体词吗": 模型吐的 PCA-32 向量, 在 held-out 候选集里
    # 用 cos 最近邻能否命中正确词. top-1/top-5/MRR vs chance=1/n. 论文里很讨喜的干净指标.
    try:
        def _identification(P, G):
            n = P.shape[0]
            if n < 5 or P.shape[1] < 2:
                return None  # d=1 (l2norm 等) 无识别意义
            Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
            Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
            S = Pn @ Gn.T  # (n,n): pred_i vs gt_j cos 相似度
            top1 = top5 = 0; rr = 0.0
            for i in range(n):
                order = np.argsort(-S[i])
                rank = int(np.where(order == i)[0][0]) + 1
                top1 += (rank == 1); top5 += (rank <= 5); rr += 1.0 / rank
            return {"n": n, "top1": top1 / n, "top5": top5 / n,
                    "mrr": rr / n, "chance_top1": 1.0 / n}
        KEY6 = []
        for sec in ["C1_lora_finetune", "C2_hard_target", "C3_deep_hidden",
                    "C4_cross_model_target", "G1_non_activation_query"]:
            KEY6 += glob.glob(os.path.join(DATA, sec, "*", ""))
        md.append("\n## 6. Identification accuracy (读出向量 → 最近邻还原 token)\n")
        md.append("> 模型吐的 PCA-32 向量在 held-out 候选里 cos 最近邻能否命中正确词. chance=1/n. 远超 chance = 向量编码了**具体**词而非泛泛相关.\n")
        md.append("| cell | n | top-1 | top-5 | MRR | chance(1/n) |")
        md.append("|---|---|---|---|---|---|")
        idrows = []
        for cell in sorted(KEY6):
            cell = cell.rstrip("\\/")
            rel = os.path.relpath(cell, DATA).replace("\\", "/")
            pr = _read_pred(cell, "ft_heldout")
            if pr is None:
                continue
            P, G, _n = pr
            idn = _identification(P, G)
            if idn is None:
                continue
            md.append(f"| {rel} | {idn['n']} | {idn['top1']:.3f} | {idn['top5']:.3f} | "
                      f"{idn['mrr']:.3f} | {idn['chance_top1']:.4f} |")
            idrows.append({"cell": rel, **idn})
        report["blocks"]["identification"] = idrows
    except Exception:
        md.append("\n## 6. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- BLOCK 7: C6 shared-readout CI + lh vs swap 控制 ----------
    # readiness 审查 §shared: lh(minority) 通道需 CI, 且必须证明 lh > 它自己的 swap 控制(否则非真双读出).
    # lh_heldout=output→lh; 它的控制=output→ie(swap_output_vs_ie). lh 真成立 ⟺ CI_lh 下界 > swap_out_ie.
    try:
        def _read_c6(cell):
            npz = os.path.join(cell, "raw", "pca_targets_c6.npz")
            fi = os.path.join(cell, "raw", "gen_ft_heldout_input.jsonl")
            fo = os.path.join(cell, "raw", "gen_ft_heldout_output.jsonl")
            if not all(os.path.exists(x) for x in (npz, fi, fo)):
                return None
            z = np.load(npz, allow_pickle=True)
            tw = [str(w) for w in z["test_words"]]
            gie = {w: z["gt_ie_test"][i] for i, w in enumerate(tw)}
            glh = {w: z["gt_lh_test"][i] for i, w in enumerate(tw)}
            def rd(p):
                d = {}
                for ln in open(p, encoding="utf-8"):
                    o = json.loads(ln)
                    if o.get("parsed") is not None:
                        d[str(o["word"])] = np.asarray(o["parsed"], float)
                return d
            pin, pout = rd(fi), rd(fo)
            words = [w for w in tw if w in pin and w in pout]
            if len(words) < 5:
                return None
            return (np.array([pin[w] for w in words]), np.array([pout[w] for w in words]),
                    np.array([gie[w] for w in words]), np.array([glh[w] for w in words]), len(words))
        md.append("\n## 7. C6 shared-readout: CI + lh 是否真高于其 swap 控制\n")
        md.append("> §shared kill-shot 防御: lh(minority) 通道补 bootstrap CI; lh>control ✅ = CI_lh 下界 > swap_out→ie.\n")
        md.append("| cell | n | ie HE [CI] | lh HE [CI] | lh的控制(swap_out→ie) | lh>control? |")
        md.append("|---|---|---|---|---|---|")
        c6rows = []
        for cell in sorted(glob.glob(os.path.join(DATA, "C6_mixed_target", "*", ""))):
            cell = cell.rstrip("\\/")
            rel = os.path.basename(cell)
            r6 = _read_c6(cell)
            if r6 is None:
                continue
            Pin, Pout, Gie, Glh, n = r6
            ie, lh = _rsa(Pin, Gie), _rsa(Pout, Glh)
            sw_out_ie = _rsa(Pout, Gie)
            ci_ie, ci_lh = _bootstrap_ci(Pin, Gie), _bootstrap_ci(Pout, Glh)
            lh_above = bool(ci_lh is not None and ci_lh[0] > sw_out_ie)
            def fci(x, ci): return f"{x:.3f} [{ci[0]:.3f},{ci[1]:.3f}]" if ci else f"{x:.3f}"
            md.append(f"| {rel} | {n} | {fci(ie,ci_ie)} | {fci(lh,ci_lh)} | {sw_out_ie:.3f} | "
                      f"{'✅' if lh_above else '❌'} |")
            c6rows.append({"cell": rel, "n": n, "ie": ie, "lh": lh,
                           "ci_ie": (list(ci_ie[:2]) if ci_ie else None),
                           "ci_lh": (list(ci_lh[:2]) if ci_lh else None),
                           "swap_out_ie": sw_out_ie, "lh_above_control": lh_above})
        report["blocks"]["c6"] = c6rows
    except Exception:
        md.append("\n## 7. (失败)\n```\n" + traceback.format_exc() + "\n```")

    # ---------- 落盘 ----------
    rep_md = os.path.join(HERE, "_analysis_report.md")
    rep_json = os.path.join(HERE, "_analysis.json")
    open(rep_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(report, open(rep_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[analyze] -> {rep_md}")
    print(f"[analyze] -> {rep_json}")
    print(f"[analyze] blocks: {list(report['blocks'].keys())}")


if __name__ == "__main__":
    main()
