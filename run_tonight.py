"""隔夜加固跑 · 多 seed 复现关键单 seed 判决性 cell + 跑完自动归纳分析。

为什么: 审计 (c13strat-1 / c7stat-3) 指出所有判决性结论 (C1 main / C3 峰 / C4 / G1 / G2)
几乎都是单 seed, 而唯一多 seed 的 C2 暴露 pythia CV≈45% —— 单 seed 数字经不起抖动.
本脚本对这些 cell 补 seed=1,2, 跑完用 mean±std + bootstrap CI 取代贴地板的 p=0.0005.

**复用 run_pending 机器** (run_one / check_done_c1 / 断点续跑 / 进程隔离 / utf-8 Tee).
GPU 串行, 单 cell ~30-40min. 任一项失败不中断, 残留半成品下次自动续 (c1_lora 自带 adapter 断点).

只跑**不依赖 G1 query_dict 的 cell** (G1 synonym/definition 多 seed 需先重建覆盖更广的词典,
见 修复清单; self 不需词典故纳入). 跑完 phase=all 会自动跑 tools/analyze_all.py 出报告.

用法 (睡前):
    .venv\\Scripts\\python.exe -u run_tonight.py            # 全跑 (seeds 后 analyze)
    .venv\\Scripts\\python.exe run_tonight.py --dry-run     # 预览 (强烈建议先看一眼)
    .venv\\Scripts\\python.exe run_tonight.py --phase analyze  # 只重算分析 (秒级, 不动 GPU)
日志: _run_tonight.log    分析产出: _analysis_report.md / _analysis.json
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import run_pending as rp  # 复用 run_one / check_done_c1 / 路径 / Tee
sys.path.insert(0, os.path.join(rp.HERE, "tools"))
import cleanup_ckpts as ck  # 每跑完一个 cell 即清其 adapter/ep* 中间 ckpt, 防隔夜把盘塞满 (final 保留)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLMO = os.path.join(rp.HERE, "models", "OLMo-2-0425-1B-Instruct")
PYTHIA = os.path.join(rp.HERE, "models", "pythia-1.4b")
ANALYZE = os.path.join(rp.HERE, "tools", "analyze_all.py")
C6 = os.path.join(rp.HERE, "src", "c6_mixed_target.py")
OUTBASE_C6 = os.path.join(rp.HERE, "results", "data", "C6_mixed_target")
OLMO_BASE = os.path.join(rp.HERE, "models", "OLMo-2-0425-1B")  # base 版 (下载完就位; 未就位则 olmobase 段跳过)
G1Q = os.path.join(rp.HERE, "materials", "G1_queries")
G1_OUTBASE = os.path.join(rp.HERE, "results", "data", "G1_non_activation_query")

DEFAULTS = {
    "vocab": "random", "target_mode": "real", "target_source": "input_embed",
    "epochs": 15, "lr": "2e-4", "lora_r": 32, "lora_alpha": 64,
    "n_train": 400, "n_test": 120, "max_new": 250,
    "target_layer": None, "target_prompt": None, "target_model": None,
    "target_metric": None, "pca_dim": None, "eval_query_mode": None,
}

# (tag, section, run_short_fmt, mpath, override). {s} = seed. 优先级从上到下.
SPECS = [
    # —— 解构三支柱里依赖单 seed 的 cell ——
    ("G2", "G2_physical_target", "olmo2-1b_l2norm_s{s}",   OLMO,   {"target_metric": "l2norm", "pca_dim": 1}),
    ("G2", "G2_physical_target", "pythia-1.4b_l2norm_s{s}", PYTHIA, {"target_metric": "l2norm", "pca_dim": 1}),
    ("G1self", "G1_non_activation_query", "olmo2-1b_self_s{s}",   OLMO,   {"eval_query_mode": "self"}),
    ("G1self", "G1_non_activation_query", "pythia-1.4b_self_s{s}", PYTHIA, {"eval_query_mode": "self"}),
    # —— C4 verdict 翻转 / pythia ie 反例 (最该补 seed) ——
    ("C4", "C4_cross_model_target", "pythia-1.4b_FT_to_olmo2-1b_ie_s{s}", PYTHIA, {"target_model": OLMO, "target_source": "input_embed"}),
    ("C4", "C4_cross_model_target", "olmo2-1b_FT_to_pythia-1.4b_ie_s{s}", OLMO,   {"target_model": PYTHIA, "target_source": "input_embed"}),
    ("C4", "C4_cross_model_target", "pythia-1.4b_FT_to_olmo2-1b_lh_s{s}", PYTHIA, {"target_model": OLMO, "target_source": "lm_head"}),
    ("C4", "C4_cross_model_target", "olmo2-1b_FT_to_pythia-1.4b_lh_s{s}", OLMO,   {"target_model": PYTHIA, "target_source": "lm_head"}),
    # —— C1 锚点 + C3 峰 ——
    ("C1", "C1_lora_finetune", "olmo2-1b_random_s{s}",   OLMO,   {}),
    ("C1", "C1_lora_finetune", "pythia-1.4b_random_s{s}", PYTHIA, {}),
    ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L12_P4_s{s}", OLMO, {"target_source": "hidden", "target_layer": 12, "target_prompt": "P4"}),
]
SEEDS = [1, 2]


def build_cmd(mpath, outdir, spec, seed):
    s = {**DEFAULTS, **spec}
    cmd = [rp.PY, rp.C1, "--model", mpath, "--outdir", outdir,
           "--train_vocab", s["vocab"], "--target_mode", s["target_mode"],
           "--target_source", s["target_source"], "--seed", str(seed),
           "--epochs", str(s["epochs"]), "--lr", str(s["lr"]),
           "--lora_r", str(s["lora_r"]), "--lora_alpha", str(s["lora_alpha"]),
           "--n_train", str(s["n_train"]), "--n_test", str(s["n_test"]),
           "--max_new", str(s["max_new"])]
    if s["target_source"] == "hidden":
        cmd += ["--target_layer", str(s["target_layer"]), "--target_prompt", s["target_prompt"]]
    if s["target_model"]:
        cmd += ["--target_model", s["target_model"]]
    if s["target_metric"]:
        cmd += ["--target_metric", s["target_metric"], "--pca_dim", str(s["pca_dim"])]
    if s["eval_query_mode"]:
        cmd += ["--eval_query_mode", s["eval_query_mode"]]
    return cmd


def plan_seeds():
    """seeds-outer: 先把所有 cell 的 s1 跑齐 (拿到 n=2), 再跑 s2 (n=3)."""
    tasks = []
    for seed in SEEDS:
        for tag, section, fmt, mpath, override in SPECS:
            if not os.path.isdir(mpath):
                print(f"[plan] {fmt}: 模型缺失 ({mpath}), 跳过", flush=True)
                continue
            run_short = fmt.format(s=seed)
            outdir = os.path.join(rp.HERE, "results", "data", section, run_short)
            cmd = build_cmd(mpath, outdir, override, seed)
            tasks.append((f"SEED · {tag}/{run_short}", outdir, cmd, rp.check_done_c1, False))
    return tasks


def check_done_c6(outdir):
    """C6 完成 = summary.json 含 ie_heldout 与 lh_heldout."""
    d = rp._read_summary(outdir)
    return d is not None and ("ie_heldout" in d) and ("lh_heldout" in d)


# C6 混合目标: (ratio=ie占比, prompt_style, seeds). CORE=headline 70/30 多seed; EXTRA=ratio扫+prompt稳健.
C6_CORE = [(0.7, "tag", [0, 1, 2])]
C6_EXTRA = [(0.5, "tag", [0]), (0.9, "tag", [0]), (0.7, "verbal", [0]), (0.7, "symbol", [0])]
C6_MODELS = [("olmo2-1b", OLMO), ("pythia-1.4b", PYTHIA)]


def plan_c6(specs=C6_CORE):
    """C6 混合目标 cell. 命名 <model>_r<RR>_<style>_s<seed>. tag 两模型, verbal/symbol 仅 OLMo.
    --skip_base: 跳过 base 零样本 eval (parse≈0 已知), 省一整趟 250-token×120词 生成, c6 提速 ~1/4."""
    tasks = []
    for ratio, style, seeds in specs:
        rr = f"r{int(round(ratio * 100))}"
        models = C6_MODELS if style == "tag" else [("olmo2-1b", OLMO)]
        for m_short, mpath in models:
            if not os.path.isdir(mpath):
                print(f"[plan][C6] {m_short}: 模型缺失, 跳过", flush=True)
                continue
            for seed in seeds:
                run_short = f"{m_short}_{rr}_{style}_s{seed}"
                outdir = os.path.join(OUTBASE_C6, run_short)
                cmd = [rp.PY, C6, "--model", mpath, "--outdir", outdir,
                       "--ratio", str(ratio), "--prompt_style", style, "--seed", str(seed),
                       "--skip_base", "--n_train", "400", "--n_test", "120", "--epochs", "15",
                       "--lr", "2e-4", "--lora_r", "32", "--lora_alpha", "64", "--max_new", "250"]
                tasks.append((f"C6 · {run_short}", outdir, cmd, check_done_c6, False))
    return tasks


def plan_g1clean(modes=("synonym",)):
    """G1-clean: 干净词典 (排形态变体 + comprehensive 多 seed 覆盖) 跑 × 2 模型 × 3 seed.
    默认只 synonym (核心激活切断判据); definition 进 --phase extra. self baseline 复用已跑 G1 self.
    泄漏分级 / clean-subset HE 由 analyze_all BLOCK 5 自动算.
    """
    tasks = []
    for mode in modes:
        for seed in [0, 1, 2]:
            for m_short, mpath, dict_tag in [("olmo2-1b", OLMO, "olmo"),
                                             ("pythia-1.4b", PYTHIA, "pythia")]:
                if not os.path.isdir(mpath):
                    continue
                dpath = os.path.join(G1Q, f"{dict_tag}_{mode}_clean.json")
                if not os.path.exists(dpath):
                    print(f"[plan][g1clean] 词典缺失 {dpath}, 跳过", flush=True)
                    continue
                run_short = f"{m_short}_{mode}_clean_s{seed}"
                outdir = os.path.join(G1_OUTBASE, run_short)
                cmd = [rp.PY, rp.C1, "--model", mpath, "--outdir", outdir,
                       "--eval_query_mode", mode, "--query_dict_path", dpath,
                       "--train_vocab", "random", "--target_mode", "real",
                       "--target_source", "input_embed", "--seed", str(seed),
                       "--epochs", "15", "--lr", "2e-4", "--lora_r", "32",
                       "--lora_alpha", "64", "--n_train", "400", "--n_test", "120",
                       "--max_new", "250"]
                tasks.append((f"G1c · {run_short}", outdir, cmd, rp.check_done_c1, False))
    return tasks


# OLMo-base C1/C2 (base/instruct 可分离对照). (tag, section, fmt, override, seeds)
OLMOBASE_SPECS = [
    ("C1b",      "C1_lora_finetune", "olmo2-1b-base_random_s{s}",             {}, [0, 1, 2]),
    ("C2b",      "C2_hard_target",   "olmo2-1b-base_random_real_lmhead_s{s}", {"target_source": "lm_head"}, [0, 1, 2]),
    ("C1b-ctrl", "C1_lora_finetune", "olmo2-1b-base_random_random_s{s}",      {"target_mode": "random"}, [0]),
]


def plan_olmobase(seeds=(0,)):
    """OLMo-2-1B-base C1(ie)+C2(lh)+control × seeds (默认 s0 先看 base/instruct 对照; s1/s2 进 --phase extra).
    与现有 OLMo-instruct 对比 → 单独 hold base/instruct 轴. 模型未就位则整段跳过 (下载完重跑纳入)."""
    if not os.path.isdir(OLMO_BASE):
        print(f"[plan][olmobase] {OLMO_BASE} 未就位 (下载中?), 整段跳过", flush=True)
        return []
    tasks = []
    for tag, section, fmt, override, allowed in OLMOBASE_SPECS:
        for s in seeds:
            if s not in allowed:
                continue
            run_short = fmt.format(s=s)
            outdir = os.path.join(rp.HERE, "results", "data", section, run_short)
            cmd = build_cmd(OLMO_BASE, outdir, override, s)
            tasks.append((f"OLMOBASE · {tag}/{run_short}", outdir, cmd, rp.check_done_c1, False))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["seeds", "c6", "g1clean", "olmobase", "extra", "analyze", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logfile", default="_run_tonight.log")
    args = ap.parse_args()

    log_handle = None
    if args.logfile and not args.dry_run:
        log_handle = open(os.path.join(rp.HERE, args.logfile), "a", encoding="utf-8", buffering=1)
        log_handle.write(f"\n\n{'#'*72}\n# run_tonight @ {datetime.now().isoformat(timespec='seconds')}\n{'#'*72}\n")
        sys.stdout = rp._Tee(sys.__stdout__, log_handle)

    print(f"\n[start] {datetime.now().isoformat(timespec='seconds')} phase={args.phase}")
    tasks = []
    if args.phase in ("seeds", "all"):
        tasks += plan_seeds()
    if args.phase in ("c6", "all"):
        tasks += plan_c6()
    if args.phase in ("g1clean", "all"):
        tasks += plan_g1clean()
    if args.phase in ("olmobase", "all"):
        tasks += plan_olmobase()
    if args.phase in ("extra", "all"):   # 二级变体放最后: c6 ratio扫/prompt稳健 + g1clean definition + olmobase s1/s2
        tasks += plan_c6(C6_EXTRA) + plan_g1clean(("definition",)) + plan_olmobase((1, 2))
    n = len(tasks)
    print(f"[plan] 共 {n} 项 (已完成自动 skip)\n")

    if args.dry_run:
        for i, (tag, outdir, cmd, check, _) in enumerate(tasks, 1):
            state = "skip(完成)" if check(outdir) else ("续跑(半成品)" if os.path.isdir(outdir) else "新跑")
            print(f"  [{i:2d}/{n}] {state:12s} {tag}")
        print("\n[dry-run] analyze 阶段会跑 tools/analyze_all.py (此处不执行)")
        return

    counts = {"ok": 0, "skipped": 0, "fail": 0}
    for i, (tag, outdir, cmd, check, cleanup_p) in enumerate(tasks, 1):
        status, dt = rp.run_one(i, n, tag, outdir, cmd, check, cleanup_partial=cleanup_p)
        counts[status] += 1
        # cell 完成即清它的 epN 中间 ckpt (prune_cell 只动 final+summary 都在的已完成 cell, 续训不受影响)
        if status in ("ok", "skipped"):
            freed = ck.prune_cell(outdir, apply=True)
            if freed > 1e8:
                print(f"    [prune] 回收 {freed/1e9:.2f} GB epN ({os.path.basename(outdir)})", flush=True)

    if tasks:
        print(f"\n[seeds done] counts={counts}")

    if args.phase in ("analyze", "all"):
        print(f"\n{'='*72}\n[analyze] 跑 tools/analyze_all.py (重算权威数字 + CI + 偏相关 + 泄漏分级)\n{'='*72}", flush=True)
        r = subprocess.run([rp.PY, ANALYZE], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(r.stdout or "")
        if r.returncode != 0:
            print("[analyze] FAIL:\n" + (r.stderr or "")[-2000:])
        else:
            print("[analyze] OK -> _analysis_report.md")

    print(f"\n[all done] {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
