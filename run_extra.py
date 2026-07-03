"""run_extra.py · 补跑两组中间层 (C3 mid-hidden, L12) 实验。

两个目的:
  1. **OLMo-base 中间层** —— 补 fig2 缺的那一格 (此前 NaN / 图中标 "n.a.")。
  2. **pythia 中间层 s1/s2** —— s0 已有但只有 1 个 seed, 没法算它的 seed/词集
     敏感度。补到 3 seed, 才能验证"pythia 中间层是不是真的对词集敏感"
     (之前是从'均值≈0 必然高方差'推断的, 未实测)。

所有 cell 配置与 OLMo-Instruct 的 C3 L12 完全一致 (target_source=hidden,
target_layer=12, target_prompt=P4, 其余默认超参), 跑 seed 0/1/2 —— 与 fig2 其它
3-seed 格口径一致。已完成的 seed (如 pythia s0) 会自动 skip。

实现上**直接复用 run_pending.run_one** (每项独立 subprocess + 断点重跑 +
GPU 串行隔离, c1_lora 自带 adapter/epN 断点), 不重写调度逻辑。

用法 (在仓库根目录):
    .venv\\Scripts\\python.exe -u run_extra.py                # 跑全部 (OLMo-base + pythia)
    .venv\\Scripts\\python.exe -u run_extra.py --only pythia   # 只跑 pythia 那组 (s0 跳过)
    .venv\\Scripts\\python.exe -u run_extra.py --only base     # 只跑 OLMo-base 那组
    .venv\\Scripts\\python.exe -u run_extra.py --dry-run       # 只看计划不跑
跑完按模型分别打印 held-out RSA 的 mean/std 方便回填 make_paper_figs.m / 分析。
单 cell ~30min (pythia s0 实测 29.7min); 注意 bit-exact 续训每 epoch 存 ~300MB
fp32 opt_state, 跑完可用 tools/cleanup_ckpts.py 回收。
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

# 复用 run_pending 的执行器与常量 (导入只定义函数/常量, 不会触发跑实验)
from run_pending import run_one, check_done_c1, OUTBASE_C3, PY, C1, HERE, _Tee

# (标签, model_short, model_path, seeds, 备注). model_short 同已有目录命名约定。
EXTRA = [
    ("OLMo-base mid-hidden", "olmo2-1b-base",
     os.path.join(HERE, "models", "OLMo-2-0425-1B"), [0, 1, 2],
     "补 fig2 缺的 OLMo-base 中间层格"),
    ("pythia mid-hidden", "pythia-1.4b",
     os.path.join(HERE, "models", "pythia-1.4b"), [0, 1, 2],
     "补 pythia 中间层 s1/s2 (s0 已有), 实测其 seed/词集敏感度"),
]


def c3_task(model_short, model_path, seed):
    """构造一项 C3 mid-hidden L12 P4 task; 配置对齐 OLMo-Instruct C3 L12。"""
    run_short = f"{model_short}_random_real_hidden_L12_P4_s{seed}"
    outdir = os.path.join(OUTBASE_C3, run_short)
    cmd = [PY, C1, "--model", model_path, "--outdir", outdir,
           "--train_vocab", "random",
           "--target_mode", "real",
           "--target_source", "hidden",
           "--target_layer", "12",
           "--target_prompt", "P4",
           "--seed", str(seed),
           "--epochs", "15",
           "--lr", "2e-4",
           "--lora_r", "32",
           "--lora_alpha", "64",
           "--n_train", "400",
           "--n_test", "120",
           "--max_new", "250"]
    # cleanup_partial=False: c1_lora 自带 adapter/epN 断点, 别 rmtree 半成品
    return (f"EXTRA · C3/{run_short}", outdir, cmd, check_done_c1, False)


def selected_entries(only):
    out = []
    for label, m_short, m_path, seeds, note in EXTRA:
        if only and (only.lower() not in label.lower()
                     and only.lower() not in m_short.lower()):
            continue
        if not os.path.isdir(m_path):
            print(f"[warn] {label}: 模型缺失 ({m_path}), 整组跳过", flush=True)
            continue
        out.append((label, m_short, m_path, seeds, note))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="只跑名字含该子串的那组 (如 pythia / base); 默认全跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不跑")
    ap.add_argument("--logfile", default="_run_extra.log",
                    help="utf-8 镜像 stdout 到此文件 (避开 PowerShell `>` 重定向坑); 空串关闭")
    args = ap.parse_args()

    entries = selected_entries(args.only)
    if not entries:
        sys.exit("[fatal] 没有可跑的组 (模型缺失或 --only 没匹配到)")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log_handle = None
    if args.logfile and not args.dry_run:
        log_handle = open(os.path.join(HERE, args.logfile), "a", encoding="utf-8", buffering=1)
        log_handle.write(f"\n\n{'#'*72}\n# run_extra @ "
                         f"{datetime.now().isoformat(timespec='seconds')}\n{'#'*72}\n")
        sys.stdout = _Tee(sys.__stdout__, log_handle)

    tasks = []
    for label, m_short, m_path, seeds, note in entries:
        for s in seeds:
            tasks.append(c3_task(m_short, m_path, s))
    n = len(tasks)
    print(f"\n[run_extra] C3 mid-hidden L12 · 组: {', '.join(e[0] for e in entries)} · {n} 项")
    for label, m_short, m_path, seeds, note in entries:
        print(f"  - {label}: seeds={seeds}  ({note})")
    print()

    if args.dry_run:
        for i, (tag, outdir, cmd, check, _cp) in enumerate(tasks, 1):
            if check(outdir):
                state = "skip(完成)"
            elif os.path.isdir(outdir):
                state = "续跑(半成品)"
            else:
                state = "新跑"
            print(f"  [{i}/{n}] {state:12s} {tag}")
        return

    counts = {"ok": 0, "skipped": 0, "fail": 0}
    for i, (tag, outdir, cmd, check, cleanup_p) in enumerate(tasks, 1):
        status, _dt = run_one(i, n, tag, outdir, cmd, check, cleanup_partial=cleanup_p)
        counts[status] += 1
    print(f"\n[run_extra done] counts={counts}")

    # 按模型分别摘 held-out RSA, 算 mean/std (含 seed 间波动 = 词集/seed 敏感度)
    print("\n[结果汇总] (held-out RSA; std/CV 即 seed/词集 敏感度)")
    for label, m_short, m_path, seeds, note in entries:
        per = []
        for s in seeds:
            sj = os.path.join(OUTBASE_C3,
                              f"{m_short}_random_real_hidden_L12_P4_s{s}", "summary.json")
            if os.path.exists(sj):
                d = json.load(open(sj, encoding="utf-8"))
                he = (d.get("ft_heldout") or {}).get("rsa")
                if isinstance(he, (int, float)):
                    per.append((s, he))
        if not per:
            print(f"  {label}: (暂无结果)")
            continue
        vals = [v for _, v in per]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = (std / mean * 100) if (len(vals) > 1 and mean) else float("nan")
        detail = ", ".join(f"s{s}={v:.3f}" for s, v in per)
        print(f"  {label:22s} mean={mean:.3f}  std={std:.3f}  CV={cv:.1f}%  (n={len(vals)})  [{detail}]")


if __name__ == "__main__":
    main()
