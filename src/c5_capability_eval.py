"""C5. 通用能力保留检验 (H0 capability preservation)

包装 lm-evaluation-harness 跑 5 task × base/LoRA, 对比 |Δ acc| 判通用能力是否退化.

5 task (覆盖 LM 基础 / common sense / 推理 / 共指):
  lambada_openai · hellaswag · piqa · arc_easy · winogrande

接口:
  python -m c5_capability_eval --model <base_path> [--adapter <adapter/final>] --outdir <dir>

PEFT adapter 通过 lm-eval CLI `peft=` 参数直接 load (不 merge), 保 4 道闸 base 权重不动.
"""
import argparse
import json
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"]


def run_lm_eval(model_path, adapter_path, outdir, tasks, batch_size, device):
    """调 lm_eval CLI 跑 5 task; 解析 results 落 summary.json + raw/."""
    os.makedirs(outdir, exist_ok=True)
    rawdir = os.path.join(outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)

    # 构造 model_args: base 或 base + peft
    model_args = f"pretrained={model_path}"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    model_args += f",dtype=bfloat16"

    # lm_eval CLI 调用 (output 进 outdir/raw, summary 自己整理)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", rawdir,
    ]
    # HF Hub endpoint:
    # - 模型本地 (models/) 已离线缓存, 不走 hub.
    # - lm-eval datasets (hellaswag/lambada/...) 必须 hub 拉. hf-mirror.com 把 /datasets/* 和
    #   /api/datasets/* 都 308 转回 huggingface.co (2026-06-14 实测) → 镜像不解决 datasets 拉取.
    #   故默认走主站; 用户本机若不能直连 HF, 自行 export HF_ENDPOINT 走可用 datasets 镜像.
    env = os.environ.copy()
    # 主动覆盖掉项目级 hf-mirror 默认（如有），避免 datasets 跑空
    if env.get("HF_ENDPOINT", "").rstrip("/") == "https://hf-mirror.com":
        env.pop("HF_ENDPOINT")
    print(f"[c5] cmd = {' '.join(cmd)}", flush=True)
    print(f"[c5] HF_ENDPOINT = {env.get('HF_ENDPOINT', '<default huggingface.co>')}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    # 保留 stdout/stderr 入 result.txt (跟其他实验一致)
    with open(os.path.join(outdir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        if proc.returncode != 0:
            f.write("\n\n=== STDERR (tail) ===\n" + (proc.stderr or "")[-4000:])
    if proc.returncode != 0:
        print(f"[c5] FAIL rc={proc.returncode}", flush=True)
        print((proc.stderr or "")[-2000:])
        return False

    # 解析 lm_eval 落盘的 results 文件 (raw/<model_name>/results_*.json)
    scores = {}
    for sub in os.listdir(rawdir):
        subp = os.path.join(rawdir, sub)
        if not os.path.isdir(subp):
            continue
        for fname in os.listdir(subp):
            if fname.startswith("results_") and fname.endswith(".json"):
                with open(os.path.join(subp, fname), encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("results", {})
                for task, m in results.items():
                    # lm-eval 用 acc 或 acc_norm 作主指标; 优先 acc
                    if "acc,none" in m:
                        scores[task] = {
                            "acc": m["acc,none"],
                            "acc_stderr": m.get("acc_stderr,none"),
                        }
                        if "acc_norm,none" in m:
                            scores[task]["acc_norm"] = m["acc_norm,none"]
                            scores[task]["acc_norm_stderr"] = m.get("acc_norm_stderr,none")
                    elif "perplexity,none" in m:
                        # lambada_openai 用 perplexity (lower 更好) + acc
                        scores[task] = {
                            "acc": m.get("acc,none"),
                            "acc_stderr": m.get("acc_stderr,none"),
                            "perplexity": m["perplexity,none"],
                        }
                break  # 只读一个 results_*.json (lm-eval 一次跑一个文件)

    summary = {
        "model": model_path,
        "adapter": adapter_path or None,
        "tasks": tasks,
        "scores": scores,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印一个简表
    print(f"\n[c5] {model_path}{' + ' + adapter_path if adapter_path else ''}")
    for task in tasks:
        if task in scores:
            s = scores[task]
            line = f"  {task:20s}  acc={s.get('acc', float('nan')):.4f}"
            if "acc_norm" in s:
                line += f"  acc_norm={s['acc_norm']:.4f}"
            if "perplexity" in s:
                line += f"  ppl={s['perplexity']:.2f}"
            print(line)
        else:
            print(f"  {task:20s}  (missing)")
    return True


def check_done(outdir):
    """C5 完成 = summary.json 含 scores 且 5 task 都有."""
    sj = os.path.join(outdir, "summary.json")
    if not os.path.exists(sj):
        return False
    try:
        data = json.load(open(sj, encoding="utf-8"))
    except Exception:
        return False
    scores = data.get("scores") or {}
    return len(scores) >= len(TASKS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="base 模型路径 (HF id 或本地 dir)")
    ap.add_argument("--adapter", default=None,
                    help="可选 PEFT adapter 路径 (adapter/final). 不传 = base only.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help=f"逗号分隔 task 列表, 默认 {','.join(TASKS)}")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    ok = run_lm_eval(args.model, args.adapter, args.outdir, tasks,
                     args.batch_size, args.device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
