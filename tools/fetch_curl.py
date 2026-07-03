"""并行 curl 下载器：高延迟链路下开多路连接成倍提速，保留字节级续传+失速重连+无限重试。
用法: python fetch_curl.py <repo_id> [<repo_id> ...]
token 从环境变量 HF_TOKEN 读(gated 需要)；端点从 HF_ENDPOINT 读(默认官方)。下到 models/<repo尾名>/。
"""
import concurrent.futures as cf
import os
import subprocess
import sys

from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN", "")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
WORKERS = int(os.environ.get("DL_WORKERS", "6"))
MIN_SPEED = int(os.environ.get("DL_MIN_SPEED", "20000"))  # bytes/s, 低于此持续 45s 则中断重试; 0=不限速
SHOW_PROGRESS = os.environ.get("DL_PROGRESS", "") not in ("", "0")  # 显示 curl 实时进度
SKIP_EXT = (".gguf", ".pth", ".onnx", ".h5", ".msgpack", ".tflite", ".png", ".jpg")
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")  # 项目根/models (脚本在 tools/ 下要上跳一级; 之前误下到 tools/models/)


def curl_file(repo, fname, dst):
    url = f"{ENDPOINT}/{repo}/resolve/main/{fname}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    quiet = [] if SHOW_PROGRESS else ["-s", "-S"]  # 去 -s 让 curl 显示进度/速度/ETA
    cmd = ["curl", "-L", "-C", "-", "--retry", "999", "--retry-delay", "5",
           "--retry-all-errors", "--connect-timeout", "30",
           "--speed-limit", str(MIN_SPEED), "--speed-time", "45", *quiet,
           "-o", dst, url]
    if TOK:
        cmd[1:1] = ["-H", f"Authorization: Bearer {TOK}"]
    return subprocess.run(cmd).returncode


def collect_tasks(repos):
    tasks = []
    for repo in repos:
        info = HfApi(endpoint=ENDPOINT).model_info(repo, token=TOK or None, files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
        outdir = os.path.join(BASE, repo.split("/")[-1])
        has_st = any(f.endswith(".safetensors") for f in sizes)
        for f, sz in sizes.items():
            if f.startswith(".") or f.endswith(SKIP_EXT):
                continue
            if f.endswith(".bin") and has_st:
                continue
            if "original/" in f or "onnx/" in f:
                continue
            dst = os.path.join(outdir, f)
            if os.path.exists(dst) and sz and os.path.getsize(dst) == sz:
                continue
            tasks.append((repo, f, dst, sz))
    return tasks


def worker(t):
    repo, f, dst, sz = t
    rc = curl_file(repo, f, dst)
    got = os.path.getsize(dst) if os.path.exists(dst) else 0
    ok = sz == 0 or got == sz
    return f"[{'ok' if ok else 'PARTIAL'}] {repo.split('/')[-1]}/{f}  {got/1e6:.0f}MB rc={rc}"


def main():
    # CLI 覆盖 (argv 比 env 可靠: WSL bash 不会把 HF_ENDPOINT 等 env 传给 Windows .exe)
    global ENDPOINT, WORKERS, MIN_SPEED, SHOW_PROGRESS
    args, repos = sys.argv[1:], []
    i = 0
    while i < len(args):
        if args[i] == "--endpoint":
            ENDPOINT = args[i + 1].rstrip("/"); i += 2
        elif args[i] == "--workers":
            WORKERS = int(args[i + 1]); i += 2
        elif args[i] == "--min-speed":
            MIN_SPEED = int(args[i + 1]); i += 2  # 低于此 bytes/s 持续 45s 才中断重试; 0=完全不限
        elif args[i] == "--progress":
            SHOW_PROGRESS = True; i += 1  # 显示 curl 实时进度 (建议配 --workers 1, 否则多条进度交错)
        else:
            repos.append(args[i]); i += 1
    print(f"[cfg] endpoint={ENDPOINT}  workers={WORKERS}  repos={repos}", flush=True)
    tasks = collect_tasks(repos)
    print(f"[plan] {len(tasks)} files across {len(repos)} repos, {WORKERS} parallel\n", flush=True)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for line in ex.map(worker, tasks):
            print(line, flush=True)
    # 汇总各 repo 体积
    print("\n[inventory]", flush=True)
    for repo in repos:
        d = os.path.join(BASE, repo.split("/")[-1])
        tot = sum(os.path.getsize(os.path.join(dp, fn))
                  for dp, _, fns in os.walk(d) for fn in fns) if os.path.exists(d) else 0
        print(f"  {repo} -> {tot/1e9:.2f} GB", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
