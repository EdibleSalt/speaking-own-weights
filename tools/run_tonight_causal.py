# coding: utf-8
"""C7 过夜 runner —— 方案1+2 因果扰动, olmo + pythia, 各自隔离子进程、可断点重跑、互不拖累。
用法(务必用 venv 的 python 启动, 这样子进程继承同一解释器):
    .venv\\Scripts\\python.exe tools\\run_tonight_causal.py

设计(防"一处出错浪费整晚"):
- 每个模型单开一个子进程跑 tools/causal_perturb.py;某个模型崩了 try/except 接住, 不影响下一个。
- 已完成(对应 json 存在且 n>=100)则跳过 → 中途中断后重跑会自动续。
- 每模型 2h 超时上限, 防卡死。
- 全程日志(带时间戳)写 results/data/C7_causal_perturb/run_log.txt, 早上看这个文件即可。
注:不含方案3(微调注入)——它未经测试, 放进无人值守的过夜任务正是浪费整晚的风险, 留待有人盯着时再跑。
"""
import os, sys, json, subprocess, time, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根(tools/ 的上一级)
os.chdir(ROOT)
OUT = "results/data/C7_causal_perturb"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "run_log.txt")
PY = sys.executable                                                  # 用哪个 python 启动本脚本, 子进程就用哪个
STAGES = ["olmo", "pythia"]
PER_MODEL_TIMEOUT = 7200                                             # 2h 上限/模型(实际约 25min)

def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def done(model):
    p = f"{OUT}/{model}_perturb.json"
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p, encoding="utf-8"))
        return d.get("summary", {}).get("n", 0) >= 20   # 实际 n≈50(受 G1 同义词集大小限), 故阈值 20
    except Exception:
        return False

log(f"==== C7 因果扰动 过夜运行开始; python={PY}; stages={STAGES} ====")
results = {}
for m in STAGES:
    if done(m):
        log(f"[{m}] 已完成(json 存在, n>=100), 跳过"); results[m] = "skipped(已完成)"; continue
    log(f"[{m}] 开始 causal_perturb ...")
    t0 = time.time()
    try:
        r = subprocess.run([PY, "tools/causal_perturb.py", m],
                           cwd=ROOT, capture_output=True, text=True, timeout=PER_MODEL_TIMEOUT)
        out_tail = "\n".join((r.stdout or "").strip().splitlines()[-30:])
        if r.returncode == 0 and done(m):
            log(f"[{m}] 完成 ({time.time()-t0:.0f}s)\n--- summary ---\n{out_tail}")
            results[m] = "ok"
        else:
            err_tail = "\n".join((r.stderr or "").strip().splitlines()[-30:])
            log(f"[{m}] 失败 rc={r.returncode} ({time.time()-t0:.0f}s)\n--- stdout 尾 ---\n{out_tail}\n--- stderr 尾 ---\n{err_tail}")
            results[m] = f"失败 rc={r.returncode}"
    except subprocess.TimeoutExpired:
        log(f"[{m}] 超时(>{PER_MODEL_TIMEOUT}s), 跳过"); results[m] = "超时"
    except Exception as e:
        log(f"[{m}] 异常: {type(e).__name__}: {e}"); results[m] = f"异常 {type(e).__name__}"
log(f"==== 全部结束 ==== {results}")
print("DONE:", results)
