"""隔夜实验调度器 · 一键跑剩余实验。

用法（每天起床执行一次即可）:
    # 方式 1（推荐, cmd 窗口避开 PowerShell 编码坑）: 双击 run_pending.bat
    # 方式 2（PowerShell 直接跑, 脚本会自己 Tee 写 _run_pending.log）:
    cd <repo_root>
    .venv\\Scripts\\python.exe -u run_pending.py

不要用 `> _run_pending.log 2>&1` 重定向——PowerShell 5.x 会把 native exe 输出
重编码成 UTF-16 + 包成 ErrorRecord, 出错时日志变 16 位乱码 (附录 H 的坑).
脚本内部已用 utf-8 同时写 _run_pending.log + stdout, 不需要 shell 重定向.

特性:
  - **断点重跑**: 每项独立 outdir; summary.json 完整 = 已完成 -> skip; 不完整或
    缺失 -> 上次中断的"半成品", 自动 rmtree 重跑. 中途随便 Ctrl+C, 下次跑接着干.
  - **进程隔离**: 每项独立 subprocess, CUDA 显存干净, 避免跨任务 OOM 传染.
  - **串行 GPU**: 上一项完成才启动下一项, 无显存冲突.
  - **顺序**: Phase A (C0-base 延伸 ①, 探针, 轻) -> Phase B (B4 A-sweep, poc 生成, 重).
    可用 --phase {A,B,C,all} 限制只跑某一段.
  - **失败不中断**: 单项失败记日志, 继续下一项; 残留 outdir 留作判别但下次会被
    cleanup_if_partial 清掉重跑.

不包含的实验:
  - C1 (OLMo LoRA): c1_lora.py 待加 adapter 保存 + 分离 eval + 限 eval 生成 三处
    改动后才能进入此调度. 见纲要 §C1.
  - C2 / D / E / F: 都需要先写新代码或下新模型, 见纲要附录 D.

输出:
  - 顶层日志: _run_pending.log
  - 全局汇总: _run_pending_summary.json (每完成一项就重写, 中断也有)
  - 每跑独立 outdir/result.txt + summary.json + raw/ (脚本自带)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class _Tee:
    """同时写多个流; 用来把 stdout 镜像到 utf-8 日志文件, 避开 PowerShell `>` 重定向坑."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PROBE = os.path.join(HERE, "src", "probe_ceiling.py")
POC = os.path.join(HERE, "src", "poc.py")
C1 = os.path.join(HERE, "src", "c1_lora.py")
C5 = os.path.join(HERE, "src", "c5_capability_eval.py")
G3 = os.path.join(HERE, "src", "g3_cross_rdm.py")

# results输出路径
OUTBASE_C0 = os.path.join(HERE, "results", "data", "C0_probe_ceiling")
OUTBASE_B4 = os.path.join(HERE, "results", "data", "B4_a_sweep")
OUTBASE_C1 = os.path.join(HERE, "results", "data", "C1_lora_finetune")
OUTBASE_C2 = os.path.join(HERE, "results", "data", "C2_hard_target")
OUTBASE_C3 = os.path.join(HERE, "results", "data", "C3_deep_hidden")
OUTBASE_C4 = os.path.join(HERE, "results", "data", "C4_cross_model_target")
OUTBASE_C5 = os.path.join(HERE, "results", "data", "C5_capability_eval")
OUTBASE_G1 = os.path.join(HERE, "results", "data", "G1_non_activation_query")
OUTBASE_G2 = os.path.join(HERE, "results", "data", "G2_physical_target")
OUTBASE_G3 = os.path.join(HERE, "results", "data", "G3_cross_rdm")
SUMMARY = os.path.join(HERE, "_run_pending_summary.json")

# 模型注册 (short, source, probe_bs)
# 0.5B 在 HF cache, 用 HF id; 其余在 models/<name>
INSTRUCT_MODELS = [
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct",                              64),
    ("qwen3-1.7b",   os.path.join(HERE, "models", "Qwen3-1.7B"),                32),
    ("gemma3-1b",    os.path.join(HERE, "models", "gemma-3-1b-it"),             32),
    ("olmo2-1b",     os.path.join(HERE, "models", "OLMo-2-0425-1B-Instruct"),   32),
    ("qwen2.5-3b",   os.path.join(HERE, "models", "Qwen2.5-3B-Instruct"),       16),
    ("llama3.2-3b",  os.path.join(HERE, "models", "Llama-3.2-3B-Instruct"),     16),
    ("smollm3-3b",   os.path.join(HERE, "models", "SmolLM3-3B"),                16),
]
PROMPTS = ["P1", "P2", "P3", "P4", "P5", "P6"]


# ----------------------------- 完成判定 -----------------------------
def _read_summary(outdir):
    sj = os.path.join(outdir, "summary.json")
    if not os.path.exists(sj):
        return None
    try:
        return json.load(open(sj, encoding="utf-8"))
    except Exception:
        return None


def check_done_probe(outdir):
    """C0 探针完成 = summary.json 含 best_layer 且 per_layer 非空."""
    d = _read_summary(outdir)
    if d is None:
        return False
    bl = d.get("best_layer") or {}
    return ("test_rsa" in bl) and bool(d.get("per_layer"))


def check_done_poc(outdir):
    """A 族 poc 完成 = summary.json 中 per_seed 长度 == seeds."""
    d = _read_summary(outdir)
    if d is None:
        return False
    per_seed = d.get("per_seed") or []
    return len(per_seed) == d.get("seeds", -1) and len(per_seed) > 0


def check_done_c1(outdir):
    """C1/G1/G2 LoRA 完成 = summary.json 含 ft_heldout 与 ft_trained 字段
    (说明训练 + FT eval 都跑完了; 部分 fail/parse-0 也算"任务结束")."""
    d = _read_summary(outdir)
    if d is None:
        return False
    return ("ft_heldout" in d) and ("ft_trained" in d)


def check_done_c5(outdir):
    """C5 capability eval 完成 = summary.json 含 scores 且 5 task 都在."""
    d = _read_summary(outdir)
    if d is None:
        return False
    scores = d.get("scores") or {}
    return len(scores) >= 5


def check_done_g3(outdir):
    """G3 静态 RDM RSA 完成 = summary.json 含 rdm_rsa 且 6 对都在."""
    d = _read_summary(outdir)
    if d is None:
        return False
    rsa = d.get("rdm_rsa") or {}
    return len(rsa) >= 6


def cleanup_if_partial(outdir, check_fn):
    """outdir 存在但未完成 -> 半成品, 清掉重跑."""
    if os.path.isdir(outdir) and not check_fn(outdir):
        print(f"  [clean] 半成品 -> rmtree {outdir}")
        shutil.rmtree(outdir, ignore_errors=True)


# ----------------------------- 子任务执行 -----------------------------
def run_one(idx, total, tag, outdir, cmd, check_fn, cleanup_partial=True):
    """跑一个子任务. 返回 ('skipped' | 'ok' | 'fail', 耗时秒).

    cleanup_partial=False 时不清半成品 outdir——给 C1 这种有内部断点
    机制的任务用 (adapter/epN ckpt 不能被 rmtree, 否则进度全丢).
    """
    head = f"[{idx}/{total}] {tag}"
    if check_fn(outdir):
        print(f"{head}  [skip] 已完成")
        return ("skipped", 0.0)
    # 决定半成品策略, 但把日志推到 task header 之后再打 (避免与上个任务 [ok] 视觉粘连)
    will_keep_partial = False
    if cleanup_partial:
        cleanup_if_partial(outdir, check_fn)
    elif os.path.isdir(outdir) and not check_fn(outdir):
        will_keep_partial = True
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*72}\n{head}\n  outdir = {outdir}\n  cmd    = {' '.join(cmd[1:])}\n{'='*72}",
          flush=True)
    if will_keep_partial:
        print(f"{head}  [keep] 半成品但保留 (任务自带断点机制)", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.time() - t0
    # stdout 直写 result.txt (子进程已 reconfigure utf-8); stderr 仅在失败时追加
    with open(os.path.join(outdir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        if proc.returncode != 0:
            f.write("\n\n=== STDERR (tail) ===\n" + (proc.stderr or "")[-4000:])
    ok = proc.returncode == 0 and check_fn(outdir)
    if ok:
        print(f"{head}  [ok] {dt:.1f}s")
    else:
        print(f"{head}  [FAIL] rc={proc.returncode}  {dt:.1f}s")
        print(f"  stderr tail:\n{(proc.stderr or '')[-1500:]}")
    return (("ok" if ok else "fail"), dt)


# ----------------------------- 任务规划 -----------------------------
def plan_phase_a():
    """Phase A: C0-base 延伸 ①  instruct × P1..P6 = 7×6 = 42 项 (已跑过的 skip)."""
    tasks = []
    for short, mpath, bs in INSTRUCT_MODELS:
        if os.path.sep in mpath and not os.path.isdir(mpath):
            print(f"[plan][A] {short}: 本地模型缺失 ({mpath}), 6 prompt 跳过", flush=True)
            continue
        for p in PROMPTS:
            tag = f"A · C0-base/{short}__{p}"
            outdir = os.path.join(OUTBASE_C0, f"{short}__{p}")
            cmd = [PY, PROBE, "--model", mpath, "--outdir", outdir,
                   "--batch_size", str(bs), "--prompt", p]
            tasks.append((tag, outdir, cmd, check_done_probe, True))
    return tasks


def plan_phase_b():
    """Phase B: B4 跨家族 A-sweep  7 instruct × poc 自发编造."""
    tasks = []
    for short, mpath, bs in INSTRUCT_MODELS:
        if os.path.sep in mpath and not os.path.isdir(mpath):
            print(f"[plan][B] {short}: 本地模型缺失 ({mpath}), 跳过", flush=True)
            continue
        tag = f"B · B4_a_sweep/{short}"
        outdir = os.path.join(OUTBASE_B4, short)
        # poc.py 生成阶段更耗存; 大模型 batch 减半保 8GB 显存
        gen_bs = max(4, bs // 2)
        cmd = [PY, POC, "--model", mpath, "--outdir", outdir,
               "--batch_size", str(gen_bs)]
        tasks.append((tag, outdir, cmd, check_done_poc, True))
    return tasks


def plan_phase_c():
    """Phase C: C1 / C1.1 / C1-control / C2 / C2-multiseed / C3 / C1-vertical.

    主靶 OLMo-2-0425-1B-Instruct (untied + C0 sweep 选出 0.68 天花板靶, 见 §C1).
    横向: pythia-1.4b (untied + base 模型) 作 C2 普适性验证 (论文级"跨模型 robust").
    c1_lora.py 自带断点重跑 (adapter/ep{N}/ + final), cleanup_partial=False 保 adapter.

    段落:
    - **已跑 10 项** (OLMo + pythia × 5 cell): C1 main / C1.1 vocab / control 2×2 / C2 lm_head
    - **C2-multiseed 4 项** (2026-06-10 立): OLMo+pythia × seed=1,2 把 C2 主结论从 seed=0 单点
      升到 3 seed × 2 模型 (见 §C3 backlog)
    - **C3 10 项** (2026-06-10 立, 升级 (iii) 到 transformer 中间层 hidden, 见 §C3):
      - OLMo 深度剖面: L ∈ {0(sanity), 4, 8, 12, 16(末)} × P4 = 5 跑
      - OLMo prompt 扫 (L=16 末层固定): {chat, P1, P3} = 3 跑 (P4 已在深度剖面)
      - pythia 跨模型: L ∈ {12, 24} × P4 = 2 跑
    - **C1-vertical 1 项** (2026-06-10 立): OLMo r=64 ep=30 n_train=800 看能否逼近 C0 上界 0.68

    每项参数 dict 形式. 默认 hyperparam (n_train=400, n_test=120, ep=15, lora_r=32,
    lora_alpha=64, lr=2e-4, max_new=250); 显式覆盖只写差异.
    """
    olmo_path   = os.path.join(HERE, "models", "OLMo-2-0425-1B-Instruct")
    pythia_path = os.path.join(HERE, "models", "pythia-1.4b")

    def _outdir(section_short, run_short):
        return os.path.join(HERE, "results", "data", section_short, run_short)

    # 默认 hyperparam, 显式覆盖时由 spec.get 拿
    defaults = {
        "vocab": "random", "target_mode": "real", "target_source": "input_embed",
        "seed": 0, "epochs": 15, "lr": "2e-4",
        "lora_r": 32, "lora_alpha": 64,
        "n_train": 400, "n_test": 120, "max_new": 250,
        "target_layer": None, "target_prompt": None,
        "target_model": None,  # C4: 跨模型 target 时给 path
        "intersect_tokenizer": None,  # C4-control: 强制双向交集词集 (跟 target_model 解耦)
    }
    # (tag_prefix, section_short, run_short, mpath, override_dict)
    runs = [
        # ---------- 已跑过的 10 项 (OLMo + pythia × 5 cell), 默认 hyperparam ----------
        ("C1", "C1_lora_finetune", "olmo2-1b_random_s0",             olmo_path,   {}),
        ("C1", "C1_lora_finetune", "olmo2-1b_basic_real_s0",         olmo_path,   {"vocab": "basic"}),
        ("C1", "C1_lora_finetune", "olmo2-1b_random_random_s0",      olmo_path,   {"target_mode": "random"}),
        ("C1", "C1_lora_finetune", "olmo2-1b_basic_random_s0",       olmo_path,   {"vocab": "basic", "target_mode": "random"}),
        ("C2", "C2_hard_target",   "olmo2-1b_random_real_lmhead_s0", olmo_path,   {"target_source": "lm_head"}),
        ("C1", "C1_lora_finetune", "pythia-1.4b_random_s0",             pythia_path, {}),
        ("C1", "C1_lora_finetune", "pythia-1.4b_basic_real_s0",         pythia_path, {"vocab": "basic"}),
        ("C1", "C1_lora_finetune", "pythia-1.4b_random_random_s0",      pythia_path, {"target_mode": "random"}),
        ("C1", "C1_lora_finetune", "pythia-1.4b_basic_random_s0",       pythia_path, {"vocab": "basic", "target_mode": "random"}),
        ("C2", "C2_hard_target",   "pythia-1.4b_random_real_lmhead_s0", pythia_path, {"target_source": "lm_head"}),
        # ---------- C2-multiseed 4 项 (seed=1,2 × OLMo+pythia × lm_head 同 cell) ----------
        ("C2m", "C2_hard_target", "olmo2-1b_random_real_lmhead_s1",    olmo_path,   {"target_source": "lm_head", "seed": 1}),
        ("C2m", "C2_hard_target", "olmo2-1b_random_real_lmhead_s2",    olmo_path,   {"target_source": "lm_head", "seed": 2}),
        ("C2m", "C2_hard_target", "pythia-1.4b_random_real_lmhead_s1", pythia_path, {"target_source": "lm_head", "seed": 1}),
        ("C2m", "C2_hard_target", "pythia-1.4b_random_real_lmhead_s2", pythia_path, {"target_source": "lm_head", "seed": 2}),
        # ---------- C3 10 项: OLMo 深度剖面 5 + OLMo prompt 扫 3 + pythia 跨模型 2 ----------
        # OLMo 深度剖面 (P4 固定, L 扫)
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L0_P4_s0",  olmo_path, {"target_source": "hidden", "target_layer": 0,  "target_prompt": "P4"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L4_P4_s0",  olmo_path, {"target_source": "hidden", "target_layer": 4,  "target_prompt": "P4"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L8_P4_s0",  olmo_path, {"target_source": "hidden", "target_layer": 8,  "target_prompt": "P4"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L12_P4_s0", olmo_path, {"target_source": "hidden", "target_layer": 12, "target_prompt": "P4"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L16_P4_s0", olmo_path, {"target_source": "hidden", "target_layer": 16, "target_prompt": "P4"}),
        # OLMo prompt 扫 (L=16 末层固定, 3 prompt; P4 已在上面)
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L16_chat_s0", olmo_path, {"target_source": "hidden", "target_layer": 16, "target_prompt": "chat"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L16_P1_s0",   olmo_path, {"target_source": "hidden", "target_layer": 16, "target_prompt": "P1"}),
        ("C3", "C3_deep_hidden", "olmo2-1b_random_real_hidden_L16_P3_s0",   olmo_path, {"target_source": "hidden", "target_layer": 16, "target_prompt": "P3"}),
        # pythia 跨模型 (P4 固定, L 扫中+末)
        ("C3", "C3_deep_hidden", "pythia-1.4b_random_real_hidden_L12_P4_s0", pythia_path, {"target_source": "hidden", "target_layer": 12, "target_prompt": "P4"}),
        ("C3", "C3_deep_hidden", "pythia-1.4b_random_real_hidden_L24_P4_s0", pythia_path, {"target_source": "hidden", "target_layer": 24, "target_prompt": "P4"}),
        # ---------- C1-vertical 1 项 (OLMo, r=64 ep=30 n_train=800) ----------
        ("C1v", "C1_lora_finetune", "olmo2-1b_random_r64_ep30_n800_s0", olmo_path,
         {"lora_r": 64, "lora_alpha": 128, "epochs": 30, "n_train": 800}),
        # ---------- C4 4 项: 跨模型 target (排除 H3 "通用语义→向量回归") ----------
        # OLMo FT, target=pythia 的嵌入/lm_head
        ("C4", "C4_cross_model_target", "olmo2-1b_FT_to_pythia-1.4b_ie_s0", olmo_path,
         {"target_model": pythia_path, "target_source": "input_embed"}),
        ("C4", "C4_cross_model_target", "olmo2-1b_FT_to_pythia-1.4b_lh_s0", olmo_path,
         {"target_model": pythia_path, "target_source": "lm_head"}),
        # pythia FT, target=OLMo 的嵌入/lm_head (双向对照)
        ("C4", "C4_cross_model_target", "pythia-1.4b_FT_to_olmo2-1b_ie_s0", pythia_path,
         {"target_model": olmo_path, "target_source": "input_embed"}),
        ("C4", "C4_cross_model_target", "pythia-1.4b_FT_to_olmo2-1b_lh_s0", pythia_path,
         {"target_model": olmo_path, "target_source": "lm_head"}),
        # ---------- C4-control 4 项: self target + 双向交集词集 (排除 C4 cross-model 的词集 confound) ----------
        ("C4c", "C4_cross_model_target", "olmo2-1b_self_ie_intersect_pythia_s0", olmo_path,
         {"intersect_tokenizer": pythia_path}),
        ("C4c", "C4_cross_model_target", "olmo2-1b_self_lh_intersect_pythia_s0", olmo_path,
         {"target_source": "lm_head", "intersect_tokenizer": pythia_path}),
        ("C4c", "C4_cross_model_target", "pythia-1.4b_self_ie_intersect_olmo_s0", pythia_path,
         {"intersect_tokenizer": olmo_path}),
        ("C4c", "C4_cross_model_target", "pythia-1.4b_self_lh_intersect_olmo_s0", pythia_path,
         {"target_source": "lm_head", "intersect_tokenizer": olmo_path}),
    ]

    tasks = []
    for tag_prefix, section_short, run_short, mpath, override in runs:
        if not os.path.isdir(mpath):
            print(f"[plan][C] {run_short}: 模型缺失 ({mpath}), 跳过", flush=True)
            continue
        spec = {**defaults, **override}
        outdir = _outdir(section_short, run_short)
        cmd = [PY, C1, "--model", mpath, "--outdir", outdir,
               "--train_vocab", spec["vocab"],
               "--target_mode", spec["target_mode"],
               "--target_source", spec["target_source"],
               "--seed", str(spec["seed"]),
               "--epochs", str(spec["epochs"]),
               "--lr", str(spec["lr"]),
               "--lora_r", str(spec["lora_r"]),
               "--lora_alpha", str(spec["lora_alpha"]),
               "--n_train", str(spec["n_train"]),
               "--n_test", str(spec["n_test"]),
               "--max_new", str(spec["max_new"])]
        if spec["target_source"] == "hidden":
            cmd += ["--target_layer", str(spec["target_layer"]),
                    "--target_prompt", spec["target_prompt"]]
        if spec["target_model"] is not None:
            cmd += ["--target_model", spec["target_model"]]
        if spec["intersect_tokenizer"] is not None:
            cmd += ["--intersect_tokenizer", spec["intersect_tokenizer"]]
        tasks.append((f"C · {tag_prefix}/{run_short}", outdir, cmd, check_done_c1, False))
    return tasks


def plan_phase_c5():
    """Phase C5: capability preservation eval batch. **必须在 Phase G 之后跑**,
    因为 batch list 包含 G1/G2 训出来的 adapter, 它们要先就位. (2026-06-15 拆出来,
    旧版混在 Phase C 里跑会漏 G1/G2 adapter — 见用户 catch).
    """
    return _plan_c5_cells()


def _plan_c5_cells():
    """C5 cell (capability preservation, H0 排除) - 跟 Phase C 的 c1_lora cell 合并跑.

    **2026-06-15 batch 列表 · 全 FT adapter + 2 base** (用户决定一次性跑完, 离线
    HF_DATASETS_OFFLINE=1, 数据集已缓存). 单 cell ~30 min, 全表 ~50 cell 估 25h
    隔夜两轮跑完. 任一 adapter 缺失自动 skip, 不阻塞.

    每 cell 跑 5 task: lambada_openai / hellaswag / piqa / arc_easy / winogrande.
    base vs LoRA 对比 |Δ acc| 判通用能力是否退化 (见 §C5).

    runs 表 = (run_short, base_model, adapter_outbase, adapter_run_short)
    adapter_outbase=None 表示 base 模型 (无 adapter); 否则 adapter 路径 =
      {adapter_outbase}/{adapter_run_short}/adapter/final
    """
    olmo_path   = os.path.join(HERE, "models", "OLMo-2-0425-1B-Instruct")
    pythia_path = os.path.join(HERE, "models", "pythia-1.4b")

    runs = []

    # ---------- base 模型 (2 cell) ----------
    runs += [
        ("olmo2-1b_base_s0",    olmo_path,   None, None),
        ("pythia-1.4b_base_s0", pythia_path, None, None),
    ]

    # ---------- C1 家族 (4 vocab × 2 模型 = 8 cell) ----------
    # vocab_kind: random (=C1 main, real 省略) / basic_real / random_random (control) / basic_random (control)
    for m_short, m_path in [("olmo2-1b", olmo_path), ("pythia-1.4b", pythia_path)]:
        for vocab_kind in ["random", "basic_real", "random_random", "basic_random"]:
            runs.append((f"{m_short}_LoRA_C1_{vocab_kind}_s0", m_path, OUTBASE_C1,
                         f"{m_short}_{vocab_kind}_s0"))

    # ---------- C1-vertical (1 cell) ----------
    runs.append(("olmo2-1b_LoRA_C1vert_s0", olmo_path, OUTBASE_C1,
                 "olmo2-1b_random_r64_ep30_n800_s0"))

    # ---------- C2 + C2-multiseed (3 seed × 2 模型 = 6 cell) ----------
    for m_short, m_path in [("olmo2-1b", olmo_path), ("pythia-1.4b", pythia_path)]:
        for seed in [0, 1, 2]:
            adapter_short = f"{m_short}_random_real_lmhead_s{seed}"
            runs.append((f"{m_short}_LoRA_C2_s{seed}", m_path, OUTBASE_C2, adapter_short))

    # ---------- C3 OLMo 深度剖面 + prompt 扫 (8 cell) ----------
    for L in [0, 4, 8, 12, 16]:
        runs.append((f"olmo2-1b_LoRA_C3_L{L}_P4_s0", olmo_path, OUTBASE_C3,
                     f"olmo2-1b_random_real_hidden_L{L}_P4_s0"))
    for prompt in ["chat", "P1", "P3"]:
        runs.append((f"olmo2-1b_LoRA_C3_L16_{prompt}_s0", olmo_path, OUTBASE_C3,
                     f"olmo2-1b_random_real_hidden_L16_{prompt}_s0"))

    # ---------- C3 pythia (2 cell) ----------
    for L in [12, 24]:
        runs.append((f"pythia-1.4b_LoRA_C3_L{L}_P4_s0", pythia_path, OUTBASE_C3,
                     f"pythia-1.4b_random_real_hidden_L{L}_P4_s0"))

    # ---------- C4 + C4-control (8 cell, source 模型 LoRA × {target=other, intersect}) ----------
    # 注: C4-control intersect 后缀实际目录名 OLMo 用 "pythia", pythia 用 "olmo" (不带 2)
    c4_specs = [
        ("olmo2-1b",   olmo_path,   "pythia-1.4b", "pythia"),
        ("pythia-1.4b", pythia_path, "olmo2-1b",    "olmo"),
    ]
    for src_short, src_path, tgt_short, intersect_tag in c4_specs:
        for kind in ["ie", "lh"]:
            runs.append((f"{src_short}_LoRA_C4_to_{tgt_short}_{kind}_s0", src_path, OUTBASE_C4,
                         f"{src_short}_FT_to_{tgt_short}_{kind}_s0"))
            runs.append((f"{src_short}_LoRA_C4c_self_{kind}_int_{intersect_tag}_s0",
                         src_path, OUTBASE_C4,
                         f"{src_short}_self_{kind}_intersect_{intersect_tag}_s0"))

    # ---------- G1 (6 cell: self/synonym/definition × 2 模型) ----------
    for m_short, m_path in [("olmo2-1b", olmo_path), ("pythia-1.4b", pythia_path)]:
        for mode in ["self", "synonym", "definition"]:
            runs.append((f"{m_short}_LoRA_G1_{mode}_s0", m_path, OUTBASE_G1,
                         f"{m_short}_{mode}_s0"))

    # ---------- G2 (3 metric × 2 模型 = 6 cell) ----------
    for m_short, m_path in [("olmo2-1b", olmo_path), ("pythia-1.4b", pythia_path)]:
        for metric in ["l2norm", "pca_recon_err", "tokenid_binary"]:
            runs.append((f"{m_short}_LoRA_G2_{metric}_s0", m_path, OUTBASE_G2,
                         f"{m_short}_{metric}_s0"))

    tasks = []
    for run_short, mpath, adapter_outbase, adapter_short in runs:
        if not os.path.isdir(mpath):
            print(f"[plan][C5] {run_short}: 模型缺失 ({mpath}), 跳过", flush=True)
            continue
        adapter_final = None
        if adapter_outbase is not None:
            adapter_final = os.path.join(adapter_outbase, adapter_short, "adapter", "final")
            if not os.path.isdir(adapter_final):
                print(f"[plan][C5] {run_short}: adapter 未就位 ({adapter_final}), 暂跳过",
                      flush=True)
                continue
        outdir = os.path.join(OUTBASE_C5, run_short)
        cmd = [PY, C5, "--model", mpath, "--outdir", outdir]
        if adapter_final:
            cmd += ["--adapter", adapter_final]
        tasks.append((f"C · C5/{run_short}", outdir, cmd, check_done_c5, True))
    return tasks


def _g1_align_self_with_synonym(g1_outbase, m_short):
    """G1 self cell adapter 复用 synonym cell adapter, 保 fair compare.

    背景 (2026-06-15 用户 catch 修复): 旧版 systematic_vocab_words 依赖
    `tokenizer.get_vocab()` dict iter order, 跨 transformers/hub 版本不稳 → 同 seed
    不同库版本给不同词集 (G1 self 在 hub 1.x 时代跑出 ['chicas',...], 而 G1
    synonym/definition 在 hub 0.35.x 时代跑出 ['relations',...]). c1_lora 已修成
    sorted by token_id 防再发生; 但已跑 cells 词集 freeze 在 raw/pca_targets.npz 里.

    本 helper 让 G1 self 复用 G1 synonym 的 adapter + pca_targets: synonym 模式
    训出来的 adapter 跟 self 模式训出来的 bit-exact 一样 (训练阶段不依赖 query_mode,
    仅 eval 阶段替换 prompt), 复用后 c1_lora 看 adapter/final → skip training →
    直接走 base + FT eval (mode=self) → 输出 fair self HE on 同 synonym 词集.

    无 synonym adapter 不动. self 已就位不动. 安全 idempotent.
    """
    self_dir = os.path.join(g1_outbase, f"{m_short}_self_s0")
    syn_dir  = os.path.join(g1_outbase, f"{m_short}_synonym_s0")
    syn_adapter_final = os.path.join(syn_dir, "adapter", "final")
    syn_pca = os.path.join(syn_dir, "raw", "pca_targets.npz")
    self_adapter_final = os.path.join(self_dir, "adapter", "final")
    if not (os.path.isdir(syn_adapter_final) and os.path.exists(syn_pca)):
        return  # synonym 还没跑完, 不动 self
    if os.path.isdir(self_adapter_final):
        return  # self adapter 已就位 (人工 cp 或之前 align 过), 不重复 cp
    print(f"[g1-align] {m_short}: synonym cell adapter 就位, 自动 cp 到 self cell "
          f"(保 fair compare)", flush=True)
    os.makedirs(os.path.join(self_dir, "raw"), exist_ok=True)
    shutil.copy2(syn_pca, os.path.join(self_dir, "raw", "pca_targets.npz"))
    self_adapter_dir = os.path.join(self_dir, "adapter")
    syn_adapter_dir  = os.path.join(syn_dir, "adapter")
    if not os.path.isdir(self_adapter_dir):
        shutil.copytree(syn_adapter_dir, self_adapter_dir)


def plan_phase_g():
    """Phase G: 判决性实验 (Falsification, 见 lab-notes/G族_判决性实验.md).

    包含:
    - **G1** (6 cell): 非激活查询 (self / synonym / definition 评估模式, 验证 Introspection 假说)
    - **G2** (2 cell): 物理目标 (target = l2norm 单 float, 验证物理参数可访问性与语义同构脱耦)
    - **G3** (1 cell): 跨模型 input_embed / lm_head 静态 RDM RSA (量化 C4 漏洞).
    """
    olmo_path   = os.path.join(HERE, "models", "OLMo-2-0425-1B-Instruct")
    pythia_path = os.path.join(HERE, "models", "pythia-1.4b")

    tasks = []

    # ---------- G1 Tasks (非激活查询) ----------
    # self 是 baseline (prompt 含目标 token); synonym/definition 用 WordNet 词典替换 prompt 里的词,
    # ground-truth 仍按原词嵌入. 词典走 materials/G1_queries/{model_short}_{mode}.json,
    # model_short = "olmo" / "pythia" (不带 -2-1b / -1.4b 后缀, 跟 tools/g1_build_queries.py 一致).
    for m_short, m_path, dict_tag in [("olmo2-1b", olmo_path, "olmo"),
                                       ("pythia-1.4b", pythia_path, "pythia")]:
        if not os.path.isdir(m_path):
            print(f"[plan][G1] {m_short}: 模型缺失 ({m_path}), 跳过", flush=True)
            continue
        # 自动 align G1 self cell adapter 复用 synonym (保 fair compare). 详见 helper.
        _g1_align_self_with_synonym(OUTBASE_G1, m_short)
        for mode in ["self", "synonym", "definition"]:
            run_short = f"{m_short}_{mode}_s0"
            outdir = os.path.join(OUTBASE_G1, run_short)
            cmd = [PY, C1, "--model", m_path, "--outdir", outdir,
                   "--eval_query_mode", mode,
                   "--train_vocab", "random",
                   "--target_mode", "real",
                   "--target_source", "input_embed",
                   "--seed", "0"]
            if mode != "self":
                dict_path = os.path.join(HERE, "materials", "G1_queries",
                                          f"{dict_tag}_{mode}.json")
                if not os.path.exists(dict_path):
                    print(f"[plan][G1] {run_short}: 词典缺失 ({dict_path}), 跳过 "
                          f"(先跑 tools/g1_build_queries.py 生成)", flush=True)
                    continue
                cmd += ["--query_dict_path", dict_path]
            tasks.append((f"G · G1/{run_short}", outdir, cmd, check_done_c1, False))

    # ---------- G2 Tasks (物理目标 · 3 个 metric × 2 模型 = 6 cell) ----------
    # pca 基线对应主实验 C1, G2 跑物理目标:
    #   (i)   l2norm          - 单 float ||emb||_2 (跟词频强相关, 需 freq baseline)
    #   (ii)  pca_recon_err   - 单 float ||emb - PCA-8(emb)|| (主成分外残差)
    #   (iii) tokenid_binary  - 16 floats binary(token_id) (完全无语义, 关键 disjudicator)
    g2_specs = [
        ("olmo2-1b_l2norm_s0",         olmo_path,   "l2norm",         "1"),
        ("olmo2-1b_pca_recon_err_s0",  olmo_path,   "pca_recon_err",  "1"),
        ("olmo2-1b_tokenid_binary_s0", olmo_path,   "tokenid_binary", "16"),
        ("pythia-1.4b_l2norm_s0",         pythia_path, "l2norm",         "1"),
        ("pythia-1.4b_pca_recon_err_s0",  pythia_path, "pca_recon_err",  "1"),
        ("pythia-1.4b_tokenid_binary_s0", pythia_path, "tokenid_binary", "16"),
    ]
    for run_short, mpath, metric, pca_dim in g2_specs:
        if not os.path.isdir(mpath):
            print(f"[plan][G2] {run_short}: 模型缺失 ({mpath}), 跳过", flush=True)
            continue
        outdir = os.path.join(OUTBASE_G2, run_short)
        cmd = [PY, C1, "--model", mpath, "--outdir", outdir,
               "--target_metric", metric,
               "--pca_dim", pca_dim,
               "--train_vocab", "random",
               "--target_mode", "real",
               "--target_source", "input_embed",
               "--seed", "0"]
        tasks.append((f"G · G2/{run_short}", outdir, cmd, check_done_c1, False))

    # ---------- G3 Task (静态 RDM RSA) ----------
    if os.path.isdir(olmo_path) and os.path.isdir(pythia_path):
        outdir = os.path.join(OUTBASE_G3, "olmo_vs_pythia_s0")
        cmd = [PY, G3, "--model_a", olmo_path, "--model_b", pythia_path,
               "--outdir", outdir, "--n_words", "300", "--seed", "0"]
        tasks.append(("G · G3/olmo_vs_pythia_s0", outdir, cmd, check_done_g3, True))
    else:
        print(f"[plan][G] OLMo 或 pythia 模型缺失, G3 跳过", flush=True)

    return tasks


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["A", "B", "C", "G", "C5", "all"],
                    help="只跑某段 (A=C0-base 延伸; B=B4 A-sweep; C=C 族 c1_lora cells "
                         "(不含 C5); G=判决性实验 G1/G2/G3; C5=capability eval batch "
                         "全 FT adapter + 2 base, 必须 G 跑完后再跑才能覆盖 G1/G2; "
                         "all=A→B→C→G→C5 顺序)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印计划 (skip/run) 不实际启动子进程, 用来预览")
    ap.add_argument("--logfile", default="_run_pending.log",
                    help="脚本会用 utf-8 把 stdout 镜像写到这里, 避开 PowerShell `>` 重定向坑. "
                         "传空串 '' 关闭文件镜像")
    args = ap.parse_args()

    # 开 Tee 日志 (utf-8, append, 即时 flush)
    log_handle = None
    if args.logfile and not args.dry_run:
        log_handle = open(os.path.join(HERE, args.logfile), "a", encoding="utf-8", buffering=1)
        log_handle.write(f"\n\n{'#'*72}\n# new run @ {datetime.now().isoformat(timespec='seconds')}\n{'#'*72}\n")
        sys.stdout = _Tee(sys.__stdout__, log_handle)

    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n[start] {started}  phase={args.phase}")
    print(f"[env]   PY={PY}")
    print(f"[env]   HERE={HERE}\n")

    tasks = []
    # 'all' 顺序: A → B → C → G → C5. C5 必须最后, 因为 batch list 含 G1/G2 adapter
    if args.phase in ("A", "all"):
        tasks.extend(plan_phase_a())
    if args.phase in ("B", "all"):
        tasks.extend(plan_phase_b())
    if args.phase in ("C", "all"):
        tasks.extend(plan_phase_c())  # 仅 c1_lora cells, 不含 C5
    if args.phase in ("G", "all"):
        tasks.extend(plan_phase_g())
    if args.phase in ("C5", "all"):
        tasks.extend(plan_phase_c5())  # 全 FT adapter + 2 base, 覆盖 C+G 训出来的

    n = len(tasks)
    print(f"[plan]  共 {n} 项 (含已完成会自动 skip).\n")

    if args.dry_run:
        print("[dry-run] 计划预览 (不实际跑):")
        n_done = n_todo = n_partial = 0
        for i, (tag, outdir, cmd, check, cleanup_p) in enumerate(tasks, 1):
            if check(outdir):
                state = "skip(完成)"; n_done += 1
            elif os.path.isdir(outdir):
                state = "续跑(半成品)" if not cleanup_p else "重跑(半成品)"
                n_partial += 1
            else:
                state = "新跑"; n_todo += 1
            print(f"  [{i:2d}/{n}] {state:12s}  {tag}")
        print(f"\n[dry-run] 汇总: 已完成 skip={n_done}, 新跑={n_todo}, 半成品={n_partial}")
        return

    counts = {"ok": 0, "skipped": 0, "fail": 0}
    timings = []
    rows = []
    for i, (tag, outdir, cmd, check, cleanup_p) in enumerate(tasks, 1):
        status, dt = run_one(i, n, tag, outdir, cmd, check, cleanup_partial=cleanup_p)
        counts[status] += 1
        timings.append(dt)
        rows.append({"tag": tag, "outdir": os.path.relpath(outdir, HERE),
                     "status": status, "secs": round(dt, 1)})
        # 每完成一项立即写汇总, 中断也保现场
        with open(SUMMARY, "w", encoding="utf-8") as f:
            json.dump({"started": started,
                       "updated": datetime.now().isoformat(timespec="seconds"),
                       "phase": args.phase, "counts": counts,
                       "elapsed_min": round(sum(timings) / 60, 1),
                       "rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"[done]  phase={args.phase}  counts={counts}  elapsed={sum(timings)/60:.1f}min")
    print(f"[done]  summary -> {os.path.relpath(SUMMARY, HERE)}")
    if counts["fail"]:
        print(f"[done]  ⚠ 有 {counts['fail']} 项失败, 重跑会自动重试")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()