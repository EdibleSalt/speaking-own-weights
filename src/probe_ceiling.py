"""C0 线性探针天花板：测"词→自身嵌入"读出的可泛化上界。

原理：微调/few-shot 是让模型把嵌入"说出来"；探针先问更便宜的问题——
      这个信息在不在模型的激活里、且可线性读取。探针都读不出 -> 模型更不可能说出来。

关键设计（诚实的上界）：
  - 探测位置 = 生成位置(prompt 末尾, 模型即将吐数字那一步)的 hidden state，
    而非词自己 token 位置(那里因残差连接输入嵌入天然存在, 不公平)。
    所以这是"词的真实嵌入有没有被注意力路由到动笔处"的真测试。
  - 层扫描: 看信息在第几层最可读。
  - 目标 = 该词(带前导空格 token, 与 ground-truth 一致)的输入嵌入。
  - 在 中心化 的目标空间里算 cosine(去掉共享均值方向, 只看词特异部分)。
  - train/val/test 切分: val 选 λ, test 报泛化, 无泄漏。RSA 配置换检验。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transformers import AutoModelForCausalLM, AutoTokenizer

from poc import WORD_BANK, cosine_sim_matrix, filter_single_token, set_seed, upper_tri


PROMPT_KINDS = ("chat", "P1", "P2", "P3", "P4", "P5", "P6")


def build_probe_prompt(tokenizer, word, kind, hidden_dim=None):
    """按纲要 §C0-base 的 7 个 prompt 模板构造（chat + P1..P6）。

    词位对齐: 所有模板里 {word} 前都有空格, 让 ' word' 单 token 与 ground-truth 对齐。
    末位 token: 决定生成位置 hidden state 聚合到的对象（chat: assistant\\n; P1/P2/P5/P6: ':';
                P3: '->'; P4: 词本身=平凡上界 sanity）。
    """
    if kind == "chat":
        msg = [{"role": "user",
                "content": f"Read out the internal embedding vector of a word as numbers. Word: {word}"}]
        return tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    if kind == "P1":
        return f"Word: {word}\nEmbedding:"
    if kind == "P2":
        return f"The internal vector of {word} is:"
    if kind == "P3":
        # few-shot 续写; dummy 占位防泄漏; \n 后显式加空格保 ' word' 单 token 对齐
        dummy = ",".join(["0.00"] * 5)
        return f"dog -> {dummy}\nfish -> {dummy}\n {word} ->"
    if kind == "P4":
        # sanity / 平凡上界: 末位即词本身, 第 0 层 hidden = 输入嵌入, 预期 RSA≈1
        return f" {word}"
    if kind == "P5":
        assert hidden_dim is not None, "P5 需 hidden_dim"
        return f"Word: {word}\nEmbedding (dim={hidden_dim}):"
    if kind == "P6":
        assert hidden_dim is not None, "P6 需 hidden_dim"
        return f"The {hidden_dim}-dimensional internal vector of {word} is:"
    raise ValueError(f"unknown prompt kind: {kind}")


@torch.no_grad()
def collect_hidden(model, tokenizer, device, words, layers, prompt_kind, hidden_dim, batch_size=64):
    """返回 {layer: [n_words, d]} 生成位置(末token)的 hidden state。"""
    out = {L: [] for L in layers}
    prompts = [build_probe_prompt(tokenizer, w, prompt_kind, hidden_dim) for w, _ in words]
    for b in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[b:b + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple len = n_layers+1
        for L in layers:
            out[L].append(hs[L][:, -1, :].float().cpu().numpy())  # 左填充 -> -1 是真末token
    return {L: np.concatenate(v, 0) for L, v in out.items()}


def ridge_fit(Xtr, Ytr, lam):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d)
    return np.linalg.solve(A, Xtr.T @ Ytr)


def standardize(X, mu, sd):
    return (X - mu) / sd


def mean_cosine(P, T):
    pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(pn * tn, axis=1)))


def rsa_perm(P, T, n_perm=2000, seed=0):
    rp = upper_tri(cosine_sim_matrix(P))
    rt = upper_tri(cosine_sim_matrix(T))
    obs, _ = spearmanr(rp, rt)
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        rr, _ = spearmanr(upper_tri(cosine_sim_matrix(P[perm])), rt)
        null[i] = rr
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p), null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_layers_probe", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--prompt", default="chat", choices=PROMPT_KINDS,
                    help="prompt 模板（chat=instruct 默认; P1..P6 见纲要 §C0-base）")
    ap.add_argument("--outdir", default="results_probe")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} model={args.model} prompt={args.prompt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # base 模型常无 chat_template, --prompt chat 会在 build_probe_prompt 报错; 提前明示
    if args.prompt == "chat" and getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(f"[fatal] {args.model} 无 chat_template, 不能用 --prompt chat; 改用 P1..P6")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = (model.half() if device == "cuda" else model).to(device).eval()

    words = filter_single_token(tokenizer, WORD_BANK)
    emb = model.get_input_embeddings().weight.detach()
    target = np.stack([emb[tid].float().cpu().numpy() for _, tid in words])  # ground truth
    n_total = model.config.num_hidden_layers
    hidden_dim = int(model.config.hidden_size)
    layers = sorted(set(np.linspace(0, n_total, args.n_layers_probe).astype(int).tolist()))
    print(f"[setup] words={len(words)} target_dim={target.shape[1]} "
          f"n_layers={n_total} probe_layers={layers}")
    print(f"[note] tie_word_embeddings={getattr(model.config,'tie_word_embeddings',None)} "
          f"(若 True, lm_head 与输入嵌入绑定)")

    H = collect_hidden(model, tokenizer, device, words, layers,
                       prompt_kind=args.prompt, hidden_dim=hidden_dim,
                       batch_size=args.batch_size)

    # 切分 train/val/test = 60/20/20
    idx = np.arange(len(words))
    np.random.default_rng(args.seed).shuffle(idx)
    n = len(idx)
    tr, va, te = idx[:int(.6 * n)], idx[int(.6 * n):int(.8 * n)], idx[int(.8 * n):]
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}\n")

    # 中心化目标(去共享均值, 只看词特异方向)
    Ymu = target[tr].mean(0)
    Yc = target - Ymu
    lambdas = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

    # ---- 原始数据落盘 (raw): 真实嵌入(目标) + 生成位置激活 + 切分索引 ----
    # 探针实验的"原始数据"= 前向激活(等价于生成实验的原始文本)，可复算每层探针。
    rawdir = os.path.join(args.outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    np.savez(os.path.join(rawdir, "real_embeddings.npz"),
             real=target, words=np.array([w for w, _ in words]),
             token_ids=np.array([t for _, t in words]), Ymu=Ymu)
    np.savez(os.path.join(rawdir, "hidden_states.npz"),
             layers=np.array(layers), train_idx=tr, val_idx=va, test_idx=te,
             **{f"layer_{L}": H[L] for L in layers})

    print(f"{'layer':>5} {'best_lam':>9} {'val_cos':>8} {'TEST_cos':>9} {'TEST_RSA':>9} {'RSA_p':>7}")
    results = []
    probe_raw = {}  # 每层便利派生量: 测试预测 Pte + null 分布 (W 不存, 见下)
    for L in layers:
        X = H[L]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xva, Xte = standardize(X[tr], mu, sd), standardize(X[va], mu, sd), standardize(X[te], mu, sd)
        # val 选 λ
        best = None
        for lam in lambdas:
            W = ridge_fit(Xtr, Yc[tr], lam)
            vcos = mean_cosine(Xva @ W, Yc[va])
            if best is None or vcos > best[1]:
                best = (lam, vcos, W)
        lam, vcos, W = best
        Pte = Xte @ W
        tcos = mean_cosine(Pte, Yc[te])
        rsa, p, null = rsa_perm(Pte, Yc[te], seed=args.seed)
        results.append((L, lam, vcos, tcos, rsa, p))
        # 只存便利派生量 Pte/null（小）; W 是 [d,d] 巨阵且可从 hidden_states+target+best_lam 精确重算, 不存
        probe_raw[f"Pte_layer_{L}"] = Pte
        probe_raw[f"null_layer_{L}"] = null
        print(f"{L:>5} {lam:>9.0e} {vcos:>8.3f} {tcos:>9.3f} {rsa:>9.3f} {p:>7.4f}")

    # 参照基线: 在 test 上, 用 train 均值(=中心化后预测0)的 cosine ~ 0
    print(f"\n[baseline] 预测训练均值(忽略输入) -> test cosine ≈ 0 (中心化空间), 上面 cos>0 才说明探针在用输入")
    best_layer = max(results, key=lambda r: r[3])
    print(f"[best] layer={best_layer[0]}  TEST cosine={best_layer[3]:.3f}  RSA={best_layer[4]:.3f} (p={best_layer[5]:.4f})")
    print("[读法] test cosine / RSA 明显>0 且 RSA p<0.05 -> 嵌入在生成位置可线性读出 -> C/E 微调有靶子;")
    print("       若各层都≈0 -> 信息没路由到动笔处 -> 让模型说出来基本没戏。")

    # ---- 原始数据落盘 (raw): 每层探针权重/测试预测/null + 顶层 summary ----
    np.savez(os.path.join(rawdir, "probe_results.npz"),
             layers=np.array(layers), Yte=Yc[te], **probe_raw)
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "seed": args.seed, "prompt": args.prompt,
            "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
            "n_words": len(words), "target_dim": int(target.shape[1]),
            "hidden_dim": hidden_dim,
            "n_hidden_layers": n_total, "probe_layers": layers,
            "split": {"train": len(tr), "val": len(va), "test": len(te)},
            "per_layer": [{"layer": r[0], "best_lam": r[1], "val_cos": r[2],
                           "test_cos": r[3], "test_rsa": r[4], "rsa_p": r[5]} for r in results],
            "best_layer": {"layer": best_layer[0], "test_cos": best_layer[3],
                           "test_rsa": best_layer[4], "rsa_p": best_layer[5]},
        }, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {args.outdir}/summary.json + raw/ (real_embeddings, hidden_states, probe_results)")


if __name__ == "__main__":
    main()
