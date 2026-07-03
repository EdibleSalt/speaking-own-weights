r"""G1 辅助: 用 WordNet 给一组 held-out 词生成 query_dict.

输出 JSON: {word: synonym_or_definition_str}
- mode=synonym: 取第一个**真正切断激活路径**的 lemma (排除形态变体, 见下)
- mode=definition: 取 synset.definition() 作描述, 且**定义里不含原词作子串**

## 2026-06-16 反泄漏重写 (审计修复)

旧版两大泄漏:
  (1) WordNet 找不到 -> eval 时 fallback 回原词 = 没切断激活路径 (baseline 行为).
  (2) "同义词" 常是同根屈折形 (deny/denying, farmer/farmers, teach/teaches) ->
      跟原词共享 BPE 词干, 模型看到 prompt 里换的词照样能激活原词嵌入 = 也没切断.

本版改动:
  - find_synonym: **排除形态变体** —— 拒绝与原词 (a) 互为子串 / (b) 共享前 4 字符 /
    (c) 编辑距离 <= 2 / (d) 去掉常见后缀 (s/es/ed/ing/er/ly...) 后词干相同 的候选;
    并要求同义词本身是**干净单 token 词** (best-effort, 给了 tokenizer 才查).
  - find_definition: 确保定义文本里**不含原词作为子串** (大小写无关), 含则换下一个 synset.
  - --comprehensive: 从模型 tokenizer 枚举一大批干净单 token 词 (上限 --max_words),
    为每个生成 synonym/definition -> 一份词典覆盖多个 seed 的 test 词集
    (旧版 --pca_targets 只覆盖单个 cell 的 120 词).
  - --selfcheck: 用 c1_lora.systematic_vocab_words 抽 seed 0/1/2 的 120 词 test 集,
    报告新词典对这些词的"干净命中率" (命中且非形态变体的比例).

"干净单 token 词"的判据与 c1_lora.systematic_vocab_words 一致:
  前导空格单 token + 小写 ascii 字母 + len >= 3 + encode(' '+w) 回到同一个 token.

输出路径约定: `materials/G1_queries/{model_short}_{mode}[_clean].json`,
其中 model_short = "olmo" / "pythia" (跟 run_pending.plan_phase_g 读的路径一致).

用法:
    # 旧的单 cell 模式 (保留, 不变):
    python tools/g1_build_queries.py \
        --pca_targets .../raw/pca_targets.npz \
        --mode synonym --out materials/G1_queries/olmo_synonym.json

    # 新的 comprehensive 模式:
    python tools/g1_build_queries.py \
        --comprehensive --model models/pythia-1.4b \
        --mode synonym --max_words 4000 \
        --out materials/G1_queries/pythia_synonym_clean.json

    # 覆盖率自检 (对比旧 vs 新词典):
    python tools/g1_build_queries.py \
        --selfcheck --model models/pythia-1.4b \
        --old materials/G1_queries/pythia_synonym.json \
        --new materials/G1_queries/pythia_synonym_clean.json
"""
import argparse
import json
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


# ============================ 形态学工具 ============================
# 去后缀的顺序: 长后缀优先 (ing/ies 在 s 之前), 避免 'flies' -> 'flie'.
_SUFFIXES = ("ingly", "edly", "ies", "ing", "ied", "ed", "es", "er", "est",
             "ly", "s")


def _stem(w):
    """极简词干: 反复剥常见屈折后缀, 留下 >=3 字符的词干.

    不追求语言学正确, 只为"两个词是否同根"的判定够用.
    'denying'->'deny'(ing+y还原), 'farmers'->'farmer'? 见下: 'farmers'->'farmer'->'farm'.
    所以两边都 _stem 后比较, deny/denying / farmer/farmers 都会落到同一词干.
    """
    w = w.lower()
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stem = w[: -len(suf)]
                # ing/ed 常伴随 'y'->'i' 或末字母重复; 简单还原 i->y
                if suf in ("ies", "ied") and not stem.endswith("y"):
                    stem = stem + "y"
                w = stem
                changed = True
                break
    return w


def _edit_distance(a, b, cap=3):
    """Levenshtein, 带早停 (>cap 时返回 cap+1, 省时)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < row_min:
                row_min = v
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def is_morphological_variant(word, cand):
    """cand 是否为 word 的形态变体 (= 没真正切断激活路径). True 则拒绝.

    判据 (任一命中即视为变体):
      (a) 互为子串            cat/cats, scape/landscape
      (b) 共享前 4 字符        deny/denying, confirm/confirming
      (c) 编辑距离 <= 2        teach/teaches (dist 2), goat/goats (dist 1)
      (d) 去后缀词干相同      farmer/farmers, peaked/peak
    word/cand 都假定为小写单词 (无空格).
    """
    w, c = word.lower(), cand.lower()
    if w == c:
        return True
    # (a) 子串
    if w in c or c in w:
        return True
    # (b) 共享前 4 字符 (短词用 min 长度)
    k = min(4, len(w), len(c))
    if k >= 4 and w[:k] == c[:k]:
        return True
    # (c) 编辑距离 <= 2
    if _edit_distance(w, c, cap=2) <= 2:
        return True
    # (d) 去后缀词干相同
    if _stem(w) == _stem(c):
        return True
    return False


# ============================ 干净单 token 判据 ============================
def is_clean_word_str(w):
    """字符串层面的干净判据 (与 systematic_vocab_words 一致, 不查 tokenizer)."""
    return len(w) >= 3 and w.isascii() and w.isalpha() and w.islower()


def make_clean_single_token(tokenizer):
    """返回一个 is_clean(w)->bool 闭包: 干净单 token 词 (带 tokenizer 校验).

    与 c1_lora.systematic_vocab_words 同款: 前导空格单 token + 小写 ascii 字母
    + len>=3 + encode(' '+w) 恰好一个 token. tokenizer=None 时退化为纯字符串判据.
    """
    if tokenizer is None:
        return is_clean_word_str

    def is_clean(w):
        if not is_clean_word_str(w):
            return False
        try:
            ids = tokenizer.encode(" " + w, add_special_tokens=False)
        except Exception:
            return False
        return len(ids) == 1

    return is_clean


# ============================ 同义词 / 定义 ============================
def find_synonym(word, is_clean=is_clean_word_str):
    """第一个**非形态变体 + 干净单 token**的 lemma (跨 synset). 返回 str 或 None.

    与旧版本质区别:
      - 排除形态变体 (is_morphological_variant)
      - 要求 lemma 本身是干净单 token 词 (is_clean)
      - **不再 fallback 回原词** —— 找不到合格同义词就返回 None (eval 端按 miss 处理)
    """
    seen = set()
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if not name or name in seen:
                continue
            seen.add(name)
            if " " in name:           # 单词同义词优先, multi-word 跳过
                continue
            if is_morphological_variant(word, name):
                continue
            if not is_clean(name):
                continue
            return name
    return None


def find_definition(word):
    """第一个 synset 的 definition, 且**定义文本里不含原词作子串** (大小写无关).

    含原词的定义等于没切断 -> 换下一个 synset; 全都含则返回 None.
    """
    wl = word.lower()
    for ss in wn.synsets(word):
        d = ss.definition()
        if not d:
            continue
        if wl in d.lower():
            continue
        return d
    return None


# ============================ 词集来源 ============================
def words_from_pca(pca_targets):
    npz = np.load(pca_targets, allow_pickle=True)
    return [str(w) for w in npz["test_words"]]


def words_from_tokenizer(tokenizer, max_words):
    """枚举词表里的干净单 token 词 (排序稳定), 上限 max_words.

    与 systematic_vocab_words 的候选生成同款 (sorted by token_id), 但**不抽样**,
    取前 max_words 个 -> 覆盖多 seed 的 test 集.
    """
    vocab = tokenizer.get_vocab()
    out = []
    for tokstr, tid in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        s = tokenizer.convert_tokens_to_string([tokstr])
        if not s.startswith(" "):
            continue
        w = s[1:]
        if not (len(w) >= 3 and w.isascii() and w.isalpha() and w.islower()):
            continue
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1 and ids[0] == tid:
            out.append(w)
            if max_words and len(out) >= max_words:
                break
    return out


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def maybe_download_wordnet():
    """确保 WordNet 数据就位; 缺则只下数据集 (不 pip install). 失败抛错."""
    try:
        wn.synsets("cat")
        return
    except LookupError:
        import nltk
        print("[g1_build_queries] WordNet 数据缺, 尝试 nltk.download('wordnet') (仅数据集)...")
        ok = nltk.download("wordnet")
        if not ok:
            raise SystemExit("[fatal] nltk.download('wordnet') 失败 (无网络?). "
                             "请离线放置 wordnet 语料后重试; 禁止 pip install.")
        wn.synsets("cat")


# ============================ 构建 ============================
def build_dict(words, mode, is_clean):
    result = {}
    miss = []
    for w in words:
        if mode == "synonym":
            q = find_synonym(w, is_clean=is_clean)
        else:
            q = find_definition(w)
        if q is None:
            miss.append(w)
        else:
            result[w] = q
    return result, miss


# ============================ 自检 ============================
def _import_systematic():
    sys.path.insert(0, os.path.abspath(SRC))
    from c1_lora import systematic_vocab_words
    return systematic_vocab_words


def coverage_selfcheck(tokenizer, dicts, seeds=(0, 1, 2), n=120):
    """对 seed 0/1/2 的 test 词集, 报告每份词典的"干净命中率".

    dicts: {label: query_dict}. 命中 = word 在 dict 且 (synonym 模式) 非形态变体 /
    (这里统一按形态变体过滤, 因为 definition 已在构建期排除原词子串, 不会是变体).
    干净命中: word 在 dict 且 dict[word] 不是 word 的形态变体.
    """
    systematic_vocab_words = _import_systematic()
    rows = []
    for seed in seeds:
        pairs = systematic_vocab_words(tokenizer, n, seed, exclude_words=[])
        test_words = [w for w, _ in pairs]
        row = {"seed": seed, "n": len(test_words)}
        for label, d in dicts.items():
            hit = 0
            clean_hit = 0
            for w in test_words:
                if w in d:
                    hit += 1
                    q = str(d[w])
                    # 多词 query 必非形态变体; 单词 query 查变体
                    if " " in q or not is_morphological_variant(w, q):
                        clean_hit += 1
            row[label + "_hit"] = hit
            row[label + "_clean"] = clean_hit
            row[label + "_clean_rate"] = clean_hit / len(test_words) if test_words else 0.0
        rows.append(row)
    return rows


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca_targets", help="(单 cell 模式) raw/pca_targets.npz, 拿 test_words")
    ap.add_argument("--comprehensive", action="store_true",
                    help="(comprehensive 模式) 从 --model tokenizer 枚举干净单 token 词")
    ap.add_argument("--selfcheck", action="store_true",
                    help="(自检模式) 对比 --old / --new 词典对 seed0/1/2 test 集的干净命中率")
    ap.add_argument("--mode", choices=["synonym", "definition"])
    ap.add_argument("--model", help="模型目录 (comprehensive / selfcheck / 给 synonym 加干净校验)")
    ap.add_argument("--max_words", type=int, default=4000,
                    help="comprehensive 模式枚举词数上限")
    ap.add_argument("--out", help="输出 JSON 路径")
    ap.add_argument("--old", help="(selfcheck) 旧词典路径")
    ap.add_argument("--new", help="(selfcheck) 新词典路径")
    args = ap.parse_args()

    # ---- 自检模式 ----
    if args.selfcheck:
        if not args.model:
            raise SystemExit("--selfcheck 需要 --model")
        tok = load_tokenizer(args.model)
        dicts = {}
        if args.old and os.path.exists(args.old):
            dicts["old"] = json.load(open(args.old, encoding="utf-8"))
        if args.new and os.path.exists(args.new):
            dicts["new"] = json.load(open(args.new, encoding="utf-8"))
        if not dicts:
            raise SystemExit("--selfcheck 至少要 --old 或 --new 指向存在的文件")
        rows = coverage_selfcheck(tok, dicts)
        print(f"[selfcheck] model={args.model}")
        for r in rows:
            parts = [f"seed={r['seed']} (n={r['n']})"]
            for label in dicts:
                parts.append(f"{label}: clean {r[label+'_clean']}/{r['n']} "
                             f"= {r[label+'_clean_rate']*100:.1f}% (hit {r[label+'_hit']})")
            print("  " + " | ".join(parts))
        return

    # ---- 构建模式 (单 cell / comprehensive) ----
    if not args.mode or not args.out:
        raise SystemExit("构建模式需要 --mode 和 --out")
    maybe_download_wordnet()

    tokenizer = None
    if args.model:
        tokenizer = load_tokenizer(args.model)
    is_clean = make_clean_single_token(tokenizer)

    if args.comprehensive:
        if tokenizer is None:
            raise SystemExit("--comprehensive 需要 --model")
        words = words_from_tokenizer(tokenizer, args.max_words)
        print(f"[g1_build_queries] comprehensive: 枚举干净单 token 词 = {len(words)} "
              f"(上限 {args.max_words}), mode = {args.mode}")
    elif args.pca_targets:
        words = words_from_pca(args.pca_targets)
        print(f"[g1_build_queries] 单 cell: held-out 词数 = {len(words)}, mode = {args.mode}")
    else:
        raise SystemExit("构建模式需要 --comprehensive 或 --pca_targets")

    result, miss = build_dict(words, args.mode, is_clean)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[g1_build_queries] 覆盖 {len(result)}/{len(words)} 词 "
          f"(miss {len(miss)}); 写入 {args.out}")
    if miss[:10]:
        print(f"  miss 前 10 个: {miss[:10]}")
    if result:
        sample = list(result.items())[:8]
        print(f"  样例 8 条:")
        for w, q in sample:
            print(f"    {w!r:>16}  -> {q!r}")


if __name__ == "__main__":
    main()
