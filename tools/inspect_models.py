"""读取本地 models/ 下每个模型的 config，打印归一化属性(来源/tie_emb/维度/架构)。
只读 config 不加载权重, 秒级。"""
import os
from transformers import AutoConfig

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SRC = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct (curl 组装)",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "gpt2-large": "openai-community/gpt2-large",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
    "OLMo-2-0425-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct (免门禁镜像)",
}

for d in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, d)
    if not os.path.isdir(p):
        continue
    src = SRC.get(d, "?")
    try:
        c = AutoConfig.from_pretrained(p)
        cc = getattr(c, "text_config", None) or c   # gemma3 等可能嵌套
        hidden = getattr(cc, "hidden_size", "?")
        layers = getattr(cc, "num_hidden_layers", "?")
        vocab = getattr(cc, "vocab_size", getattr(c, "vocab_size", "?"))
        tie = getattr(c, "tie_word_embeddings", getattr(cc, "tie_word_embeddings", None))
        arch = (c.architectures or ["?"])[0]
        print(f"{d:30s} | {c.model_type:10s} | hidden={hidden} layers={layers} vocab={vocab} "
              f"| tie_emb={tie} | {arch}")
        print(f"{'':30s} | 来源: {src}")
    except Exception as e:
        print(f"{d:30s} | CONFIG ERR: {type(e).__name__}: {str(e)[:90]} | 来源: {src}")
