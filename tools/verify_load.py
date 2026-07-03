"""验证本地组装的模型能加载且权重完整（生成一句连贯输出即证明 shard 拼接无误）。"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "Qwen2.5-3B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[load] {path}")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).half().to(device).eval()
n = sum(p.numel() for p in model.parameters())
print(f"[ok] params={n/1e9:.2f}B  hidden={model.config.hidden_size}  layers={model.config.num_hidden_layers}  "
      f"tie_emb={getattr(model.config,'tie_word_embeddings',None)}")

prompt_text = "In one short sentence, what is a word embedding?"
if getattr(tok, "chat_template", None):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                  add_generation_prompt=True, return_tensors="pt").to(device)
else:  # base 模型无 chat_template -> 纯文本续写 (c1_lora build_prompt 同款回退)
    ids = tok(f"Q: {prompt_text}\nA:", return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
print("[gen]", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
print("[verdict] 若上句连贯 -> 权重完整, 3B 可用。")
