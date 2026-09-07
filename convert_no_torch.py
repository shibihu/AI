import os
import json
import requests
from safetensors import safe_open

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
OUTPUT_FILE = "smollm_135m.slm"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/raw/main"

print(f"--> Downloading config files for {MODEL_ID}...")
config = requests.get(f"{BASE_URL}/config.json").json()
tokenizer_json = requests.get(f"{BASE_URL}/tokenizer.json").json()

vocab = tokenizer_json.get("model", {}).get("vocab", {})

def remap_tensor_name(name):
    if name == "model.embed_tokens.weight":
        return "tok_emb"
    if name == "model.norm.weight":
        return "final_norm"
    if name == "lm_head.weight":
        return "output"
    
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
        
        mapping = {
            "input_layernorm.weight": "attn_norm",
            "post_attention_layernorm.weight": "ffn_norm",
            "self_attn.q_proj.weight": "wq",
            "self_attn.k_proj.weight": "wk",
            "self_attn.v_proj.weight": "wv",
            "self_attn.o_proj.weight": "wo",
            "mlp.gate_proj.weight": "w1",
            "mlp.down_proj.weight": "w2",
            "mlp.up_proj.weight": "w3"
        }
        
        if rest in mapping:
            return f"layers.{layer_idx}.{mapping[rest]}"
            
    return name

safetensor_path = "model.safetensors"

print("--> Converting & remapping tensors...")
tensors = []
binary_data = bytearray()

with safe_open(safetensor_path, framework="np", device="cpu") as f:
    for key in f.keys():
        new_name = remap_tensor_name(key)
        tensor = f.get_tensor(key)
        data = tensor.tobytes()
        
        # บังคับ offset ให้อยู่บนจุดเริ่มต้นของข้อมูลชุดใหม่เสมอ
        offset = len(binary_data)
        binary_data.extend(data)
        
        tensors.append({
            "name": new_name,
            "dtype": "f32",
            "shape": list(tensor.shape),
            "offset": offset
        })

header = {
    "type": "byte_bpe",
    "vocab": vocab,
    "merges": tokenizer_json.get("model", {}).get("merges", []),
    "config": {
        "n_layers": config.get("num_hidden_layers"),
        "n_heads": config.get("num_attention_heads"),
        "n_kv_heads": config.get("num_key_value_heads", config.get("num_attention_heads")),
        "d_model": config.get("hidden_size"),
        "d_ff": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
        "max_seq_len": config.get("max_position_embeddings", 2048),
        "norm_eps": config.get("rms_norm_eps", 1e-5),
        "rope_theta": config.get("rope_theta", 10000.0)
    },
    "dtype": "f32",
    "tensors": tensors,
    "tensor_data_offset": 0
}

dummy_json = json.dumps(header).encode('utf-8')
header["tensor_data_offset"] = len(dummy_json) + 1
json_bytes = json.dumps(header).encode('utf-8')

print(f"--> Writing to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    f.write(json_bytes)
    f.write(b'\x00')
    f.write(binary_data)

print("Done! Conversion complete.")
