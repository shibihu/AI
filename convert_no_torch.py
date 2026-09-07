import json
import requests
from safetensors import safe_open

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
OUTPUT_FILE = "smollm_135m.slm"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/raw/main"

print(f"--> Downloading config files for {MODEL_ID}...")
config = requests.get(f"{BASE_URL}/config.json").json()
tokenizer_json = requests.get(f"{BASE_URL}/tokenizer.json").json()

tokenizer_model = tokenizer_json.get("model", {})
vocab = {
    str(token): int(token_id)
    for token, token_id in tokenizer_model.get("vocab", {}).items()
}

# Some Hugging Face tokenizers keep special tokens in added_tokens instead of
# model.vocab. Include them in the serialized vocabulary as well.
special_tokens = {}
for added in tokenizer_json.get("added_tokens", []):
    content = added.get("content")
    token_id = added.get("id")
    if content is None or token_id is None:
        continue
    content = str(content)
    token_id = int(token_id)
    vocab.setdefault(content, token_id)
    special_tokens[content] = token_id


def bytes_to_unicode():
    """Return GPT-2's reversible byte-to-Unicode mapping."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]
    extra = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra)
            extra += 1
    return {str(byte): chr(codepoint)
            for byte, codepoint in zip(byte_values, unicode_values)}


model_unk_token = tokenizer_model.get("unk_token")
unk_token = model_unk_token or "<unk>"
unk_id = vocab.get(unk_token)
if unk_id is None:
    unk_id = special_tokens.get("<unk>")
if unk_id is None:
    # Keep the metadata explicit even for tokenizers without a declared unk.
    unk_token = "<unk>"
    unk_id = 0
    vocab.setdefault(unk_token, unk_id)

bos_token = tokenizer_model.get("bos_token")
if not bos_token:
    bos_token = next((candidate for candidate in
                      ("<s>", "<bos>", "<|endoftext|>") if candidate in vocab), "<s>")
eos_token = tokenizer_model.get("eos_token")
if not eos_token:
    eos_token = next((candidate for candidate in
                      ("</s>", "<eos>", "<|endoftext|>") if candidate in vocab), "</s>")
bos_id = vocab.get(bos_token, special_tokens.get(bos_token, vocab.get("<bos>", 0)))
eos_id = vocab.get(eos_token, special_tokens.get(eos_token, vocab.get("<eos>", 1)))

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
    "merges": tokenizer_model.get("merges", []),
    "byte_encoder": bytes_to_unicode(),
    "special_tokens": special_tokens,
    "unk_token": unk_token,
    "unk_id": int(unk_id),
    "bos_token": bos_token,
    "bos_id": int(bos_id),
    "eos_token": eos_token,
    "eos_id": int(eos_id),
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

def serialize_header(metadata):
    """Return UTF-8 JSON padded so the tensor payload starts on a 4-byte boundary."""
    metadata["tensor_data_offset"] = 0

    for _ in range(16):
        raw_header = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        padding = (4 - (len(raw_header) % 4)) % 4
        header_bytes = raw_header + (b" " * padding)
        tensor_data_offset = len(header_bytes)

        if metadata["tensor_data_offset"] == tensor_data_offset:
            break
        metadata["tensor_data_offset"] = tensor_data_offset
    else:
        raise RuntimeError("Could not stabilize aligned header length")

    # Serialize once more using the final absolute tensor offset. This keeps
    # the JSON content and the tensor payload offset synchronized.
    raw_header = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    padding = (4 - (len(raw_header) % 4)) % 4
    header_bytes = raw_header + (b" " * padding)

    if len(header_bytes) % 4 != 0:
        raise AssertionError("JSON header is not 4-byte aligned")
    if metadata["tensor_data_offset"] != len(header_bytes):
        raise RuntimeError("Header length changed after final serialization")
    if not header_bytes.startswith(b"{"):
        raise RuntimeError("Serialized header is not a JSON object")
    header_bytes.decode("utf-8")
    return header_bytes


header_bytes = serialize_header(header)

print(f"--> Writing to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    f.write(header_bytes)
    f.write(binary_data)

print("Done! Conversion complete.")
