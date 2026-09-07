#!/usr/bin/env python3
import json
import struct
import random
import sys

def create_dummy_model(output_path="model.slm"):
    print(f"[Python] Generating dummy SLM model at '{output_path}'...")
    random.seed(42)

    # Model configuration
    config = {
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_model": 64,
        "d_ff": 128,
        "vocab_size": 256,
        "max_seq_len": 512,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0
    }

    # Simple vocabulary
    vocab = [f"token_{i}" for i in range(config["vocab_size"])]

    # Tensor definitions
    tensors_metadata = []
    binary_payloads = bytearray()

    def add_tensor(name, shape):
        nonlocal binary_payloads
        elements = 1
        for d in shape:
            elements *= d

        # Generate pseudo-random float32 weights using stdlib random
        weights = [random.gauss(0.0, 0.02) for _ in range(elements)]
        weight_bytes = struct.pack(f"<{elements}f", *weights)

        offset = len(binary_payloads)
        size_bytes = len(weight_bytes)

        binary_payloads.extend(weight_bytes)

        tensors_metadata.append({
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "offset": offset,
            "size_bytes": size_bytes
        })

    # Global Tensors
    add_tensor("tok_emb", [config["vocab_size"], config["d_model"]])
    add_tensor("final_norm", [config["d_model"]])

    # Per-layer Tensors
    for l in range(config["n_layers"]):
        # Dot notation
        add_tensor(f"blk.{l}.attn_norm", [config["d_model"]])
        add_tensor(f"blk.{l}.wq", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}.wk", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}.wv", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}.wo", [config["d_model"], config["d_model"]])

        # Underscore variants
        add_tensor(f"blk.{l}_attn_norm", [config["d_model"]])
        add_tensor(f"blk.{l}_wq", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}_wk", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}_wv", [config["d_model"], config["d_model"]])
        add_tensor(f"blk.{l}_wo", [config["d_model"], config["d_model"]])

    header_dict = {
        "config": config,
        "vocab": vocab,
        "tensors": tensors_metadata
    }

    json_bytes = json.dumps(header_dict, indent=2).encode('utf-8')
    json_len = len(json_bytes)

    # Pad header to 8-byte boundary for memory alignment
    total_header_unpadded = 8 + json_len
    padding_needed = (8 - (total_header_unpadded % 8)) % 8
    padding_bytes = b'\x00' * padding_needed

    # Write binary file
    with open(output_path, "wb") as f:
        # 8-byte uint64_t json length header (stores unpadded json length)
        f.write(struct.pack("<Q", json_len))
        f.write(json_bytes)
        f.write(padding_bytes)
        f.write(binary_payloads)

    print(f"[Python] Model generated successfully!")
    print(f"  Header size : {json_len} bytes")
    print(f"  Payload size: {len(binary_payloads)} bytes")
    print(f"  Total size  : {8 + json_len + len(binary_payloads)} bytes")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "model.slm"
    create_dummy_model(out_file)
