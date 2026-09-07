#!/usr/bin/env python3
"""
generate_model.py — Produce a valid dummy model.slm file for the SLM engine.

File layout:
    [JSON header][0x00 byte][tensor data (float32, little-endian)]

The JSON header describes model config, vocabulary, and tensor metadata
(name, dtype, shape, byte-offset within the tensor-data section).
"""

import argparse
import json
import struct
import sys
import numpy as np


def make_vocab(vocab_size: int) -> dict:
    """Build a minimal dummy vocabulary mapping tokens -> integer ids."""
    vocab = {}
    vocab["<pad>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3
    # Fill the rest with synthetic tokens
    for i in range(4, vocab_size):
        vocab[f"tok_{i}"] = i
    return vocab


def build_model(
    n_layers: int = 4,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    d_model: int = 256,
    d_ff: int = 1024,
    vocab_size: int = 1000,
    max_seq_len: int = 2048,
    norm_eps: float = 1e-5,
    rope_theta: float = 10000.0,
):
    """Return (config_dict, tensors_list, tensor_data_dict).

    tensors_list entries: {"name", "dtype", "shape", "offset"}
    tensor_data_dict maps name -> numpy float32 array.
    """
    config = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_model": d_model,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "norm_eps": norm_eps,
        "rope_theta": rope_theta,
    }

    rng = np.random.default_rng(seed=42)
    tensors = {}  # name -> numpy array

    # Global tensors
    tensors["tok_emb"]    = rng.standard_normal((vocab_size, d_model)).astype(np.float32)
    tensors["final_norm"] = np.ones((d_model,), dtype=np.float32)

    # Per-layer tensors
    for i in range(n_layers):
        pfx = f"layers.{i}"
        tensors[f"{pfx}.attn_norm"] = np.ones((d_model,), dtype=np.float32)
        tensors[f"{pfx}.wq"]  = rng.standard_normal((n_heads * (d_model // n_heads), d_model)).astype(np.float32)
        tensors[f"{pfx}.wk"]  = rng.standard_normal((n_kv_heads * (d_model // n_heads), d_model)).astype(np.float32)
        tensors[f"{pfx}.wv"]  = rng.standard_normal((n_kv_heads * (d_model // n_heads), d_model)).astype(np.float32)
        tensors[f"{pfx}.wo"]  = rng.standard_normal((d_model, d_model)).astype(np.float32)
        tensors[f"{pfx}.ffn_norm"] = np.ones((d_model,), dtype=np.float32)
        tensors[f"{pfx}.w1"]  = rng.standard_normal((d_ff, d_model)).astype(np.float32)
        tensors[f"{pfx}.w2"]  = rng.standard_normal((d_model, d_ff)).astype(np.float32)
        tensors[f"{pfx}.w3"]  = rng.standard_normal((d_ff, d_model)).astype(np.float32)

    # Assign byte offsets sequentially
    tensor_meta = []
    offset = 0
    for name, arr in tensors.items():
        byte_len = arr.nbytes
        tensor_meta.append({
            "name": name,
            "dtype": "f32",
            "shape": list(arr.shape),
            "offset": offset,
        })
        offset += byte_len

    return config, tensor_meta, tensors


def write_slm(path: str, config: dict, tensor_meta: list, tensors: dict,
              vocab: dict):
    """Write the .slm binary file."""
    # Build the JSON header (everything except tensor data)
    header = {
        "config": config,
        "vocab": vocab,
        "tensors": tensor_meta,
        "tensor_data_offset": 0,  # placeholder, set after JSON
    }

    json_bytes = json.dumps(header, indent=2).encode("utf-8")

    # The tensor_data_offset is the byte position AFTER json_bytes + 1 (for \0)
    actual_tensor_data_offset = len(json_bytes) + 1

    # Update tensor_data_offset in the header and re-serialize
    header["tensor_data_offset"] = actual_tensor_data_offset
    json_bytes = json.dumps(header, indent=2).encode("utf-8")
    # After re-serialisation the length may change; recalculate
    actual_tensor_data_offset = len(json_bytes) + 1
    header["tensor_data_offset"] = actual_tensor_data_offset
    json_bytes = json.dumps(header, indent=2).encode("utf-8")

    total_size = len(json_bytes) + 1  # +1 for \0 separator
    for meta in tensor_meta:
        total_size += tensors[meta["name"]].nbytes

    print(f"Writing {path}  ({total_size / 1024:.1f} KB, "
          f"json={len(json_bytes)} bytes, "
          f"tensors={len(tensor_meta)}, "
          f"tensor_data_offset={actual_tensor_data_offset})")

    with open(path, "wb") as f:
        # JSON header
        f.write(json_bytes)
        f.write(b"\x00")  # null separator

        # Tensor data — must match offsets declared in metadata
        written = 0
        for meta in tensor_meta:
            arr = tensors[meta["name"]]
            # Ensure offset matches
            assert written == meta["offset"], (
                f"Offset mismatch for {meta['name']}: "
                f"expected {meta['offset']}, got {written}"
            )
            f.write(arr.tobytes())
            written += arr.nbytes

        assert written == total_size - len(json_bytes) - 1, "Size mismatch"

    print(f"Done. Total size: {total_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dummy .slm model file for the SLM inference engine"
    )
    parser.add_argument("-o", "--output", default="model.slm",
                        help="Output .slm file path (default: model.slm)")
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    args = parser.parse_args()

    config, tensor_meta, tensors = build_model(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        norm_eps=args.norm_eps,
        rope_theta=args.rope_theta,
    )

    vocab = make_vocab(args.vocab_size)
    write_slm(args.output, config, tensor_meta, tensors, vocab)


if __name__ == "__main__":
    main()
