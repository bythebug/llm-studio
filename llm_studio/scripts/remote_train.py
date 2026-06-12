#!/usr/bin/env python3
"""
Self-contained training script — runs on the remote compute instance.

Usage:
    python3 remote_train.py --data data.json --config config.json --output ./artifacts
"""
import argparse
import json
import math
import os
import random
import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


def split_data(pairs, train_ratio=0.8, seed=42):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    cut = int(n * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def run_training(data: list[dict], config: dict, output_dir: str) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install with: pip3 install torch transformers")
        sys.exit(1)

    model_name = config.get("model_name", "gpt2")
    epochs     = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 4))
    lr         = float(config.get("learning_rate", 2e-5))
    max_len    = int(config.get("max_seq_length", 128))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    log(f"[LLM Studio] Device: {device}")
    log(f"[LLM Studio] Model: {model_name} | epochs={epochs} | lr={lr} | batch={batch_size}")

    pairs = [(d["input"], d["output"]) for d in data]
    train_pairs, val_pairs = split_data(pairs)
    log(f"[LLM Studio] Data — train: {len(train_pairs)}, val: {len(val_pairs)}")

    log("[LLM Studio] Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def encode_batch(batch):
        texts = [inp + tokenizer.eos_token + out for inp, out in batch]
        enc = tokenizer(texts, return_tensors="pt", truncation=True,
                        max_length=max_len, padding=True)
        return {k: v.to(device) for k, v in enc.items()}

    def compute_loss(enc):
        import torch.nn.functional as F
        out = model(**enc)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc["input_ids"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        random.shuffle(train_pairs)
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i: i + batch_size]
            if not batch:
                continue
            enc = encode_batch(batch)
            loss = compute_loss(enc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1

        train_loss = total_loss / max(steps, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch = val_pairs[i: i + batch_size]
                if not batch:
                    continue
                enc = encode_batch(batch)
                val_loss += compute_loss(enc).item()
                val_steps += 1
        val_loss = val_loss / max(val_steps, 1)

        log(f"[LLM Studio] Epoch {epoch}/{epochs} — "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"ppl={math.exp(min(val_loss, 20)):.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            import json as _json
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                _json.dump({
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }, f, indent=2)
            log(f"[LLM Studio] ✓ Best model saved (epoch {epoch}, val_loss={val_loss:.4f})")

    log(f"[LLM Studio] Training complete. Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    log(f"[LLM Studio] ARTIFACT_PATH:{output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to training data JSON")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--output", required=True, help="Local output directory for artifacts")
    args = parser.parse_args()

    with open(args.data)   as f: data   = json.load(f)
    with open(args.config) as f: config = json.load(f)

    log(f"[LLM Studio] Remote training starting — {len(data)} samples")
    run_training(data, config, args.output)


if __name__ == "__main__":
    main()
