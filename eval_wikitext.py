import argparse
import math
import time
import torch
from datasets import load_dataset
from tqdm import tqdm

from model_loader import load_llama
from utils import compute_perplexity, measure_memory_mb, reset_cuda_stats, append_to_csv

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--use_flash_attn2", action="store_true")
    ap.add_argument("--max_eval_tokens", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--results_path", type=str, default="results/metrics.csv")
    ap.add_argument("--config_name", type=str, default="baseline_fp16")
    return ap.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_llama(
        model_name=args.model_name,
        dtype=args.dtype,
        use_flash_attn2=args.use_flash_attn2,
        device_map="auto",
    )

    # Load Wikitext-2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text and tokenize
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Truncate to max_eval_tokens
    input_ids = input_ids[:, : args.max_eval_tokens]

    n_tokens = input_ids.size(1)
    seq_len = args.seq_len

    # Sliding window evaluation with causal loss
    nll_sum = 0.0
    count = 0

    reset_cuda_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in tqdm(range(0, n_tokens - 1, seq_len)):
            # context and labels shifted by 1
            ctx = input_ids[:, i : i + seq_len]
            labels = ctx.clone()
            outputs = model(input_ids=ctx, labels=labels)
            # HF returns loss averaged over tokens; convert to nll * token_count
            loss = outputs.loss.item()
            token_count = labels.numel()
            nll_sum += loss * token_count
            count += token_count

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_mem_mb = measure_memory_mb()

    ppl = compute_perplexity(-nll_sum / count * -1, count)  # same as exp(loss_avg)

    tokens_per_sec = count / total_time

    row = {
        "config": args.config_name,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "use_flash_attn2": int(args.use_flash_attn2),
        "seq_len": seq_len,
        "n_tokens": int(count),
        "total_time_s": total_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_mb": peak_mem_mb,
        "perplexity": ppl,
    }
    append_to_csv(row, args.results_path)

    print(row)

if __name__ == "__main__":
    main()
