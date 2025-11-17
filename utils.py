import time
import torch
import pandas as pd
from pathlib import Path

def compute_perplexity(log_likelihood_sum: float, token_count: int) -> float:
    import math
    return math.exp(-log_likelihood_sum / token_count)

def measure_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_cuda_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def append_to_csv(row: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)

def time_block():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return end - start
