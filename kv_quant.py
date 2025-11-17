import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class QuantizedTensor:
    data: torch.Tensor  # int8 or uint8
    scale: torch.Tensor  # shape broadcastable to data

def quantize_per_tensor(x: torch.Tensor, num_bits: int = 8) -> QuantizedTensor:
    # Symmetric uniform quantization
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    max_val = x.abs().max(dim=-1, keepdim=True).values  # last-dim scaling
    max_val = torch.clamp(max_val, min=1e-8)
    scale = max_val / qmax

    q = torch.round(x / scale).clamp(qmin, qmax).to(torch.int8)
    return QuantizedTensor(data=q, scale=scale)

def dequantize(quant: QuantizedTensor) -> torch.Tensor:
    return quant.data.float() * quant.scale

def quantize_kv_cache(k: torch.Tensor, v: torch.Tensor, num_bits: int = 8):
    qk = quantize_per_tensor(k, num_bits=num_bits)
    qv = quantize_per_tensor(v, num_bits=num_bits)
    return qk, qv

def dequantize_kv_cache(qk: QuantizedTensor, qv: QuantizedTensor):
    return dequantize(qk), dequantize(qv)
