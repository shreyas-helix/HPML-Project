import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    dtype: str = "fp16",
    use_flash_attn2: bool = False,
    device_map: str = "auto",
):
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    attn_impl = "flash_attention_2" if use_flash_attn2 else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,  # FA2 integration per HF docs :contentReference[oaicite:1]{index=1}
    )

    model.eval()
    return model, tokenizer
