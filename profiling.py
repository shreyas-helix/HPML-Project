# add import
from kv_quant_wrapper import KVQuantConfig, wrap_llama_kv_quant

# in get_args()
ap.add_argument("--kv_bits", type=int, default=0, help="0 = no quant, 8 or 4 for KV quant")

# in main(), after loading model:
kv_bits = args.kv_bits
if kv_bits > 0:
    kv_cfg = KVQuantConfig(num_bits=kv_bits, enabled=True)
    model = wrap_llama_kv_quant(model, kv_cfg)
else:
    kv_cfg = KVQuantConfig(enabled=False)

# in row dict:
row["kv_bits"] = kv_bits
