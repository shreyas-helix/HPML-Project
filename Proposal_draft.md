# **Project Proposal (Final Draft)**

## **1. Project Title**
**Optimizing Large Language Model Inference through Key-Value (KV) Cache Quantization and Efficient Attention Mechanisms**

---

## **2. Team Members (2)**
- **Student A:** Shreyas KV [sk12200@nyu.edu] 
- **Student B:** Pavan Kishore [pr2622@nyu.edu]

---

## **3. Goal / Objective**

The goal of this project is to design and evaluate optimization strategies that improve **inference efficiency** of Large Language Models (LLMs) by reducing GPU memory footprint and latency during generation.

We aim to demonstrate that:

- Quantizing the **KV cache** to lower precisions (INT8, 4-bit) can substantially reduce memory consumption while maintaining output quality.
- Integrating **FlashAttention-2** enables faster, memory-efficient attention computations.
- Combining **hybrid precision** and **selective quantization** strategies can achieve better trade-offs between accuracy and performance.
- Comprehensive profiling can quantify the relationship between throughput, memory savings, and model quality.

Our hypothesis: combining **KV cache quantization** and **optimized attention kernels** can reduce memory usage by **50–70%**, with <5% quality degradation.

---

## **4. Challenges**

- **Memory Bottleneck:** KV cache grows linearly with sequence length and batch size, saturating GPU memory.  
- **Precision Trade-offs:** Lower precision may increase perplexity or degrade task-specific accuracy.  
- **Kernel Compatibility:** Ensuring FlashAttention-2 operates seamlessly with quantized tensors.  
- **Profiling Consistency:** Measuring VRAM usage, latency, and throughput uniformly across variants.

---

## **5. Approach / Techniques**

### **(a) Model and Dataset**

**Model:**
- **Llama 3.1 8B** (or **Llama 2 7B** as fallback)
  - Open-source, well-documented, widely used in inference efficiency research.
  - Compatible with PyTorch’s FlashAttention-2 and quantization toolkits.

**Datasets:**

| Dataset | Task | Metric | Purpose |
|----------|------|--------|----------|
| **Wikitext-2** | Language modeling | Perplexity | Baseline evaluation |
| **LongBench** | Long-context QA | Accuracy | Scaling and memory analysis |
| **SQuAD 2.0** | Question answering | F1 / EM | Functional quality check |
| *(Optional)* **HumanEval** | Code generation | pass@1 | Generative stability test |

---

### **(b) Optimization Techniques**

- **Baseline:** Standard FP16 inference using Hugging Face Transformers.  
- **FlashAttention-2:** Replace default attention kernels with IO-aware fused kernels.  
- **Quantized KV Cache:**  
  - Apply **post-training quantization (PTQ)** using `torch.ao.quantization` and `bitsandbytes`.  
  - Compare **INT8**, **4-bit**, and **mixed precision (Hybrid)** approaches.  
- **Hybrid Layer Strategy:** Keep early/late layers in higher precision; quantize middle transformer blocks.  
- **Profiling:** Use `torch.profiler`, Nsight Systems, and Roofline modeling for kernel-level efficiency analysis.

**Evaluation Configurations:**

| Configuration | Precision | Technique | Expected Memory Reduction |
|----------------|------------|------------|----------------------------|
| Baseline | FP16 | Standard attention | 0% |
| FlashAttention-2 | FP16 | Fused attention kernel | ~50% |
| INT8-KV | INT8 | Quantized KV cache | ~30% |
| 4-bit-KV | 4-bit | Aggressive quantization | ~70% |
| Hybrid | Mixed | Layer-wise selective quantization | ~25% |

---

## **6. Implementation Details**

### **Hardware**

| Component | Description |
|------------|-------------|
| **Compute** | NVIDIA A100 (40GB) via NYU HPC (`ece_gy_9143` partition) |
| **Precision** | FP16 / INT8 / 4-bit (mixed) |
| **Profiling Tools** | `torch.profiler`, Nsight Systems |
| **Deployment** | Single-node inference on PyTorch 2.x + CUDA 12.x |

---

### **Software / Framework**

| Category | Library |
|-----------|----------|
| **Deep Learning** | PyTorch ≥ 2.0, xFormers, bitsandbytes |
| **Quantization** | torch.ao.quantization |
| **Profiling** | torch.profiler, Nsight Systems |
| **Visualization** | matplotlib, seaborn |
| **Experiment Management** | wandb (optional), CSV logging |

---

### **Baseline vs Optimized Models**

| Variant | Attention | Precision | Optimization |
|----------|------------|------------|---------------|
| Baseline | Standard | FP16 | None |
| Optimized | FlashAttention-2 | FP16 / INT8 | Quantized KV + Hybrid Precision |

---

### **Evaluation Metrics**

| Category | Metric | Description |
|-----------|---------|-------------|
| **Performance** | Throughput (tokens/s), TTFT, latency | Speed and efficiency |
| **Memory** | Peak VRAM, bytes/token | Memory utilization |
| **Quality** | Perplexity, F1 / EM | Model accuracy |
| **Efficiency** | Compression ratio, SM utilization | GPU performance |

---

## **7. Demo Planned**

The demo will include:

- **Live Inference Walkthrough:** showing all configurations (FP16 → INT8 → Hybrid).  
- **Profiling Visualizations:** timeline and Roofline plots from `torch.profiler`.  
- **Comparative Charts:**
  - *Memory vs Sequence Length* (512 → 8192 tokens)  
  - *Throughput vs Batch Size*  
  - *Perplexity vs Precision*
- **Performance Highlights:**
  - Example: “INT8-KV achieves 55% memory reduction and 1.5× throughput gain with only +3% perplexity increase.”

**Deliverables:**
- Jupyter notebook demonstration  
- Comparative graphs and tables  
- 15–20 minute final presentation summarizing results  

---

## **8. References**

1. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 35.  
2. Luo, Y. et al. (2024). *INT-FlashAttention: Enabling FlashAttention for INT8 Quantization.* arXiv:2409.16997.  
3. Liu, Z. et al. (2025). *MiniKV: 2-bit Layer-Discriminative KV Cache Compression.* arXiv:2501.xxxxx.  
4. Chen, X. et al. (2024). *TurboAttention: Efficient Attention Approximation for High-Throughput LLMs.* NeurIPS 2024.  
5. Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.  
6. Kwon, W. et al. (2023). *PagedAttention: Efficient Memory Management for LLM Serving.* SOSP 2023.

---

## **Summary**

This project investigates practical, course-level optimizations for large language model inference.  
By combining **quantized KV cache** techniques with **FlashAttention** kernels, it aims to:

- Reduce GPU memory usage by up to **70%**  
- Maintain model quality within **<5% degradation**  
- Deliver reproducible **profiling-based efficiency benchmarks**

