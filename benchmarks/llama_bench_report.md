# llama.cpp C++ 底层硬件性能评测实测报告 (llama-bench)

## 一、 测试环境规格
- **主机平台**: Linux 4.18.0 (CentOS 8 / Skylake AVX-512)
- **CPU 规格**: 8 核心多线程调度 (`-t 8`)
- **推理底座**: llama.cpp (编译模式: Release, Native SIMD AVX2/FMA 加速)
- **测试模型**: `qwen_lora_merged.Q4_K_M.gguf`
- **参数规模**: 1.54 B
- **模型体积**: **934.69 MiB** (对比 FP16 3.09 GB，压缩比 **70.3%**)

---

## 二、 llama-bench 实测成绩单

```text
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       8 |         pp128   |        186.69 ± 19.31 |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       8 |          tg64   |         26.56 ±  2.50 |
```

---

## 三、 指标深度剖析 (双轨面试说辞)

### 1. Prefill 阶段 (Prompt 预填充, `pp128`)
- **吞吐速度**: `186.69 tokens/s`
- **首字时延 (TTFT)**: 128 个 Token 预计算耗时仅 **0.68 秒**。
- **技术本质**: 计算密集型 (Compute-Bound)。利用多核 AVX 并发并行计算矩阵乘法，极速填充 KV Cache。

### 2. Decode 阶段 (自回归逐字生成, `tg64`)
- **吞吐速度**: `26.56 tokens/s`
- **单 Token 时延**: 仅 **37.6 毫秒/字**。
- **技术本质**: 访存受限 (Memory-Bound)。通过 Q4_K_M 混合量化将每次前向搬运权重的数据量从 3GB 压到 934MB，打破端侧内存带宽墙，速度达到人类正常阅读速度 (4~5 tokens/s) 的 **5~6 倍**！
