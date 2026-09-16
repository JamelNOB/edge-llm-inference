#!/usr/bin/env bash
# ==============================================================================
# Model Quantization Pipeline Script (FP16 -> GGUF F16 -> GGUF Q4_K_M)
# [简历指标验证] 将 3.09 GB 原生权重极限压缩至 934.69 MiB (70.3% 体积缩减)
# ==============================================================================

set -e

LLAMA_CPP_DIR="${HOME}/llama.cpp"
INPUT_MERGED_DIR="${HOME}/llm_practice/models/qwen_lora_merged"
OUTPUT_F16_GGUF="${HOME}/llm_practice/models/qwen_lora_merged.f16.gguf"
OUTPUT_Q4_GGUF="${HOME}/llm_practice/models/qwen_lora_merged.Q4_K_M.gguf"

QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

echo "=== 步骤 1: 将合并后的 HuggingFace 权重转换为 GGUF F16 格式 ==="
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
  "${INPUT_MERGED_DIR}" \
  --outfile "${OUTPUT_F16_GGUF}" \
  --outtype f16

echo "=== 步骤 2: 执行 llama-quantize 进行 INT4 (Q4_K_M) 混合精度量化 ==="
"${QUANTIZE_BIN}" "${OUTPUT_F16_GGUF}" "${OUTPUT_Q4_GGUF}" Q4_K_M

echo "=== 步骤 3: 检查量化结果与体积对比 ==="
ls -lh "${OUTPUT_F16_GGUF}" "${OUTPUT_Q4_GGUF}"
echo "✅ 模型量化已成功完成！产物位于: ${OUTPUT_Q4_GGUF}"
