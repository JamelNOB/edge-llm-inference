#!/usr/bin/env bash
# ==============================================================================
# llama.cpp C++ Hardware Benchmark Script
# [简历指标验证] 实测 Qwen2.5-1.5B Q4_K_M (934.69 MiB) 在 CPU 多线程下的真实吞吐
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

if [ ! -f "${MODEL_PATH}" ]; then
    # 尝试读取软链接或默认训练产物路径
    MODEL_PATH="${HOME}/llm_practice/models/qwen_lora_merged.Q4_K_M.gguf"
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[!] 错误: 未找到量化模型 GGUF 文件: ${MODEL_PATH}"
    echo "    请先执行 scripts/quantize_model.sh 或下载模型权重。"
    exit 1
fi

LLAMA_BENCH_BIN="${HOME}/llama.cpp/build/bin/llama-bench"
if [ ! -f "${LLAMA_BENCH_BIN}" ]; then
    LLAMA_BENCH_BIN="$(which llama-bench 2>/dev/null || true)"
fi

if [ -z "${LLAMA_BENCH_BIN}" ] || [ ! -x "${LLAMA_BENCH_BIN}" ]; then
    echo "[!] 未检测到已编译的 llama-bench 可执行文件。"
    echo "    请在 llama.cpp 目录执行: cmake --build build --target llama-bench"
    exit 1
fi

echo "======================================================================"
echo "🚀 开始执行 llama.cpp C++ 底层硬件性能评测 (llama-bench)"
echo "   模型路径: ${MODEL_PATH}"
echo "   CPU 线程: 8 Threads"
echo "   Prompt 长度 (pp): 128 tokens"
echo "   生成长度 (tg): 64 tokens"
echo "======================================================================"

"${LLAMA_BENCH_BIN}" \
  -m "${MODEL_PATH}" \
  -p 128 \
  -n 64 \
  -t 8

echo "======================================================================"
echo "✅ 压测完成！"
