#!/bin/bash
# VLLM Server Startup Script for recsys-universe
# Usage: ./run_server.sh

# Note: Configure your proxy settings in environment or shell profile
# export http_proxy=your_proxy
# export https_proxy=your_proxy

# Start VLLM server with Qwen3-1.7B model
python3.11 -m vllm.entrypoints.openai.api_server \
    --model=Qwen/Qwen3-1.7B \
    --reasoning-parser deepseek_r1
