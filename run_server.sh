export http_proxy=xx
export https_proxy=xx
#python3.11 -m vllm.entrypoints.openai.api_server --model=Qwen/Qwen3-1.7B --enable-reasoning --reasoning-parser deepseek_r1
python3.11 -m vllm.entrypoints.openai.api_server --model=Qwen/Qwen3-1.7B --reasoning-parser deepseek_r1


export http_proxy=
export https_proxy=


