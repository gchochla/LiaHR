python scripts/prompting/api_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name gpt-4o-mini-2024-07-18 \
    --shot 5 15 25 55 75 --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ distribution

# python scripts/prompting/vllm_prompting_clsf.py \
#     QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
#     --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct \
#     --shot 5 15 25 --annotation-mode aggregate --quantization fp8 \
#     --alternative {model_name_or_path}-query-prompt-uniform-{query_label_mode}-{shot}-shot --seed 0 1 2 \
#     --include-query-in-demos true --test-debug-len 100 \
#     --query-label-mode _None_ distribution --sampling uniform

python scripts/prompting/api_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name gpt-4o-mini-2024-07-18 --instruction ./configs/QueerReclaimLex/instruction-binary.txt \
    --shot 5 15 25 55 75 --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-bin-{type}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ distribution --type out in

# python scripts/prompting/vllm_prompting_clsf.py \
#     QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
#     --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct --instruction ./configs/QueerReclaimLex/instruction-binary.txt \
#     --shot 5 15 25 --annotation-mode aggregate --quantization fp8 \
#     --alternative {model_name_or_path}-query-prompt-uniform-bin-{type}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
#     --include-query-in-demos true --test-debug-len 100 \
#     --query-label-mode _None_ distribution --type out in --sampling uniform