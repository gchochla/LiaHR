python scripts/prompting/llm_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct \
    --shot 5 15 25 --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-llm-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ opposite

python scripts/prompting/llm_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct \
    --shot 5 15 25 --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-llm-uniform-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ opposite --sampling uniform

python scripts/prompting/llm_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct --instruction ./configs/QueerReclaimLex/instruction-binary.txt \
    --shot 5 15 25 --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-llm-bin-{type}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ opposite --type out in

python scripts/prompting/llm_prompting_clsf.py \
    QueerReclaimLex --gridparse-config ./configs/QueerReclaimLex/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct --instruction ./configs/QueerReclaimLex/instruction-binary.txt \
    --shot 5 15 25 --annotation-mode aggregate --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-llm-uniform-bin-{type}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 \
    --query-label-mode _None_ opposite --type out in --sampling uniform