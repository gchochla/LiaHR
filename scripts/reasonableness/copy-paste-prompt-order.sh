# dataset_dir=

# semeval
python scripts/prompting/llm_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1 \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --model-name-or-path meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Llama-2-70b-chat-hf --max-new-tokens 18 --accelerate --shot 15 --label-format json \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --alternative {model_name_or_path}-query-prompt-{query_label_mode}-{shot}-shot-{query_order}-order \
    --include-query-in-demos true --query-order 3 6 9 12 15 --query-label-mode _None_ distribution --seed 0 1 2

python scripts/prompting/api_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1 \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot-{query_order}-order \
    --include-query-in-demos true --query-order 3 6 9 12 15 --query-label-mode _None_ distribution --seed 0 1 2
