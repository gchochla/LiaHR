# dataset_dir=

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --root-dir $dataset_dir/goemotions --emotion-clustering-json $dataset_dir/goemotions/emotion_clustering.json \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --max-new-tokens 18 --accelerate --shot 5 --label-format json \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --alternative {model_name_or_path}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --root-dir $dataset_dir/goemotions --emotion-clustering-json $dataset_dir/goemotions/emotion_clustering.json \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --model-name $model --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --root-dir $dataset_dir/mfrc \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --max-new-tokens 18 --accelerate --shot 5 --label-format json \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --alternative {model_name_or_path}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    MFRC --root-dir $dataset_dir/mfrc \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}.\n' --incontext $'Input: {text}\nMoral foundation(s): {label}\n' \
    --model-name $model --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

