python scripts/prompting/api_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --incontext $'Input: {text}\nMoral foundation(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml \
    --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --annotation-mode aggregate \
    --alternative {model_name}-full-labels-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --incontext $'Input: {text}\nEmotions(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --incontext $'Input: {text}\nMoral foundation(s): {label}\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 15 25 55 75 \
    --annotation-mode aggregate --sampling uniform \
    --alternative {model_name}-query-prompt-uniform-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --query-label-mode _None_ distribution

# objective
python scripts/prompting/api_prompting_clsf.py \
    TREC --gridparse-config ./configs/TREC/config.yaml \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-query-prompt-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 300 --query-label-mode _None_ distribution