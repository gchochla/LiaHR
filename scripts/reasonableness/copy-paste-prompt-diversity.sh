python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --test-split dev test \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 --annotation-mode both \
    --alternative {model_name_or_path}-query-prompt-diversity-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --load-in-4bit \
    --train-keep-same-examples --test-keep-same-examples --train-debug-ann 3

python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml --test-split dev test \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 --annotation-mode both \
    --alternative {model_name_or_path}-query-prompt-diversity-{shot}-shot --seed 0 1 2 \
    --include-query-in-demos true --test-debug-len 100 --load-in-4bit \
    --train-keep-same-examples --test-keep-same-examples --train-debug-ann 3
