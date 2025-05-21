# filtered labels
# TODO: see if uniform is better for MFRC
python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-full-train-predictions --seed 0 \
    --include-query-in-demos true --query-label-mode _None_

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-full-train-predictions --seed 2 \
    --include-query-in-demos true --query-label-mode _None_

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-query-prompt-full-train-predictions --seed 0 \
    --include-query-in-demos true --query-label-mode _None_

# actual labels
python scripts/prompting/llm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-full-train-predictions --seed 0

python scripts/prompting/llm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-full-train-predictions --seed 2

python scripts/prompting/llm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split dev --test-split train \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --load-in-4bit \
    --alternative {model_name_or_path}-full-train-predictions --seed 0
