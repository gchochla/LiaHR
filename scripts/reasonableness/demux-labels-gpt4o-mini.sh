# filtered labels
python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split dev --test-split train \
    --incontext $'Input: {text}\nEmotion(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-full-train-predictions --seed 2 \
    --include-query-in-demos true --query-label-mode _None_

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split dev --test-split train \
    --incontext $'Input: {text}\nEmotions(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-full-train-predictions --seed 0 \
    --include-query-in-demos true --query-label-mode _None_

# actual labels
python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split dev --test-split train \
    --incontext $'Input: {text}\nEmotion(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-full-train-predictions --seed 2

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split dev --test-split train \
    --incontext $'Input: {text}\nEmotions(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-full-train-predictions --seed 0

# filtered labels for dev and test
python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --test-split dev test \
    --incontext $'Input: {text}\nEmotion(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-full-eval-predictions --seed 2 \
    --include-query-in-demos true --query-label-mode _None_

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --test-split dev test \
    --incontext $'Input: {text}\nEmotions(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-query-prompt-full-eval-predictions --seed 0 \
    --include-query-in-demos true --query-label-mode _None_

# actual labels for dev and test
python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --test-split dev test \
    --incontext $'Input: {text}\nEmotion(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 5 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-full-eval-predictions --seed 2

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --test-split dev test \
    --incontext $'Input: {text}\nEmotions(s): {label}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 18 --shot 15 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-full-eval-predictions --seed 0

# filter labels for train dev test with baseline

python scripts/prompting/api_pair_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --instruction ./configs/GoEmotions/reasonableness/instruction.txt --incontext ./configs/GoEmotions/reasonableness/incontext.txt --system ./configs/GoEmotions/reasonableness/system.txt \
    --train-split train --test-split dev test \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 \
    --annotation-mode aggregate \
    --alternative {model_name}-baseline-full-eval-predictions --seed 0

python scripts/prompting/api_pair_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --instruction ./configs/GoEmotions/reasonableness/instruction.txt --incontext ./configs/GoEmotions/reasonableness/incontext.txt --system ./configs/GoEmotions/reasonableness/system.txt \
    --train-split dev --test-split train \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 \
    --annotation-mode aggregate \
    --alternative {model_name}-baseline-full-train-predictions --seed 0

python scripts/prompting/api_pair_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --instruction ./configs/GoEmotions/reasonableness/instruction.txt --incontext ./configs/GoEmotions/reasonableness/incontext.txt --system ./configs/GoEmotions/reasonableness/system.txt \
    --train-split train --test-split dev test \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 15 \
    --annotation-mode aggregate \
    --alternative {model_name}-baseline-full-eval-predictions --seed 0

python scripts/prompting/api_pair_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --instruction ./configs/GoEmotions/reasonableness/instruction.txt --incontext ./configs/GoEmotions/reasonableness/incontext.txt --system ./configs/GoEmotions/reasonableness/system.txt \
    --train-split dev --test-split train \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 15 \
    --annotation-mode aggregate \
    --alternative {model_name}-baseline-full-train-predictions --seed 0