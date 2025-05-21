# dataset_dir = 

python scripts/prompting/api_pair_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1/ --language English \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided emotion label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}\n\n' --incontext $'Input: {text}\nLabel: {label}\n{r}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 15 25 55 75 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_pair_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1/ --language English \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided emotion label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}\n\n' --incontext $'Input: {text}\nLabel: {label}\nReasonable? {r}\n\n' \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 15 25 55 75 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ --yn-labels True

python scripts/prompting/api_pair_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml \
    --instruction ./configs/MFRC/reasonableness/instruction.txt --incontext ./configs/MFRC/reasonableness/incontext.txt --system ./configs/MFRC/reasonableness/system.txt \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/api_pair_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml \
    --instruction ./configs/GoEmotions/reasonableness/instruction.txt --incontext ./configs/GoEmotions/reasonableness/incontext.txt --system ./configs/GoEmotions/reasonableness/system.txt \
    --model-name gpt-4o-mini-2024-07-18 --max-new-tokens 5 --shot 5 15 25 55 75 \
    --annotation-mode aggregate \
    --alternative {model_name}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution