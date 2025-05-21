# dataset_dir=

python scripts/prompting/llm_pair_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1/ --language English \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided emotion label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}.\n\n' --incontext $'Input: {text}\nLabel: {label}\n{r}\n\n' \
    --model-name meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 5 --shot 5 15 25 55 75 \
    --cache-dir /data2/foundation_models_checkpoints \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --accelerate --load-in-4bit \
    --alternative {model_name_or_path}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/llm_pair_prompting_clsf.py \
    SemEval --root-dir $dataset_dir/semeval2018task1/ --language English \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided emotion label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}.\n\n' --incontext $'Input: {text}\nLabel: {label}\nReasonable? {r}\n\n' \
    --model-name meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 5 --shot 5 15 25 55 75 \
    --cache-dir /data2/foundation_models_checkpoints \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --accelerate --load-in-4bit \
    --alternative {model_name_or_path}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ --yn-labels True

python scripts/prompting/llm_pair_prompting_clsf.py \
    MFRC --root-dir $dataset_dir/mfrc \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided moral label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}.\n\n' --incontext $'Input: {text}\nLabel: {label}\n{r}\n\n' \
    --model-name meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 5 --shot 5 15 25 55 75 \
    --cache-dir /data2/foundation_models_checkpoints \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --accelerate --load-in-4bit \
    --alternative {model_name_or_path}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution

python scripts/prompting/llm_pair_prompting_clsf.py \
    GoEmotions --root-dir $dataset_dir/goemotions --emotion-clustering-json $dataset_dir/goemotions/emotion_clustering.json \
    --train-split train --test-split dev \
    --system ' ' --instruction $'Assess the reasonableness of the provided emotion label for each input. Namely, evaluate whether the label makes sense for its corresponding input, under some reasonable interpretation. Reply only with {labels}.\n\n' --incontext $'Input: {text}\nLabel: {label}\n{r}\n\n' \
    --model-name meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --max-new-tokens 5 --shot 5 15 25 55 75 \
    --cache-dir /data2/foundation_models_checkpoints \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --accelerate --load-in-4bit \
    --alternative {model_name_or_path}-{yn_labels}-{query_label_mode}-{shot}-shot --seed 0 1 2 \
    --test-debug-len 100 --query-label-mode _None_ distribution


CUDA_VISIBLE_DEVICES=3 python scripts/prompting/vllm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct --label-format json \
    --seed 0 --shot 5 --annotation-mode aggregate --quantization fp8 \
    --alternative aaaa-{model_name_or_path}/test --test-debug-len 10