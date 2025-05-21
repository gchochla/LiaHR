#!/bin/bash

# train with original dataset

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --alternative normal-{intra_loss_coef}

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --alternative normal-{intra_loss_coef}

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --alternative normal-{intra_loss_coef}

# train with corrected labels

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative cp-gpt4o-{intra_loss_coef}

# TODO: same for llama 3

# python scripts/training/demux.py MFRC \
#     --gridparse-config ./configs/MFRC/config.yaml \
#     --train-split train --dev-split dev --test-split dev test \
#     --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
#     --model-name bert-base-uncased --text-preprocessor true --seed 0 \
#     --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
#     --intra-loss-coef 0 0.2 --seed 0 1 2 \
#     --train-pred-log-dir ./logs/MFRCOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
#     --train-pred-log-index 0 --alternative cp-gpt4o-{intra_loss_coef}

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative cp-gpt4o-{intra_loss_coef}



# train with only labels that pass filter

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --filter-predictions \
    --train-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative cp-filter-gpt4o-{intra_loss_coef}

# TODO: same for llama 3

# python scripts/training/demux.py MFRC \
#     --gridparse-config ./configs/MFRC/config.yaml \
#     --train-split train --dev-split dev --test-split dev test \
#     --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
#     --model-name bert-base-uncased --text-preprocessor true --seed 0 \
#     --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
#     --intra-loss-coef 0 0.2 --seed 0 1 2 --filter-predictions \
#     --train-pred-log-dir ./logs/MFRCOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
#     --train-pred-log-index 0 --alternative cp-filter-gpt4o-{intra_loss_coef}

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --filter-predictions \
    --train-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative cp-filter-gpt4o-{intra_loss_coef}

# train with model predictions

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative gpt4o-{intra_loss_coef}

# TODO: same for llama 3

# python scripts/training/demux.py MFRC \
#     --gridparse-config ./configs/MFRC/config.yaml \
#     --train-split train --dev-split dev --test-split dev test \
#     --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
#     --model-name bert-base-uncased --text-preprocessor true --seed 0 \
#     --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
#     --intra-loss-coef 0 0.2 --seed 0 1 2 \
#     --train-pred-log-dir ./logs/MFRCOpenAI/gpt-4o-mini-2024-07-18-full-train-predictions_0 \
#     --train-pred-log-index 0 --alternative gpt4o-{intra_loss_coef}

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative gpt4o-{intra_loss_coef}

# train with filtered labels and test with filtered labels

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 \
    --dev-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-eval-predictions_0 \
    --dev-pred-log-index 0 \
    --test-pred-log-dir ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-eval-predictions_0 \
    --test-pred-log-index 0 \
    --alternative cp-gpt4o-{intra_loss_coef}

# TODO: same for llama 3

# python scripts/training/demux.py MFRC \
#     --gridparse-config ./configs/MFRC/config.yaml \
#     --train-split train --dev-split dev --test-split dev test \
#     --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
#     --model-name bert-base-uncased --text-preprocessor true --seed 0 \
#     --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
#     --intra-loss-coef 0 0.2 --seed 0 1 2 \
#     --train-pred-log-dir ./logs/MFRCOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
#     --train-pred-log-index 0 --alternative cp-gpt4o-{intra_loss_coef}

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 \
    --train-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-train-predictions_0 \
    --train-pred-log-index 0 \
    --dev-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-eval-predictions_0 \
    --dev-pred-log-index 0 \
    --test-pred-log-dir ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-full-eval-predictions_0 \
    --test-pred-log-index 0 \
    --alternative cp-gpt4o-{intra_loss_coef}

# train with filtered labels with baseline

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --filter-predictions \
    --train-pred-log-dir ./logs/GoEmotionsOpenAIReason/gpt-4o-mini-2024-07-18-baseline-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative bsl-filter-gpt4o-{intra_loss_coef}


python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split dev test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 5 \
    --intra-loss-coef 0 0.2 --seed 0 1 2 --filter-predictions \
    --train-pred-log-dir ./logs/SemEvalOpenAIReason/gpt-4o-mini-2024-07-18-baseline-full-train-predictions_0 \
    --train-pred-log-index 0 --alternative bsl-filter-gpt4o-{intra_loss_coef}