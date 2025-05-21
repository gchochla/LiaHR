python scripts/reasonableness/double_blind_generation.py \
    --task icl_vs_gt \
    --exp ../acii-24-logs/SemEvalOpenAI/gpt-3.5-turbo-random-retrieval-35-shot_0 \
    --n 30 --o logs/analysis/reasonableness/icl 

python scripts/reasonableness/double_blind_generation.py \
    --task query_prompt_vs_gt \
    --exp ./logs/analysis/reasonableness/SemEval/meta-llama--Meta-Llama-3-70B-Instruct-query-prompt-None-15-shot_0 \
    --n 15 --o logs

python scripts/reasonableness/double_blind_generation.py \
    --exp logs/SemEvalOpenAI/gpt-3.5-turbo-query-prompt-None-15-shot_0 \
    --n 15 --out logs/analysis/reasonableness 

# liahr gpt4o
# baseline gpt4o, llama-3-70b

python scripts/reasonableness/double_blind_generation.py \
    --task query_prompt_vs_gt \
    --exp ./logs/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-None-15-shot_0 \
    --n 20 --o logs/analysis/reasonableness/query_prompt

python scripts/reasonableness/double_blind_generation.py \
    --task baseline_vs_gt \
    --exp ./logs/SemEvalOpenAIReason/gpt-4o-mini-2024-07-18-False-None-25-shot_0 \
    --n 20 --o logs/analysis/reasonableness/baseline

python scripts/reasonableness/double_blind_generation.py \
    --task baseline_vs_gt \
    --exp ./logs/SemEvalReason/meta-llama--Meta-Llama-3-70B-Instruct-False-None-25-shot_0 \
    --add_exp ./logs/SemEval/meta-llama--Meta-Llama-3-70B-Instruct-query-prompt-None-15-shot_0 \
    --n 20 --o logs/analysis/reasonableness/baseline


# goemotions liahr gpt 4o, baseline gpt4o

python scripts/reasonableness/double_blind_generation.py \
    --task query_prompt_vs_gt \
    --exp ./logs/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-None-15-shot_0 \
    --n 20 --o logs/analysis/reasonableness/query_prompt

python scripts/reasonableness/double_blind_generation.py \
    --task baseline_vs_gt \
    --exp ./logs/GoEmotionsOpenAIReason/gpt-4o-mini-2024-07-18-False-None-25-shot_0 \
    --n 20 --o logs/analysis/reasonableness/baseline

python scripts/reasonableness/double_blind_test.py \
    --task query_prompt_vs_gt \
    --exp logs/analysis/reasonableness/query_prompt/SemEval/meta-llama--Meta-Llama-3-70B-Instruct-query-prompt-None-15-shot_0 \
    logs/analysis/reasonableness/query_prompt/SemEvalOpenAI/gpt-4-turbo-2024-04-09-query-prompt-None-15-shot_0 \
    logs/analysis/reasonableness/query_prompt/SemEvalOpenAI/gpt-3.5-turbo-query-prompt-None-15-shot_0 \
    logs/analysis/reasonableness/query_prompt/SemEvalOpenAI/gpt-4o-mini-2024-07-18-query-prompt-None-15-shot_0 \
    logs/analysis/reasonableness/query_prompt/GoEmotionsOpenAI/gpt-4o-mini-2024-07-18-query-prompt-None-15-shot_0

python scripts/reasonableness/double_blind_test.py \
    --task icl_vs_gt \
    --exp logs/analysis/reasonableness/icl/SemEvalOpenAI/gpt-3.5-turbo-random-retrieval-35-shot_0 \
    logs/analysis/reasonableness/icl/SemEvalOpenAI/gpt-4-1106-preview-similarity-retrieval-25-shot_0

python scripts/reasonableness/double_blind_test.py \
    --task baseline_vs_gt \
    --exp logs/analysis/reasonableness/baseline/SemEvalReason/meta-llama--Meta-Llama-3-70B-Instruct-False-None-25-shot_0 \
    logs/analysis/reasonableness/baseline/SemEvalOpenAIReason/gpt-4o-mini-2024-07-18-False-None-25-shot_0 \
    logs/analysis/reasonableness/baseline/GoEmotionsOpenAIReason/gpt-4o-mini-2024-07-18-False-None-25-shot_0