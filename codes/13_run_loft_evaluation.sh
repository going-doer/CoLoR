python run_evaluation.py \
  --answer_file_path ./datasets/webis_touche2020/128k/test_queries.jsonl \
  --pred_file_path ./outputs/webis_touche2020/128k_summary/loft_eval/4m_custom3_orpo_xmulti_hard_01_short_b8_1e6_beta2.5_preds.jsonl \
  --task_type retrieval


python run_evaluation.py \
    --answer_file_path ${BASE_DIR}/data/${TASK_TYPE}/${DATASET}/${LENGTH}/dev_queries.${answer_file_extension} \
    --pred_file_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --task_type ${TASK_TYPE}



# BASE_DIR=$1
# DATASET=$2
# LENGTH="128k"
# TASK_TYPE="retrieval"
# SPLIT="dev"
# PROMPT_TYPE="few_shot_with_cot"
# PROMPT="${TASK_TYPE}_${DATASET}_${LENGTH}_${SPLIT}:${PROMPT_TYPE}"
# echo "Prompt: ${PROMPT}"

# mkdir -p ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}
# answer_file_extension="jsonl"

# python run_evaluation.py \
#     --answer_file_path ${BASE_DIR}/data/${TASK_TYPE}/${DATASET}/${LENGTH}/dev_queries.${answer_file_extension} \
#     --pred_file_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
#     --task_type ${TASK_TYPE}