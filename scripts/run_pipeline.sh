for DATASET_NAME in fever fiqa hotpotqa msmarco musique nq qampari scifact
do
    mkdir -p ./datasets/${DATASET_NAME}_train/128k_summary/gpt_4o_mini

    python ./codes/1_gpt_summarize.py  \
        --data_path ./datasets/${DATASET_NAME}_train/128k/corpus.jsonl \
        --output_dir_path ./datasets/${DATASET_NAME}_train/128k_summary/gpt_4o_mini 

    python ./codes/2_gpt_preprocess.py \
        --gpt_results_dir_path ./datasets/${DATASET_NAME}_train/128k_summary/gpt_4o_mini \
        --output_dir_path ./datasets/${DATASET_NAME}_train/128k_summary/gpt_4o_mini 

    mkdir -p ./datasets/${DATASET_NAME}_train/128k_summary/mistral_7b_0.3_inst
    mkdir -p ./datasets/${DATASET_NAME}_train/128k_summary/phi3_mini_4k

    python ./codes/3_lclm_summarize.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.3 \
        --max_length 32000 --max_new_tokens 2048 \
        --input_filepath "./datasets/${DATASET_NAME}_train/128k/corpus.jsonl" \
        --output_filepath "./datasets/${DATASET_NAME}_train/128k_summary/mistral_7b_0.3_inst/corpus.jsonl"

    python ./codes/3_lclm_summarize.py \
        --model_id microsoft/Phi-3-mini-4k-instruct \
        --max_length 32000 --max_new_tokens 2048 \
        --input_filepath "./datasets/${DATASET_NAME}_train/128k/corpus.jsonl" \
        --output_filepath "./datasets/${DATASET_NAME}_train/128k_summary/phi3_mini_4k/corpus.jsonl"

    # evaluation 
    mkdir -p ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_4o_mini
    mkdir -p ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_phi
    mkdir -p ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_mistral

    python ./codes/4_evaluation_summary.py \
        --dataset ${DATASET_NAME} \
        --input_summary_dir gpt_4o_mini \
        --output_path ./outputs \
        --output_dir ${DATASET_NAME}_train/128k_summary/gpt_4o_mini_4o_mini \

    python ./codes/5_run_gpt.py \
        --batch_input_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_4o_mini/batch_query_summary.jsonl \
        --batch_output_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_4o_mini/batch_query_summary_results.jsonl


    python ./codes/4_evaluation_summary.py \
        --dataset ${DATASET_NAME} \
        --input_summary_dir phi3_mini_4k \
        --output_path ./outputs \
        --output_dir ${DATASET_NAME}_train/128k_summary/gpt_4o_mini_phi \

    python ./codes/5_run_gpt.py \
        --batch_input_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_phi/batch_query_summary.jsonl \
        --batch_output_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_phi/batch_query_summary_results.jsonl

    python ./codes/4_evaluation_summary.py \
        --dataset ${DATASET_NAME} \
        --input_summary_dir mistral_7b_0.3_inst \
        --output_path ./outputs \
        --output_dir ${DATASET_NAME}_train/128k_summary/gpt_4o_mini_mistral \

    python ./codes/5_run_gpt.py \
        --batch_input_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_mistral/batch_query_summary.jsonl \
        --batch_output_filepath ./outputs/${DATASET_NAME}_train/128k_summary/gpt_4o_mini_mistral/batch_query_summary_results.jsonl

    python ./codes/6_convert_file.py \
        --dataset ${DATASET_NAME}_train \
        --result_dir gpt_4o_mini_4o_mini \
        --output_postfix 4m_4m
    
    python ./codes/6_convert_file.py \
        --dataset ${DATASET_NAME}_train \
        --result_dir gpt_4o_mini_phi \
        --output_postfix 4m_phi

    python ./codes/6_convert_file.py \
        --dataset ${DATASET_NAME}_train \
        --result_dir gpt_4o_mini_mistral \
        --output_postfix 4m_mistral
done


# -- after all train data
python ./codes/7_make_orpo_dataset.py \
    --output_dir_path ./datasets/x_train/orpo

bash ./codes/8_train_orpo.sh


# -- evaluation with color
for DATASET_NAME in fever fiqa hotpotqa msmarco musique nq qampari scifact
do
    python 9_color_summarize.py \
        --model_id "./codes/color-phi-orpo-b8-lr1e6-beta2.5" \
        --input_filepath "./datasets/${DATASET_NAME}/128k/corpus.jsonl" \
        --output_filepath "./outputs/${DATASET_NAME}/128k_summary/color/corpus.jsonl"
    
    python ./codes/10_evaluation.py \
        --dataset ${DATASET_NAME} \
        --input_summary_dir color \
        --output_path ./outputs \
        --output_dir ${DATASET_NAME}/128k_summary/gpt_4o_mini_color \

    python ./codes/11_run_gpt.py \
        --batch_input_filepath ./outputs/${DATASET_NAME}/128k_summary/gpt_4o_mini_color/batch_query_summary.jsonl \
        --batch_output_filepath ./outputs/${DATASET_NAME}/128k_summary/gpt_4o_mini_color/batch_query_summary_results.jsonl

    mkdir -p ./outputs/${DATASET_NAME}/128k_summary/loft_eval
    python ./codes/12_convert_file.py \
        --dataset ${DATASET_NAME} \
        --result_dir gpt_4o_mini_color \
        --output_postfix 4m_color
    
    python 13_loft_run_evaluation.py \
        --answer_file_path ./datasets/${DATASET_NAME}/128k/test_queries.jsonl \
        --pred_file_path ./outputs/${DATASET_NAME}/128k_summary/loft_eval/4m_color_preds.jsonl \
        --task_type retrieval
done
