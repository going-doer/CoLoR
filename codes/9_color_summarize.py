
# CUDA_VISIBLE_DEVICES=0 python 3_lclm_summarize.py \
#     --max_length 20000 \
#     --model_id "./codes/color-phi-orpo-b8-lr1e6-beta2.5" \
#     --input_filepath "./datasets/scifact/128k/corpus.jsonl" \
#     --output_filepath "./outputs/scifact/128k_summary/custom_orpo_xfact_hard_short_01/corpus_custom3_b8_1e6_beta2.5_results.jsonl"

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import jsonlines
from tqdm import tqdm
import json
import copy
from utils import lclm_summarize

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.random.manual_seed(0)

if __name__ == '__main__':                                                                                                                                             
    parser = argparse.ArgumentParser()                                                                                                                                 
    parser.add_argument("--model_id") 
    parser.add_argument("--max_length", default=2048) 
    parser.add_argument("--max_new_tokens", type=int, default=1024) 
    parser.add_argument("--input_filepath") 
    parser.add_argument("--output_filepath")

    args = parser.parse_args()

    lclm_summarize(args.model_id, args.max_length, args.max_new_tokens, args.input_filepath, args.output_filepath)
