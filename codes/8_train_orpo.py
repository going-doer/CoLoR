# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the ORPO training script with the following command with some example arguments.
In general, the optimal configuration for ORPO will be similar to that of DPO without the need for a reference model:

"""
# FlashAttention only support fp16 and bf16 data type

import multiprocessing
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ORPOConfig, get_peft_config # ORPOTrainer
from custom_orpo_trainer import Custom3ORPOTrainer

import torch 

@dataclass
class ScriptArguments:
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style",
        metadata={"help": "The name of the dataset to use."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ORPOConfig, ModelConfig))
    args, orpo_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    if model_config.model_name_or_path == "google/gemma-2-2b-it":
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.bfloat16, 
            device_map="auto" 
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
            torch_dtype="auto", 
            attn_implementation="flash_attention_2",
            device_map="auto" 
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
        cache_dir="/projects/minju/cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    
    ds = load_dataset('json', data_files='./datasets/x_train/orpo/color_train.jsonl') 
    val_ds = load_dataset('json', data_files='./datasets/x_train/orpo/color_val.jsonl')

    if orpo_args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=1 if orpo_args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    val_ds = val_ds.map(
        process,
        num_proc=1 if orpo_args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    # eval_dataset = ds["test"]
    eval_dataset = val_ds["train"]

    ################
    # Training
    ################
    trainer = Custom3ORPOTrainer(
        model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # train and save the model
    trainer.train()
    trainer.save_model(orpo_args.output_dir)