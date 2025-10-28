import torch
from typing import List, Optional

import jsonlines
import fire
import json # mj
from tqdm import tqdm # mj
import pickle
import copy
import os
import traceback
import time

## seed fixed
import torch 
import numpy as np
import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

from openai import OpenAI

OPENAI_API_KEY=""
class ChatGPT(object):
    def __init__(self):
        super(ChatGPT, self).__init__()
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def generate(self, content, max_gen_len=512):    

        messages=[
            {"role": "user", "content": f"Summarize the following content. \nContent:\n{content}\n Summary:"},
        ]
            
        # temperature, top_p 
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini", # "gpt-3.5-turbo-0125",
            messages=messages,
            logprobs=True,
            max_tokens=max_gen_len,
        )

        # gpt-4o-mini
        in_usage = completion.usage.prompt_tokens * 0.150 / 1000000
        out_usage = completion.usage.completion_tokens * 0.6 / 1000000
        cost = (in_usage + out_usage)
        
        results = {
            'created': completion.created,
            'role': completion.choices[0].message.role,
            'content': completion.choices[0].message.content,
        }

        logprobs = [o.logprob for o in completion.choices[0].logprobs.content]
        

        result_dict = {
            "results": results, 
            "probabilities": sum(logprobs)/len(logprobs),
            'logprobs': logprobs, 
            'usage_info': {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens,
            },
            'cost': cost,
        }
        return result_dict, completion

def main(
    data_path: str,
    output_dir_path: str,
    postfix: str = "",
):
    lang_model = ChatGPT()
    total_results = []
    output_results = []
    output_raw_results= []
    total_cost = 0
    error_summaries = []

    input_lines=[]
    with jsonlines.open(data_path) as f:
        for line in f.iter():
            input_lines.append(line)

    print(f"{data_path}: {len(input_lines)}")
    
    for idx, data in enumerate(tqdm(input_lines)):
        success = False
        while not success:
            try:
                output, output_raw = lang_model.generate(data['passage_text'], max_gen_len=512)
                success = True
                result_dict = {
                    "idx": idx,
                    "input": data['passage_text'],
                    "input_data": data,
                    "results": output["results"],
                    "probabilities": output["probabilities"],
                    'logprobs': output['logprobs'],
                    'usage_info': output['usage_info'],
                    'cost': output['cost']
                }
                total_results.append(result_dict)
                output_results.append(output)
                # output_raw_results.append(output_raw)
                
                with open(f"{output_dir_path}/{postfix}streaming_results.jsonl", 'a') as outfile:
                    outfile.write(json.dumps(result_dict))
                    outfile.write('\n')
                    
                total_cost += output['cost']

                
            except Exception as e:
                import pdb; pdb.set_trace()
                error_msg = traceback.format_exc()
                error_summaries.append(error_msg)
                time.sleep(9)


    with open(f"{output_dir_path}/{postfix}total_results.jsonl", 'a') as outfile:
        for total_result in total_results:
            outfile.write(json.dumps(total_result))
            outfile.write('\n')

    with open(f"{output_dir_path}/{postfix}output_results.jsonl", 'a') as outfile:
        for output_result in output_results:
            outfile.write(json.dumps(output_result))
            outfile.write('\n')

    with open(f"{output_dir_path}/{postfix}error_msgs.jsonl", 'a') as outfile:
        for error_summary in error_summaries:
            outfile.write(f'{error_summary}\n')    
    

    print(f"TOTAL COST: {total_cost}")

if __name__ == "__main__":
    fire.Fire(main)

# arguana : 0.11021000000000009
# scifact: 0.12521300000000005



