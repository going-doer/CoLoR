import jsonlines
import json
from openai import OpenAI


client = OpenAI(
    api_key = ''
)

DATASET_PATH="./datasets"

def serialize_completion(completion):
    return {
        "id": completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                    "function_call": {
                        "arguments": json.loads(
                            choice.message.function_call.arguments) if choice.message.function_call and choice.message.function_call.arguments else None,
                        "name": choice.message.function_call.name
                    } if choice.message and choice.message.function_call else None
                } if choice.message else None
            } for choice in completion.choices
        ],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "system_fingerprint": completion.system_fingerprint,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens
        }
    }



def api_call(completion_args):
    completion = client.chat.completions.create(**completion_args)
    # completion = client.chat.completions.create(
    #     model="chatgpt-4o-latest", 
    #     model=gpt_version,
    #     temperature=1, # default value
    #     messages=msg # [{'role': '~', 'content': ''}]
    # )

    # return completion # print(completion.choices[0].message)
    return serialize_completion(completion)
    


def get_summary_full_prompt(dataset="musique", 
                            original_input_dir="musique/128k", 
                            original_input_query_dir="", 
                            original_input_corpus_file_name="corpus.jsonl",

                            original_input_summary_dir="musique/128k_summary", 
                            original_input_summary_corpus_file_name="corpus.jsonl",
                            
                            gpt_model="gpt-4o", gpt_max_tokens=512, temperature=1.0,
                            test_split="test", content_text_type="summary_text",
                            output_path="./outputs", output_dir="musique/128k_summary", 
                            output_propmt_file_name="prompt.txt", output_batch_query_file_name="batch_query.jsonl"):
    # corpus
    output_lines = []
    with jsonlines.open(f'{DATASET_PATH}/{original_input_summary_dir}/{original_input_summary_corpus_file_name}') as f:
        idx = 0
        for line in f.iter():

            line_content_text = line[content_text_type]
            if content_text_type == "summary_text" and len(line_content_text) == 0:
                line_content_text = line['passage_text'] # for empty summary text
                print(f"empty summary {idx}")

                # 1004 selective context의 경우
                # reduce_length = int(len(line['passage_text'])*0.3)
                # line_content_text = line['passage_text'][:reduce_length]

            if len(line['title_text']) == 0:
                if content_text_type == "title_text":
                    raise Exception("There is no title")
                
                # if content_text_type=="summary_text" and content_text_type not in line: 
                #     line_lst = line['passage_text'].split()
                #     line[content_text_type] = line_lst[:200]
                #     print(idx)
                # output_lines.append(f"ID: {idx} | CONTENT: {line[content_text_type]} | END ID: {idx}")
                output_lines.append(f"ID: {idx} | CONTENT: {line_content_text} | END ID: {idx}")

            else: # 0925
                if content_text_type == "title_text":
                    output_lines.append(f"ID: {idx} | TITLE: {line['title_text']} | END ID: {idx}")
                else:
                    # output_lines.append(f"ID: {idx} | TITLE: {line['title_text']} | CONTENT: {line[content_text_type]} | END ID: {idx}")
                    output_lines.append(f"ID: {idx} | TITLE: {line['title_text']} | CONTENT: {line_content_text} | END ID: {idx}")
            idx += 1

    # 0827
    if not os.path.exists(f"{output_path}/{output_dir}"):
        os.makedirs(f"{output_path}/{output_dir}")
        print(f"[DIRECTORY MAKING] {output_path}/{output_dir}")

    with open(f'{output_path}/{output_dir}/{output_propmt_file_name}', 'a') as f:
        for line in output_lines:
            f.write(line)
            f.write("\n\n")

    # few-shot demonstrations
    original_corpus_dict = {}
    with jsonlines.open(f'{DATASET_PATH}/{original_input_dir}/{original_input_corpus_file_name}') as f:
        idx = 0
        for line in f.iter():
            original_corpus_dict[f"{line['pid']}"] = line
            original_corpus_dict[f"{line['pid']}"]['idx'] = f"{idx}"
            idx += 1

    output_lines = []
    with jsonlines.open(f'{DATASET_PATH}/{original_input_dir}/few_shot_queries.jsonl') as f:
        idx = 1
        for line in f.iter():
            output_lines.append(f"====== Example {idx} ======")
            
            output_lines.append("Which document is most relevant to answer the query? Print out the TITLE and ID of the document. Then format the IDs into a list.")
            output_lines.append("If there is no perfect answer output the closest one. Do not give an empty final answer.")
            output_lines.append(f"query: {line['query_text']}")
            output_lines.append("The following documents can help answer the query:")
            output_lines.append("")
            answers = []
            for qid, _ in line['answers']:
                try:
                    if len(original_corpus_dict[qid]['title_text']) == 0:
                        output_lines.append(f"ID: {original_corpus_dict[qid]['idx']}")
                    else:   
                        output_lines.append(f"TITLE: {original_corpus_dict[qid]['title_text']} | ID: {original_corpus_dict[qid]['idx']}")
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                answers.append(original_corpus_dict[qid]['idx'])

            output_lines.append("")
            output_lines.append(f"Final Answer: {answers}")
            output_lines.append("")
            idx += 1

    with open(f'{output_path}/{output_dir}/{output_propmt_file_name}', 'a') as f:
        for line in output_lines:
            f.write(f"{line}\n")


    # batch_query
    with open(f'{output_path}/{output_dir}/{output_propmt_file_name}') as f:
        lines = f.readlines()


    output_jsonl = []

    print_out_target = "TITLE and ID"
    # if dataset == "arguana": 
    #     print_out_target = "ID"
    output_json = {
        'custom_id': '',
        'method': "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": gpt_model, 
            "temperature": temperature, # 0827 추가
            "messages": [
                {"role": "system", "content": f"""{SYSTEM_PROMPT[dataset]} Print out the {print_out_target} of each document.

Your final answer should be a list of IDs, in the following format:
Final Answer: [id1, id2, ...]
If there is only one ID, it should be in the format:
Final Answer: [id1]

If there is no perfect answer output the closest one. Do not give an empty final answer.

"""},],"max_tokens": gpt_max_tokens}
    }

    cic_prompt_txt = ""
    for line in lines:
        cic_prompt_txt += line

    test_split_queries_path = f'{DATASET_PATH}/{original_input_dir}/{test_split}_queries.jsonl'
    if not os.path.isfile(test_split_queries_path):
        test_split_queries_path = f'{DATASET_PATH}/{original_input_summary_dir}/{test_split}_queries.jsonl'
    if len(original_input_query_dir) > 0:
        test_split_queries_path = f'{DATASET_PATH}/{original_input_query_dir}/{test_split}_queries.jsonl'
    with jsonlines.open(test_split_queries_path) as f:
        idx = 1
        for line in f.iter():
            tmp_cic_prompt_txt = cic_prompt_txt
            tmp_cic_prompt_txt += f"""
====== Now let's start! ======
Which document is most relevant to answer the query? Print out the {print_out_target} of the document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {line['query_text']}
The following documents can help answer the query:"""

            tmp_output_json = copy.deepcopy(output_json)
            tmp_output_json['custom_id'] = f"request-{idx}"
            tmp_output_json['body']['messages'].append({"role": "user", "content": tmp_cic_prompt_txt})
            output_jsonl.append(tmp_output_json)
            idx += 1

    with open(f'{output_path}/{output_dir}/{output_batch_query_file_name}', 'w') as f:
        for ojson in output_jsonl:  
            json.dump(ojson, f)
            f.write("\n")

    # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    # {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}




def print_results(input_dir="scifact/128k", 
                  input_corpus_filename="corpus.jsonl",
                  test_dir="",
                  test_split="dev", 
                  result_path="/d1/minjuseo/workspace/LCLMs/outputs", result_dir="scifact/128k", 
                  result_file_name="summary_result.jsonl",
                  save_output_path=""):
    corpus_qid_dict = {}
    corpus_idx_to_qid = {}
    with jsonlines.open(f'{DATASET_PATH}/{input_dir}/{input_corpus_filename}') as f:
        idx = 0
        for line in f.iter():
            tdict = copy.deepcopy(line)
            tdict['idx'] = idx
            corpus_qid_dict[f"{line['pid']}"] = tdict
            corpus_idx_to_qid[idx] = f"{line['pid']}"
            idx += 1

    answers_dict = {}
    if len(test_dir) == 0:
        test_dir = input_dir

    with jsonlines.open(f'{DATASET_PATH}/{test_dir}/{test_split}_queries.jsonl') as f:
        idx = 0
        for line in f.iter():
            answer_lst = []
            for answer in line['answers']:
                # answer_idx = corpus_qid_dict[answer[0]]['idx']
                answer_lst.append(corpus_qid_dict[answer[0]])
            answers_dict[idx] = {
                'answer_lst': [f"{answer['idx']}" for answer in answer_lst],
                'answers_info': answer_lst
            }
            idx += 1

    acc_cnt = 0
    answers = []
    acc_dict = {}
    with jsonlines.open(f'{result_path}/{result_dir}/{result_file_name}') as f:
        idx = 0
        for line in f.iter():
            response = line['response']['body']['choices'][0]['message']['content']
            acc_str = "x"

            try:
                response_answer_lst = eval(response.split("\n")[-1].split(": ")[-1])
                response_answer_lst = [f"{tansw}" for tansw in response_answer_lst]
                union = list(set(response_answer_lst) & set(answers_dict[idx]['answer_lst']))
            
                if len(union) == len(answers_dict[idx]['answer_lst']) and len(response_answer_lst) == len(answers_dict[idx]['answer_lst']):
                    # print(f"{idx}\t{response_answer_lst}\t{answers_dict[idx]['answer_lst']}")
                    acc_cnt += 1
                    acc_str = "o"
                    
            except Exception as e:
                print("="*40)
                print(e)
                num_tokens = num_tokens_from_string(response)
                print(f"{idx}\t{num_tokens}: {response}")
                response_answer_lst = []
            
            acc_dict[idx] = {
               'label': answers_dict[idx]['answer_lst'],
                'response': response_answer_lst,
                'acc': 1 if acc_str == "o" else 0 # 1 = o
            }
            
            answers.append(response_answer_lst)
            idx += 1

    print(f"{input_dir} acc_cnt: {acc_cnt}")

    if len(save_output_path) > 0:
        with open(f'/d1/minjuseo/workspace/LCLMs/outputs/{save_output_path}', 'w') as f:
            json.dump(acc_dict, f)

    return answers



def lclm_summarize(model_id, max_length, max_new_tokens, input_filepath, output_filepath):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        # torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        cache_dir='/c1/minjuseo/workspace/hf_flant5/cache',
        # attn_implementation="flash_attention_2",
        device_map="auto" # minju
    )

    assert torch.cuda.is_available(), "This model needs a GPU to run ..."
    # device = torch.cuda.current_device()
    device ='cuda' 
    

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir='/c1/minjuseo/workspace/hf_flant5/cache')


    input_lines = []
    with jsonlines.open(input_filepath) as f:
        for line in f.iter():
            input_lines.append(line)

    output_lines = []
    for line in tqdm(input_lines):

        messages=[
                {"role": "user", "content": f"Summarize the following content. \nContent:\n{line['passage_text']}\n Summary:"},
        ]

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", max_length=max_length).to(device) # model.device
        output_json = copy.deepcopy(line)
        try:
            # output = pipe(line['messages'], **generation_args)
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            # outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, num_return_sequences=5, eos_token_id=tokenizer.eos_token_id)


            if len(outputs) == 1:
                sample_output = outputs[0]
                decoded_text = tokenizer.decode(sample_output[len(inputs[0]):], skip_special_tokens=True)
                output_json[f'summary_text'] = decoded_text
            else:
                for i, sample_output in enumerate(outputs):
                    decoded_text = tokenizer.decode(sample_output[len(inputs[0]):], skip_special_tokens=True)
                    output_json[f'summary_{i}'] = decoded_text
            
        except Exception as e:
            print("--Exception--")
            print(e)
            # import pdb; pdb.set_trace()

        output_lines.append(output_json) 
    
    with open(output_filepath, 'w') as f:
        for line in output_lines:
            json.dump(line, f)
            f.write('\n')