import random
import jsonlines
import json

def make_orpo_dataset(result_mistral_corpus_path, result_phi_corpus_path, result_4m_corpus_path,
                      acc_dict_dir, output_path, acc_dict_dir_name="acc_dict", save_orpo_type=""):
    
    RESULT_CORPUS_PATH={
        f'4m_mistral': result_mistral_corpus_path, #  './datasets/scifact_train/all_summary/mistral_7b_0.3_inst/corpus.jsonl', 
        f'4m_phi': result_phi_corpus_path, # './datasets/scifact_train/all_summary/phi3_mini_4k/corpus.jsonl', 
        f'4m_4m': result_4m_corpus_path , # './datasets/scifact_train/all_summary/corpus.jsonl'
    }
    summary_name_lst=[f'4m_mistral', f'4m_phi', f'4m_4m']
    summary_dict = {}

    for name in summary_name_lst:
        lines = []
        with jsonlines.open(RESULT_CORPUS_PATH[name]) as f:
            for line in f.iter():
                lines.append(line)
        summary_dict[name] = lines

    aggregate_results_dict={}
    for name in summary_name_lst:
        # with open(f'./outputs/{dataset_name}/all_summary/acc_dict/{name}_acc_dict.json') as f:
        with open(f'{acc_dict_dir}/{acc_dict_dir_name}/{name}_acc_dict.json') as f:
            result_json = json.load(f)
        aggregate_results_dict[name] = result_json


    result_keys = [*aggregate_results_dict[f'4m_4m'].keys()]

    final_results_dict = {}
    for key in result_keys:
        choosen_lst = []
        reject_lst = []
        for name in summary_name_lst:

            tmp_len_dict={}
            for name in summary_name_lst:
                tmp_len_dict[name] = len(summary_dict[name][int(key)]['summary_text'])
             
            if acc_dict_dir_name=="acc_dict":
                if aggregate_results_dict[name][key]['acc'] == 1:
                    choosen_lst.append({name: tmp_len_dict[name]})
                else:
                    reject_lst.append({name: tmp_len_dict[name]})
            elif acc_dict_dir_name=="partial_acc_dict":
                if aggregate_results_dict[name][key]['acc'] > 0:
                    choosen_lst.append({name: tmp_len_dict[name]})
                else:
                    reject_lst.append({name: tmp_len_dict[name]})

        # weak
        if len(reject_lst)==0 or len(choosen_lst) == 0:
            sorted_tmp_len_dict_lst = sorted(tmp_len_dict.items(), key = lambda item: item[1]) 
            reject = sorted_tmp_len_dict_lst[-1]
            choosen = sorted_tmp_len_dict_lst[0] # shortest
        else:
            sorted_reject_lst = sorted(reject_lst.items(), key = lambda item: item[1]) 
            reject = sorted_reject_lst[-1] # longest

            sorted_choosen_lst = sorted(choosen_lst.items(), key = lambda item: item[1]) 
            choosen = sorted_choosen_lst[0] #shortest
        
        
        if save_orpo_type=="hard" and len(choosen_lst)==0:
            continue        

        reject_line = summary_dict[reject[0]][int(key)]
        choosen_line = summary_dict[choosen[0]][int(key)]
        
        final_results_dict[int(key)] = {
            'prompt': f"Summarize the following content. \nContent:\n{reject_line['passage_text']}\n Summary:",
            'rejected': [
                {"role": "user", "content": f"Summarize the following content. \nContent:\n{reject_line['passage_text']}\n Summary:"},
                {'role': 'assistant', 'content': reject_line['summary_text']}
            ],
            'chosen': [
                {"role": "user", "content": f"Summarize the following content. \nContent:\n{choosen_line['passage_text']}\n Summary:"},
                {'role': 'assistant', 'content': choosen_line['summary_text']}
            ],
            # 'reject_lst': reject_lst,
            # 'choosen_lst': choosen_lst
        }   
    
    print(f"len(final_results_dict): {len(final_results_dict)}")
    # import pdb; pdb.set_trace()

    output_dir_path=os.path.dirname(output_path)
    if not os.path.exists(output_dir_path):
        print(f"[DIRECTORY MAKING] {output_dir_path}")
        os.mkdir(output_dir_path)

    print(output_path)
    
    # with open(f'./datasets/scifact_train/all_summary/orpo/dataset.json', 'w') as f:
    with open(output_path, 'w') as f:
        for key, value in sorted(final_results_dict.items()):
            json.dump(value, f)
            f.write('\n')

def read_lines(filepath):
    lines=[]
    with jsonlines.open(filepath) as f:
        for line in f.iter():
            lines.append(line)
    return lines
    
    
def main(args):
    output_dir_path = args.output_dir_path

    for args_dataset in ["fever_train", "fiqa_train", "hotpotqa_train", "msmarco_train", "musique_train", "nq_train", "qampari_train", "scifact_train"]
        make_orpo_dataset(
            result_mistral_corpus_path=f"./datasets/{args_dataset}/128k_summary/mistral_7b_0.3_inst/corpus.jsonl", 
            result_phi_corpus_path=f"./datasets/{args_dataset}/128k_summary/phi3_mini_4k/corpus.jsonl", 
            result_4m_corpus_path=f"./datasets/{args_dataset}/128k_summary/gpt_4o_mini/corpus.jsonl",
            acc_dict_dir=f"./outputs/{args_dataset}/128k_summary", 
            acc_dict_dir_name="acc_dict", 
            
            output_path=f"./datasets/{args_dataset}/128k_summary/orpo/dataset_hard.json",
            
            save_orpo_type="hard")

    # merge all things
    total_lines=[]
    total_lines += read_lines('./datasets/fever_train/128k_summary/orpo/dataset_hard.json')
    total_lines += read_lines('./datasets/fiqa_train/128k_summary/orpo/dataset_hard.json') 
    total_lines += read_lines('./datasets/hotpotqa_train/128k_summary/orpo/dataset_hard.json') 
    total_lines += read_lines('./datasets/msmarco_train/128k_summary/orpo/dataset_hard.json') 
    total_lines += read_lines('./datasets/musique_train/128k_summary/orpo/dataset_hard.json')
    total_lines += read_lines('./datasets/nq_train/128k_summary/orpo/dataset_hard.json')
    total_lines += read_lines('./datasets/qampari_train/128k_summary/orpo/dataset_hard.json') 
    total_lines += read_lines('./datasets/scifact_train/128k_summary/orpo/dataset_hard.json') 
    random.shuffle(total_lines)

    
    val_len = len(total_lines)//10
    
    train_lines = total_lines[:-val_len]
    val_lines = total_lines[val_len+1:]

    with open(f'{output_dir_path}/color_train.jsonl', 'w') as f:
        for line in train_lines:
            json.dump(line, f)
            f.write('\n')
    
    with open(f'{output_dir_path}/color_val.jsonl', 'w') as f:
        for line in val_lines:
            json.dump(line, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", type=str)

    args = parser.parse_args()
    main(args)

