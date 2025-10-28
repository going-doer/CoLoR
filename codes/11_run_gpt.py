import jsonlines
from utils import api_call
import argparse
import json

def main(args):
    batch_input_filepath = args.batch_input_filepath
    batch_output_filepath = args.batch_output_filepath
    
    
    # with jsonlines.open(f'{output_path}/{output_dir}/batch_query_{args_hyperparameter}_summary.jsonl') as f:
    
    results_lines=[]

    with jsonlines.open(f'{batch_input_filepath}') as f:
        for line in f.iter():
            import pdb; pdb.set_trace()
            
            completion = api_call(completion_args=line['body'])
            results_lines.append({
                'id': line['custom_id'],
                'custom_id': line['custom_id'],
                'response': {
                    'body': completion
                }
            })
    
    with open(f'{batch_output_filepath}') as f:
        for line in results_lines:
            json.dump(line, f)
            f.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpt_results_dir_path", type=str)
    parser.add_argument("--batch_input_filepath", type=str)
    parser.add_argument("--batch_output_filepath", type=str)
    
    args = parser.parse_args()
    main(args)

