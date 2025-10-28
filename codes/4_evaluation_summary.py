from utils import get_summary_full_prompt
import argparse

# for args_chunk_num in range(0,3):

def main(args):
    args_chunk_num = args.chunk_num
    dataset = args.dataset
    input_summary_dir = args.input_summary_dir
    output_path = args.output_path
    output_dir = args.output_dir
    
    get_summary_full_prompt(dataset=f"{dataset}", original_input_dir=f"{dataset}_train/128k_summary", 
                            original_input_summary_dir=f"{dataset}_train/128k_summary/{input_summary_dir}",
                            original_input_query_dir=f"{dataset}_train/128k",
                            input_corpus_file_name=f"corpus.jsonl",
                            gpt_model="gpt-4o-mini", gpt_max_tokens=512,  
                            test_split=f"train", 
                            content_text_type="summary_text",
                            output_path=f"{output_path}",
                            output_dir=f"{output_dir}", 
                            output_propmt_file_name=f"prompt_summary.txt", 
                            output_batch_query_file_name=f"batch_query_summary.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--input_summary_dir", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--output_dir", type=str)
    
    args = parser.parse_args()
    main(args)

