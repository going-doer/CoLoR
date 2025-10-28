import jsonlines
from utils import convert_gpt_output_to_prediction,print_results

# _train
def main(args):
  args_dataset = args.dataset
  result_dir = args.result_dir
  output_postfix = args.output_postfix
  
  # for result_dir, output_postfix in zip(['gpt_4o_mini_4o_mini', 'gpt_4o_mini_phi', 'gpt_4o_mini_mistral'], ['4m', 'phi', 'mistral']):
  print_results(
  # print_results_multiple_answers(
        input_dir=f"{args_dataset}/128k_summary", 
        input_corpus_filename=f"corpus.jsonl",
        test_dir=f"{args_dataset}/128k", 
        test_split=f"train", 
        result_path="./outputs", 
        result_dir=f"{args_dataset}/128k_summary/{result_dir}", 
        result_file_name=f"batch_query_summary_results.jsonl",
        save_output_path=f"{args_dataset}/128k_summary/acc_dict/4m_{output_postfix}_acc_dict.json")
      # save_output_path=f"{args_dataset}/128k_summary/partial_acc_dict/4m_4m_acc_dict.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--output_postfix", type=str)

    args = parser.parse_args()
    main(args)

