import jsonlines
from utils import convert_gpt_output_to_prediction,print_results

# _train
def main(args):
    args_dataset = args.dataset
    result_dir = args.result_dir
    output_postfix = args.output_postfix

    print_results(
        input_dir=f"{args_dataset}/128k", 
        input_corpus_filename=f"corpus.jsonl",
        test_dir=f"{args_dataset}/128k", 
        test_split=f"test", 
        result_path="./outputs", 
        result_dir=f"{args_dataset}/128k_summary/{result_dir}", 
        result_file_name=f"batch_query_summary_results.jsonl",
        save_output_path=f"{args_dataset}/128k_summary/acc_dict/4m_{output_postfix}_acc_dict.json")

    try:
        convert_gpt_output_to_prediction(
            corpus_path=f"./datasets/{args_dataset}/128k/corpus.jsonl",
            acc_dict_path=f"./outputs/{args_dataset}/128k_summary/acc_dict/4m_{output_postfix}_acc_dict.json", 
            test_queries_path=f"./datasets/{args_dataset}/128k/test_queries.jsonl", 
            output_preds_path=f"./outputs/{args_dataset}/128k_summary/loft_eval/4m_{output_postfix}_preds.jsonl"
        )
        print(f"[DONE]: {args_dataset}\t{output_postfix}")
    except Exception as e:
        print(e)
        print(f"[XXXX]: {args_dataset}\t{output_postfix}")
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--output_postfix", type=str)

    args = parser.parse_args()
    main(args)
