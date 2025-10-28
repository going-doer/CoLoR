import jsonlines
import argparse
import json

def main(args):
# def preprocess_streaming_gpt(output_dir="nq/128k_summary"):
    output_lines = []
    cost = 0
    # with jsonlines.open(f'./datasets/{output_dir}/gpt_3.5_results/total_results.jsonl') as f:
    with jsonlines.open(f'{args.gpt_results_dir_path}/total_results.jsonl') as f:

        for line in f.iter():
            line['input_data']['summary_text'] = line['results']['content']
            output_lines.append(line['input_data'])

    with open(f'{args.output_dir_path}/corpus.jsonl', 'w') as f:
        for line in output_lines:
            json.dump(line,f)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_results_dir_path", type=str)
    parser.add_argument("--output_dir_path", type=str)
    
    args = parser.parse_args()
    main(args)

