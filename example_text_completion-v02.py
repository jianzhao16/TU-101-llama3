import pandas as pd  # Import pandas
import fire
from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_csv: str = './input-txt.csv', # input CSV file path
    output_csv: str = './out-txt.csv',  # Parameter for output CSV, with a default value
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Read prompts from CSV file
    try:
        data = pd.read_csv(input_csv)
        prompts_original: List[str] = data['promptupd'].tolist()
        # only run 1 query
        prompts = [prompts_original[0]]  # Ensure prompts is a list containing only the first item
    except Exception as e:
        print(f"Failed to read prompts from {input_csv}: {e}")
        return

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Process and output results
    for prompt, result in zip(prompts, results):
        print(f"> {result['generation']}")

    # Write results to a CSV file using DataFrame
    results_data = {'prompt': prompts, 'completion': [result['generation'] for result in results]}
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Results have been written to CSV at {output_csv}.")

if __name__ == "__main__":
    fire.Fire(main)
