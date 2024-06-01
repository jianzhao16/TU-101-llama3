# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List

import fire

from llama import Llama

import time

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    # Start the timer
    start_time = time.time()

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Load Model Elapsed time: {elapsed_time:.2f} seconds")


    start_time = time.time()


    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        #"I believe the meaning of life is",
        #"Simply put, the theory of relativity states that ",
        #""A brief message congratulating the team on the launch:

        #Hi everyone,

        #I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        #""Translate English to French:

        #sea otter => loutre de mer
        #peppermint => menthe poivrée
        #plush girafe => girafe peluche
        #cheese =>""",

        #"How are you today",

        """Extract service and zipcode:
        I want food service nearby 19140 => service:Food, zip:19140, status:okay
        Counseling services => service: Mental Health, zip:000000, status:okay
        Temporary housing => service:Shelter, zip:000000, status:okay
        He want to find Mental Health service nearby 19123 =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print('Question:\n'+prompt+'\n')
        print(f"Answer: > {result['generation']}")
        print("\n==================================\n")

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    fire.Fire(main)
