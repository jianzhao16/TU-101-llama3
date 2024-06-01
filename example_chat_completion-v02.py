
from typing import List

import fire

from llama import Llama


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
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # "I believe the meaning of life is",
        #"Simply put, the theory of relativity states that ",
        #A brief message congratulating the team on the launch:

        # Hi everyone,

        #I just bash: y: command not found

        # Few shot prompt (providing a few examples before asking model to complete more);
        """Extract the type of service and zipcode from the following user query:

        I want to food service nearby 19140 => service:Food, zipcode:19140, status:okay
        He want to food lunch nearby 19132 => service:Food, zipcode:19132, status:okay
        find Meal services nearby Temple University Center City Campus =>  service:Food, zipcode:19102, status:okay
        help find Temporary housing nearby 1920 N Van Pelt St, Philadelphia, PA => service:Shelter, zipcode:19121, status:okay
        Counseling services => service: Mental Health, zipcode:000000, status:okay
    	Emergency food support => service:Food, zipcode:000000, status:okay
        Meal services => service: Food, zipcode:000000, status:okay
        Temporary housing => service:Shelter,zipcode:000000, status:okay
        Counseling services => service:Mental Health, zipcode:000000, status:okay
        Emergency food support nearby 1200 N Dupont Hwy, Dover, DE => service: Food, zipcode:19901, status:okay
        Mental health counseling => service:Mental Health, zipcode:000000, status:okay
    	Mental health counseling is close to 160 E Erie Ave, Philadelphia, PA => service: Mental Health, zipcode: 19134, status:okay
        He want to find Mental Health service nearby 19123 =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        #print(prompt)
        print(f"> {result['generation']}")
        #print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
