#!/bin/bash
torchrun --nproc_per_node 1 example_text_completion-v02.py \
    --ckpt_dir Meta-Llama-3-8B/ \
    --tokenizer_path Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4  >> out-txt.txt
