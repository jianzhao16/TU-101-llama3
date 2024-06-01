#!/bin/bash
torchrun --nproc_per_node 1 example_chat_completion-v02.py     --ckpt_dir Meta-Llama-3-8B-Instruct/     --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model     --max_seq_len 128 --max_batch_size 4 >> out-chat.txt
