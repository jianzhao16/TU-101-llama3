# TU-061-LLM
LLM

##pyenv

Step 1: Install Required Dependencies
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git



Step 2: Install pyenv

curl https://pyenv.run | bash



Step 3: Configure Your Environment

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

source ~/.bashrc  # Or source ~/.zshrc if you use Zsh



Step 4: Verify Installation

pyenv --version

Step 5: Install Python Versions

pyenv install --list


pyenv global 3.8.5



pyenv local 3.10.1


python --version

******************************************************************
Create Virtual Python in Some Projects

pyenv local 3.10.1
python --version
python -m venv ubutu2004-py10
source ubutu2004-py10/bin/activate

******************************************************************



Conda (Optinal Not Necessary)


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create -n ubu2004py10 python=3.10
conda activate ubu2004py10
conda install numpy

conda deactivate



******************************************************************



Run Cmd : 

at last I choose virtaul Python 3.10 offered by the system


(ubu2204-py10) tus35240@gpu:~/mydata/mygit/llama3$ 

torchrun --nproc_per_node 1 example_text_completion.py     --ckpt_dir Meta-Llama-3-8B/     --tokenizer_path Meta-Llama-3-8B/tokenizer.model     --max_seq_len 128 --max_batch_size 4


torchrun --nproc_per_node 1 example_chat_completion-v01-good.py     -
-ckpt_dir Meta-Llama-3-8B-Instruct/     --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model     --max_seq_len 128 --max_batch_size 4



