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






******************************************************************



Run Cmd : 





