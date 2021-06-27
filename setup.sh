sudo apt install python3-venv python3-tk && \
python3 -m venv DSLR && \
source ./DSLR/bin/activate && \
pip install -r requirements.txt && \
echo -e "\n[+] SETUP COMPLETE\n" && \
python logreg_train.py -h
