.PHONY: setup lint run-char-rnn
setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt
run-char-rnn:
\tpython Beginner/02-CharRNN/char_rnn.py --data Beginner/02-CharRNN/data/input.txt --epochs 3 --seq-len 128 --batch-size 64 --embed 128 --hidden 256 --lr 3e-3 --temp 0.8
