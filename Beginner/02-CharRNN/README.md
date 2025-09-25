# 02 — Character-Level RNN (GRU/LSTM)
Train a char-level LM and generate text.

Quick start:
  python -m venv .venv && source .venv/bin/activate
  pip install -U pip -r ../../requirements.txt
  # Install PyTorch per your OS/GPU: https://pytorch.org/get-started/locally/
  python char_rnn.py --data ./data/input.txt --epochs 5 --seq-len 256 --batch-size 64 --embed 256 --hidden 512 --lr 3e-3 --temp 0.8
