A hands-on curriculum to learn and build with Large Language Models — from **Beginner** to **Professional**.

## Levels
- **Beginner** — Deep learning & NLP fundamentals with code (RNNs, embeddings, seq2seq, mini-Transformer).
- **Intermediate** — Pretrained models & fine-tuning (GPT-2, BERT), evaluation, prompt engineering.
- **Advanced** — Efficient finetuning (LoRA/QLoRA), RAG systems, latency & cost optimization.
- **Professional** — Productionization: API serving, monitoring, safety, CI/CD, containers, scaling.

> Start in `Beginner/02-CharRNN` to train your first character-level language model.

## Environment
- Python >= 3.10
- GPU recommended, but CPU works for tiny experiments.

### Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Install PyTorch for your platform: https://pytorch.org/get-started/locally/
