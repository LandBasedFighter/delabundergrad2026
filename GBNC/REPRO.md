# Reproducing GBNC

## 1. Setup

```bash
cd GBNC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Auto-downloads into `dataset/` on first run via `torch_geometric`.

## 3. Run

```bash
python main.py --dataset cora --models GCN
```

## 4. Results

Prints per-run accuracy to stdout, ending with:

```
ave_acc: 0.XXXX +/- 0.XXXX
```
