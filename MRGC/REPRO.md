# Reproducing MRGC

## 1. Setup

```bash
cd MRGC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

```bash
mkdir -p data
for d in Cora CiteSeer PubMed dblp ogbn-arxiv; do
  curl -L -o data/$d.tar.gz https://github.com/RingBDStack/MRGC/raw/95976ac/data/$d.tar.gz
done
(cd data && sha256sum -c ../data_checksums.sha256)
for f in data/*.tar.gz; do tar -xzf "$f" -C data/; done
```

## 3. Run

```bash
GPU=0 nohup bash ratio.sh > ratio_log.txt 2>&1 &
```

Clean baseline and single-attack runs (Cora, logs in `result/ablation/`):

```bash
GPU=0 nohup bash ablation.sh > ablation_log.txt 2>&1 &
```

Turning an attack off (`--feature_attack 0`, `--structure_attack 0`, `--label_attack 0`) swaps in the clean
component from the public source (downloaded to `data/clean_source/` on first use). Each run prints how much of
the clean data matches the attacked copy and stops if the two don't line up.

## 4. Results

```bash
python collect_results.py
```

Output: `reports/report_*.txt` and `repro_results.csv`.
