# Reproducing MRGC

## 1. Setup

```bash
git checkout repro-v1
cd MRGC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
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

## 4. Results

```bash
python collect_results.py
```

Output: `reports/report_*.txt` and `repro_results.csv`.
