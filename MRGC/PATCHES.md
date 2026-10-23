# Patches vs. upstream MRGC @ 95976ac

Makes the code run:
- `dataset.py`: build the adjacency with the true node count (shape mismatch when trailing nodes are isolated).
- `dataset.py`, `utils.py`: import `NeighborSampler` from `torch_geometric.loader` (moved in PyG 2.x), with fallback.
- `exp.sh`: fix `CiteSeer--reduction_rate` typo; pin all runs to one GPU via `$GPU`.
- `main.py`: ASCII-only log output (`+/-` instead of the plus-minus sign).

Makes it report what the paper reports (Sec. 4.1, 5 repeats, mean +/- std):
- `main.py`: per-run seeding, `--runs`/`--seed` args, final mean +/- std line.
- `train.py`: re-enabled the every-100-epoch eval logging; return the test accuracy at the best-validation checkpoint.

Adds clean and single-attack runs (upstream parses the attack flags but never uses them):
- `dataset.py`: `--feature_attack/--structure_attack/--label_attack 0` load the clean component from the public
  source (Planetoid, CitationFull DBLP, ogbn-arxiv), since `clean.pth` only holds splits and sizes. A check
  refuses to run if the clean data doesn't match the attacked copy's node order.
- `main.py`: runs with any attack off log to `result/ablation/` with an `-f?s?l?` suffix, so they never mix
  into the paper comparison. `ablation.sh` runs the Cora set.

Changes the method (disclose when reporting):
- `train.py`: skip any optimizer step whose loss or gradients are non-finite, and log how many were skipped.
  Without it, every reduction_rate 1.0 run on Cora, CiteSeer and dblp died: a NaN reached `feat_syn` and the
  `pca_svd` except-branch called `exit()`. Likely source is the eigh/SVD backward in the curvature loss
  (`tools.py` `compute_normals_batch` / `compute_curvature_batch`) or the 0/0 Ricci normalisation in `train.py`; not confirmed.

Not changed: hyperparameters (`config/*.json` as shipped; they differ from the paper's Table 4 in places, and Table 4 has visible typos).

## Expected variation

Seeds fix the data sampling and initialisation, but CUDA sparse/scatter ops are non-deterministic, so reruns
(especially on a different GPU or driver) will not match bit-for-bit. Expect differences within roughly one std.
