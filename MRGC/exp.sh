# Runs everything on one GPU; pick which with e.g. `GPU=2 bash exp.sh`
# ==================== Ratio ====================
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.5 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 1.0 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.5 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 1.0 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.5 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 1.0 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.5 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 1.0 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.2

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name ogbn-arxiv --reduction_rate 0.001 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name ogbn-arxiv --reduction_rate 0.005 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name ogbn-arxiv --reduction_rate 0.01 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2


# ==================== Budget ====================
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.05 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.1 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.2 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.3 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.3
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.4

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.05 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.1 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.2 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.3 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.3
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name CiteSeer --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.4

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.005 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.2 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.3 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.3
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name dblp --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.4

CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.005 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.2 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.3 --label_budget 0.2
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.3
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py --name PubMed --reduction_rate 0.25 --structure_budget 0.001 --feature_budget 0.1 --label_budget 0.4
