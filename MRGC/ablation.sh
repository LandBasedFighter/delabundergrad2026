# Clean baseline and one-attack-at-a-time runs on Cora (experiment 2), same budgets and config as the main table.
# Flags are feature/structure/label; 1 = attacked copy, 0 = clean. Logs go to result/ablation/.
# The all-attacks run (1/1/1) is the main-table run from ratio.sh. Pick the GPU with e.g. `GPU=2 bash ablation.sh`
ARGS="--name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2"
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py $ARGS --feature_attack 0 --structure_attack 0 --label_attack 0
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py $ARGS --feature_attack 0 --structure_attack 1 --label_attack 0
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py $ARGS --feature_attack 1 --structure_attack 0 --label_attack 0
CUDA_VISIBLE_DEVICES=${GPU:-0} python main.py $ARGS --feature_attack 0 --structure_attack 0 --label_attack 1
