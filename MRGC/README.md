# MRGC

This is the code repository for **M**anifold-constrained **R**obust **G**raph **C**ondensation (**MRGC**).

## 📂 Dataset

All datasets are provided in the `data.tar.gz` compressed package. To use them, first clone this repository and then extract the dataset using the following command:

```
for file in data/*.tar.gz; do tar -xzvf "$file" -C data/; done
```

## 🚀 Training & Testing

To reproduce all experiments from our paper, execute the corresponding commands found in the `exp.sh` script.

For example, to run an experiment on the **Cora** dataset with a condensation ratio of **1.30%** under attack budgets of **10% (features)**, **1% (structure)**, and **20% (labels)**, use the following command:

```
python main.py --name Cora --reduction_rate 0.25 --structure_budget 0.01 --feature_budget 0.1 --label_budget 0.2
```

