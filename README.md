# EchoTrace-Diagnosing-Recursive-Risks-in-LLM-Powered-Recommender-System

# EchoTrace

<p align="center">
  <img src="./assets/overview.png" width="800"/>
</p>

Official implementation of:

> **Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems under Feedback Loops**

This repository provides implementations of multiple LLM-powered recommendation baselines under iterative feedback loops.

---

# 📂 Repository Structure

```
EchoTrace/
├── LLMRec/                 # LLMRec baseline
├── ALLMRec/                # A-LLMRec baseline
├── ColdItemAug/            # ColdItemAug baseline
├── data/                   # Dataset directory (to be created)
├── requirements/
│   ├── llmrec.txt
│   ├── allmrec.txt
│   └── cold_aug.txt
└── README.md
```

---

# 📊 Datasets

Please download the datasets from the official sources and place them under:

```
EchoTrace/data/
```

| Dataset | Link | 
|----------|------|
| **Amazon Books** | https://jmcauley.ucsd.edu/data/amazon/ |
| **MovieLens-1M** | https://grouplens.org/datasets/movielens/1m/ |

---

# 🛠️ Environment Setup

Each baseline uses a separate conda environment.  
We provide individual requirement files under:

```
EchoTrace/requirements/
```

## 1️⃣ LLMRec Environment

```bash
conda create -n llmrec python=3.10
conda activate llmrec
pip install -r requirements/llmrec.txt
```

## 2️⃣ A-LLMRec Environment

```bash
conda create -n allmrec python=3.10
conda activate allmrec
pip install -r requirements/allmrec.txt
```

## 3️⃣ Cold-item Augmentation Environment

Cold-item augmentation experiments use the same environment as LLMRec.

```bash
conda activate llmrec
```
---

# 📦 Data Preparation

Complete preprocessing before running experiments.

## 🔹 LLMRec Setup

**Required files**
- `train.txt`
- `label.txt`
- `item_attribute.csv`

### Step 1: Generate interaction matrix

```bash
python data_construction.py
```

This generates:

```
train_mat.npz
```

### Step 2: Generate text and image features

Run:

```bash
data_construction.ipynb
```

We use a zero vector as `image_feat.npy`.

---

## 🔹 A-LLMRec Setup

**Required files**
- `train.txt`
- `label.txt`

### Preprocessing

```bash
EchoTrace/A-LLMRec/pre_train/sasrec/python data_preprocess_{dataset}.py
```

This generates:

```
{dataset}_text_name_dict.json.gz
```

---

## 🔹 Cold-item Augmentation Setup

**Required files**
- `train.txt`
- `label.txt`
  
---

# 🚀 Running Experiments

Run each baseline inside its directory.

---

## LLMRec

```bash
cd EchoTrace/LLMRec
python Feedback_Loop.py
```

## A-LLMRec

```bash
cd EchoTrace/ALLMRec
python Feedback_Loop.py
```

## Cold-item Augmentation

```bash
cd EchoTrace/ALLMRec
python Aug_Feedback_Loop_{Dataset}.py
```

---

# 📌 Notes

- GPU is recommended for LLM-based models.
- LLM inference may require high VRAM.
- Fix random seeds for reproducibility.
- Ensure dataset paths are correctly set before execution.

---
