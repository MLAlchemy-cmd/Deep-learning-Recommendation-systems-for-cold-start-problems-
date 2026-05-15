# Multimodal Sequential Recommendation for Amazon Beauty

A time-aware sequential recommendation system that fuses text, image, and interaction signals to improve cold-start product recommendation on the Amazon Beauty dataset.

---

## Overview

This project implements **MultiModal-TiSASRec**, a sequential recommender that extends Time-aware Self-Attention (TiSASRec) with multimodal item representations. Items are encoded using:

- **Text embeddings** — product title, description, features, and categories via `all-MiniLM-L6-v2`
- **Image embeddings** — product images via ResNet-50
- **ID embeddings** — learned item identity vectors

A gated cross-attention fusion module combines these modalities before passing sequences through a causal Transformer. The system is evaluated with stratified metrics across **cold**, **few-shot**, and **warm** item tiers to explicitly measure cold-start performance.

---

## Repository Structure

```
├── Beauty_Amazon.ipynb               # Data pipeline: download, clean, subsample, embed
├── user_interactions_training__5_.ipynb  # Model definition and training
├── ablation_study__1_.ipynb          # Ablation across 4 modality variants
└── README.md
```

---

## Pipeline

### 1. `Beauty_Amazon.ipynb` — Data Preparation & Embedding

Runs once to produce all processed files saved to your `SAVE_DIR`.

| Step | Description | Output |
|------|-------------|--------|
| Download | Stream Amazon Reviews 2023 (All Beauty) + metadata | `amazon_beauty_2023_full.csv` |
| Clean | Drop nulls, remove unused columns | `amazon_beauty_cleaned.csv` |
| Subsample | Stratified 2M-row sample (cold/few-shot/warm balance) | `amazon_beauty_2M.csv` |
| Text embed | `all-MiniLM-L6-v2` on title + description + features + categories | `text_embeddings.npy` |
| Image embed | ResNet-50 on product images (parallel download + batch inference) | `image_embeddings.npy` |

> **Note**: Set `SAVE_DIR` at the top of the notebook to your own Google Drive path before running.

### 2. `user_interactions_training__5_.ipynb` — Model & Training

Loads the preprocessed files and trains the full MultiModal-TiSASRec model.

- Builds user/item index mappings and leave-one-out splits (train / val / test)
- Trains with BPR loss and cosine-annealing LR schedule for 30 epochs
- Evaluates with sampled 100-way ranking (1 positive + 99 negatives) stratified by cold-start tier

### 3. `ablation_study__1_.ipynb` — Ablation Study

Trains and evaluates 4 modality variants on a stratified 20% subsample to isolate each component's contribution:

| Variant | Modalities used |
|---------|----------------|
| ID only | Item ID embeddings only |
| Text only | Text + ID |
| Image only | Image + ID |
| Text + Image | Full multimodal fusion (cross-attention gated) |

---

## Model Architecture

```
Item sequence  ──►  [Text proj]  ─┐
                    [Image proj] ─┤── Gated cross-attention fusion ──► fused item repr
                    [ID embed]   ─┘
                                        + Time-interval encoding
                                        + Positional encoding
                                              │
                                    Causal Transformer (2 layers, 8 heads)
                                              │
                                        Last-position repr
                                              │
                                    Score against K candidates
```

**Key hyperparameters**: `shared_dim=128`, `max_len=50`, `dropout=0.2`, `batch_size=256`, `lr=1e-3`

---

## Evaluation Protocol

Metrics follow the standard **sampled evaluation** protocol:

- **Recall@K** and **NDCG@K** at K ∈ {5, 10, 20}
- 100-way ranking per user (1 positive + 99 random negatives not seen by the user)
- Results stratified by item training frequency:
  - **Cold**: 0 training interactions
  - **Few-shot**: 1–4 training interactions
  - **Warm**: ≥ 5 training interactions

---

## Setup

### Requirements

```bash
pip install torch torchvision sentence-transformers faiss-cpu datasets pandas numpy pillow requests tqdm
```

All notebooks are designed to run on **Google Colab** with a GPU runtime (T4 or better recommended for the embedding and training steps).

### Data

Data is loaded automatically from [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) via the `datasets` library. No manual download needed.

Large files (`*.npy`, `*.csv`) are excluded from this repository. Set `SAVE_DIR` in `Beauty_Amazon.ipynb` to a Google Drive path with at least **15 GB free** before running.

### Running order

```
1. Beauty_Amazon.ipynb              ← run fully first (generates all data + embeddings)
2. user_interactions_training__5_.ipynb  ← train the full model
3. ablation_study__1_.ipynb         ← run ablation variants
```

---

## References

- Wang et al. (2020). [Time Interval Aware Self-Attention for Sequential Recommendation](https://dl.acm.org/doi/10.1145/3336191.3371786). WSDM.
- McAuley Lab. [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). HuggingFace Datasets.
- Reimers & Gurevych (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084). EMNLP.
- He et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.
