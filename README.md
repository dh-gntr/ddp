# DDP
# PostHoc-UQ: Dirichlet Meta-Models for Post-Hoc Uncertainty Quantification

A research framework for post-hoc uncertainty quantification (UQ) using Dirichlet-based meta-models trained on frozen classifiers. The repository contains implementations of multiple uncertainty estimation strategies, ablation studies, and evaluation protocols across medical and natural image datasets.

---

## Overview

Deep neural networks often produce overconfident predictions, particularly under distribution shift or on difficult samples. This repository investigates post-hoc uncertainty quantification methods that operate on top of a pre-trained frozen classifier without modifying the base network.

The core idea is to train lightweight meta-models that consume representations extracted from one or more layers of a frozen backbone and predict uncertainty-aware Dirichlet distributions.

The framework includes:

* Dirichlet Evidential Meta-Model
* Multi-Layer Dirichlet Meta-Model
* HydraNet
* Multi-Head Meta-Model
* MetaConsensus
* HydraMagic
* Base-Model Concatenation Variants
* Weight Entropy Maximization (Max-WEnt)
* Alternative uncertainty losses and calibration objectives
* Extensive evaluation metrics and ablations

---

## Repository Structure

```text
.
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── datasets/
│   ├── bach/
│   ├── ham10000/
│   ├── breakhis/
│   └── ...
│
├── models/
│   ├── backbone/
│   ├── metamodel/
│   ├── hydranet/
│   ├── hydramagic/
│   ├── metaconsensus/
│   └── multihead/
│
├── losses/
│   ├── belief_matching.py
│   ├── evidential_loss.py
│   ├── entropy_regularization.py
│   └── calibration_losses.py
│
├── metrics/
│   ├── auroc.py
│   ├── calibration.py
│   ├── uncertainty.py
│   └── ood.py
│
├── experiments/
│   ├── baselines/
│   ├── ablations/
│   └── comparisons/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
└── README.md
```

---

# Methodology

## 1. Base Classifier

A backbone classifier is trained conventionally using cross-entropy.

Examples:

* ResNet-18
* ResNet-50
* DenseNet-121
* EfficientNet
* Vision Transformers

The backbone remains frozen during meta-model training.

---

## 2. Dirichlet Meta-Model

The meta-model receives intermediate features extracted from selected backbone layers.

```text
Input Image
      │
      ▼
 Frozen Backbone
      │
 ┌────┼────┐
 │    │    │
Layer1 Layer2 Layer3
 │     │     │
 └─────┴─────┘
       │
       ▼
 Dirichlet Meta-Model
       │
       ▼
 α = [α1,...,αK]
```

The model predicts Dirichlet concentration parameters:

```math
p \sim Dir(\alpha)
```

from which both predictions and uncertainty estimates are derived.

---

## 3. Multi-Layer Feature Extraction

Instead of relying solely on final-layer embeddings, features are extracted from multiple stages of the network.

Benefits:

* Low-level texture information
* Mid-level structural information
* High-level semantic information

This improves uncertainty estimation under domain shift and difficult examples.

---

# Implemented Methods

## Dirichlet Meta-Model

Baseline evidential meta-model.

Features:

* Frozen backbone
* Single-head uncertainty estimator
* Dirichlet evidence prediction

---

## HydraNet

HydraNet introduces multiple uncertainty estimation heads.

```text
Shared Encoder
       │
 ┌─────┼─────┐
 │     │     │
Head1 Head2 Head3
```

Motivation:

* Capture diverse uncertainty hypotheses
* Improve epistemic uncertainty estimation
* Increase robustness

---

## Multi-Head Meta-Model

Extension of the basic meta-model with independent prediction heads.

Each head predicts:

```math
Dir(\alpha_h)
```

Outputs can be:

* Averaged
* Weighted
* Consensus aggregated

---

## MetaConsensus

MetaConsensus combines predictions from multiple uncertainty heads using a learned consensus mechanism.

Goals:

* Reduce noisy uncertainty estimates
* Improve calibration
* Improve OOD detection

---

## HydraMagic

HydraMagic combines:

* Multi-layer feature extraction
* Multi-head uncertainty prediction
* Entropy regularization
* Consensus aggregation

to generate richer uncertainty representations.

---

## Base-Model Concatenation

Several experiments investigate concatenating:

* Backbone logits
* Backbone probabilities
* Intermediate representations
* Confidence scores

with meta-model features.

Examples:

```text
Meta Features
      │
      ▼
 Concatenate
      ▲
 Base Logits
```

This studies whether classifier outputs provide complementary uncertainty information.

---

# Weight Entropy Maximization (Max-WEnt)

To encourage diversity among learned scaling weights:

```math
H(w) = -\sum_i w_i \log w_i
```

is maximized.

Motivation:

* Prevent weight collapse
* Promote diverse feature utilization
* Improve epistemic uncertainty estimation

---

# Loss Functions

## Evidential Deep Learning Loss

Combines:

* Data fitting loss
* Variance regularization
* KL regularization

to learn Dirichlet evidence.

---

## Belief Matching Loss

Optimizes:

```math
\mathbb{E}_{p \sim Dir(\alpha)}
[\log p_y]
```

while regularizing towards a prior Dirichlet distribution.

Benefits:

* Probabilistically grounded
* Better calibrated uncertainty
* Improved epistemic estimation

---

## Entropy Regularization

Encourages diversity across uncertainty heads and feature scales.

---

# Evaluation Metrics

## Classification Metrics

* Accuracy
* Balanced Accuracy
* F1 Score
* Precision
* Recall

---

## Calibration Metrics

### Expected Calibration Error (ECE)

Measures calibration gap between confidence and accuracy.

### Maximum Calibration Error (MCE)

Worst-case calibration error across bins.

### Adaptive ECE

Adaptive binning-based calibration measure.

### Brier Score

Measures probabilistic prediction quality.

### Negative Log Likelihood (NLL)

Evaluates probabilistic correctness.

---

## Uncertainty Metrics

### Predictive Entropy

```math
H(p) = -\sum_i p_i \log p_i
```

---

### Mutual Information

Measures epistemic uncertainty.

---

### Dirichlet Total Evidence

```math
S = \sum_i \alpha_i
```

Lower evidence implies higher uncertainty.

---

### Vacuity

Measures lack of evidence.

---

### Dissonance

Measures conflicting evidence.

---

## OOD Detection Metrics

* AUROC
* AUPR-In
* AUPR-Out
* FPR95

---

# Experimental Studies

## Backbone Comparison

Evaluates uncertainty performance across:

* ResNet variants
* DenseNet variants
* EfficientNet variants
* Vision Transformers

---

## Feature Layer Ablations

Studies:

* Early layers
* Mid layers
* Late layers
* Multi-layer combinations

---

## Head Count Ablations

Investigates:

* 1 Head
* 2 Heads
* 4 Heads
* 8 Heads

for uncertainty estimation.

---

## Entropy Regularization Ablations

Evaluates impact of:

* No entropy regularization
* Fixed entropy coefficient
* Adaptive entropy coefficient

---

## Consensus Strategy Ablations

Comparison between:

* Mean aggregation
* Weighted aggregation
* Learned consensus
* HydraMagic consensus

---

# Datasets

Medical Imaging:

* BACH
* HAM10000
* BreakHIS
* DIV2K

Natural Images:

* Fashion-MNIST
* SVHN
* CIFAR
* ImageNet-C

---

# Running Experiments

Train a meta-model:

```bash
python train.py \
    --config configs/experiments/metamodel.yaml
```

Evaluate:

```bash
python evaluate.py \
    --checkpoint checkpoints/model.ckpt
```

Generate uncertainty metrics:

```bash
python scripts/compute_metrics.py
```

---

# Citation

If you use this repository, please cite the associated publications and reports describing the Dirichlet meta-model framework and its extensions.
