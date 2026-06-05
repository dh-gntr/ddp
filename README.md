# DDP
# PostHoc-UQ: Dirichlet Meta-Models for Post-Hoc Uncertainty Quantification

A research framework for post-hoc uncertainty quantification (UQ) using Dirichlet-based meta-models trained on frozen classifiers. The repository contains implementations of multiple uncertainty estimation strategies, ablation studies, and evaluation protocols across medical and natural image datasets.

---

## Overview

Deep neural networks often produce overconfident predictions, particularly under distribution shift or on difficult samples. This repository investigates post-hoc uncertainty quantification methods that operate on top of a pre-trained frozen classifier without modifying the base network.

The core idea is to train lightweight meta-models that consume representations extracted from one or more layers of a frozen backbone and predict uncertainty-aware Dirichlet distributions. 

The various frameworks include:

* HydraNet
* Dirichlet Evidential Meta-Model
* Multi-Head Meta-Model
* MetaConsensus
* HydraMagic
* Base-Model Concatenation Variants
* Weight Entropy Maximization (Max-WEnt)
* Alternative uncertainty losses and calibration objectives
* Extensive evaluation metrics and ablations on noise and compression and class detection performance

---

## Repository Structure

```text
.

├── data/
│   ├── bach/
│   ├── ham10000/
│   ├── breakhis/
|   ├── cifar100/
|   ├── cifar10/
|   ├── MNIST/
|   ├── FashionMNIST
|   ├── SVHN
│   └── ...
│
├── models/
│   ├── resnet18meta.py
│   ├── vgg.py
│   ├── init.py
│   ├── lenet.py
│
├── checkpoint/
├── results/
├── train_meta_model_combine.py
├── train_base_model.py
├── losses.py
├── preproc.py
├── metrics.py
├── baseline.ipynb
├── hydranet.ipynb
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

## Dirichlet Meta-Model

Baseline evidential meta-model.

Features:

* Frozen backbone
* Single-head uncertainty estimator
* Dirichlet evidence prediction

---


## Multi-Head Meta-Model

Extension of the basic meta-model with independent feature heads, inspired from the Hydranet model. One of the four features is randomly chosen and applied projector on.

Each projector-classifier predicts:

```math
Dir(\alpha_h)
```

Outputs can be:

* Averaged
* Weighted
* Consensus aggregated
during inference
---

## MetaConsensus

MetaConsensus samples two of the four features and applies a KL divergence loss between their predicted distributions to reduce disagreement for ID samples.

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

It provides multiple projectors per block and multiple feature choices chosen randomly during training, thus maximising diversity richness during training.

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



# Evaluation Metrics

## Classification Metrics

* Accuracy


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



## OOD Detection Metrics

* AUROC


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

* Early layers (0123)
* Mid layers (0246)
* Late layers (46810)
* Multi-layer combinations (04812) (371115)

---

## Head Count Ablations

Investigates:

* 2 Head
* 3 Heads
* 4 Heads
* 5 Heads

for uncertainty estimation.

---


# Datasets



Natural Images:

* Fashion-MNIST
* SVHN
* CIFAR
* ImageNet-C

---

# Running Experiments

Train a meta-model:

```bash
python train_meta_model_combine.py \
    
```

Train base model:

```bash
python train_base_model.py \
    --checkpoint checkpoint/model.ckpt
```



---

# Citation

If you use this repository, please cite the associated publications and reports describing the Dirichlet meta-model framework and its extensions.
