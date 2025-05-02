# TrueFace - Fake vs Real Face Classifier

![Built with PyTorch](https://img.shields.io/badge/Built%20With-PyTorch-red.svg)  ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
---
The paper can be found here: [TrueFace_Final_Paper.pdf](https://github.com/user-attachments/files/19956870/COMP_560_Final_Paper.pdf)

---

## Abstract

Deepfake detection must keep pace with ever-improving generative models that
produce near-indistinguishable face forgeries. 
We propose TrueFace, a two-stage
ResNet-50 framework enhanced by exponential moving average (EMA) weight
smoothing and domain-aware augmentations. In Stage I, we pretrain at 224×224 on
real vs. GAN face pairs to capture global inconsistencies. In Stage II, we fine-tune
at 1024×1024 on a balanced mix of GAN, diffusion, and photo-real images under
realistic distortions (JPEG compression, blur, noise, and MixUp). Evaluated on
three unseen domains (StyleGAN2, DALL·E2, FFHQ subset), TrueFace achieves
up to 99.96% ROC-AUC and 99.10% accuracy. We analyze confusion matrices to
pinpoint residual weaknesses and outline future work—such as scaling diffusion
datasets and adapting TrueFace for temporal consistency in video streams, rather
than just static images.


---

## Pipeline Overview

```mermaid
flowchart LR
    Start([Start])

    Start --> PretrainingPhase([Pretraining Phase])
    PretrainingPhase --> LoadAndPrepare([Load 140K Dataset and Resize to 224x224])
    LoadAndPrepare --> Warmup([Train 5 epochs with light augmentations])
    Warmup --> StrongAugments([Switch to strong augmentations and continue training])
    StrongAugments --> TrainingLoop([Training Loop: EMA updates + Cosine LR + Checkpoint])
    TrainingLoop -->|Early stopping or Max epochs| FinetuningPhase

    FinetuningPhase([Finetuning Phase])
    FinetuningPhase --> LoadAndPrepareFinetune([Load EMA Model and High-Res Datasets 1024x1024])
    LoadAndPrepareFinetune --> AugmentAndMixUp([Apply strong augmentations + MixUp])
    AugmentAndMixUp --> FinetuneLoop([Finetuning Loop: EMA updates + Cosine LR + Checkpoint])
    FinetuneLoop -->|Early stopping or Max epochs| EvaluationPhase

    EvaluationPhase([Evaluation Phase])
    EvaluationPhase --> Testing([Evaluate on TPDNE, 140K, DALL-E datasets])
    Testing --> AggregateResults([Aggregate final metrics])
    AggregateResults --> End([End])

    %% Color styling
    style PretrainingPhase fill:#4F8EF7,stroke:#333,stroke-width:2px,color:#fff
    style LoadAndPrepare fill:#4F8EF7,stroke:#333,stroke-width:2px,color:#fff
    style Warmup fill:#4F8EF7,stroke:#333,stroke-width:2px,color:#fff
    style StrongAugments fill:#4F8EF7,stroke:#333,stroke-width:2px,color:#fff
    style TrainingLoop fill:#4F8EF7,stroke:#333,stroke-width:2px,color:#fff

    style FinetuningPhase fill:#FF851B,stroke:#333,stroke-width:2px,color:#fff
    style LoadAndPrepareFinetune fill:#FF851B,stroke:#333,stroke-width:2px,color:#fff
    style AugmentAndMixUp fill:#FF851B,stroke:#333,stroke-width:2px,color:#fff
    style FinetuneLoop fill:#FF851B,stroke:#333,stroke-width:2px,color:#fff

    style EvaluationPhase fill:#2ECC40,stroke:#333,stroke-width:2px,color:#fff
    style Testing fill:#2ECC40,stroke:#333,stroke-width:2px,color:#fff
    style AggregateResults fill:#2ECC40,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#AAAAAA,stroke:#333,stroke-width:2px,color:#fff
```

---

## Table of Contents

- [Motivation & Methodology](#motivation--methodology)
- [Data Sources](#data-sources)
- [Pipeline Improvements (Final Version)](#pipeline-improvements-final-version)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Presentation Slides](#presentation-slides)

---

## Motivation & Methodology

Modern generative models (StyleGAN, Diffusion, DALL-E 2) produce synthetic faces that are increasingly indistinguishable from real photographs, posing challenges for digital media verification.  
Fake faces can be exploited for misinformation campaigns, identity fraud, or bypassing biometric authentication systems, making robust detection critically important.

To address this, we designed a two-stage deep learning pipeline:

- **Pretraining Phase**:  
  - **Dataset**: 140K aligned real-vs-fake faces at 512×512 resolution.
  - **Training**: Images are resized to 224×224.  
  - **Warm-up**: First 5 epochs use light augmentations (resize, crop, flip) to stabilize feature extraction.
  - **Progressive Augmentation**: After warm-up, strong augmentations are introduced (RandAugment, color jitter, JPEG compression, Gaussian blur, random erasing, additive Gaussian noise).
  - **Scheduling**: Cosine annealing with early warm-up ramp.
  - **EMA**: Exponential Moving Average (decay=0.9999) of model weights is maintained.

- **Finetuning Phase**:  
  - **Datasets**: High-resolution images from TPDNE (StyleGAN) and DALL-E 2, along with held-out 140K samples.
  - **Training**: Images are resized to 1024×1024 and subjected to strong augmentations and **MixUp regularization** (α=0.2) to enhance robustness.
  - **Domain-Balanced Sampling**: Ensures even exposure to different fake generation types.
  - **EMA Continuation**: EMA weights are continually updated during finetuning.

- **Evaluation**:  
  Models are evaluated separately on TPDNE, 140K, and DALL-E test splits, reporting AUC, accuracy, precision, recall, and F1 scores.

By combining staged training, progressive difficulty, domain mixing, and EMA smoothing, our approach produces a highly generalizable detector resistant to shortcut learning and domain-specific artifacts.

---

## Data Sources

- **140K Real-vs-Fake**: 140,000 aligned face images at **512×512** resolution (50% real, 50% GAN-generated).  
  Source: [Kaggle – 140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

- **TPDNE (ThisPersonDoesNotExist)**: 20,000 StyleGAN-generated headshots at 1024×1024 resolution.  
  Source: [Kaggle – TPDNE Dataset](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist/data)

- **FFHQ**: ~70,000 high-quality real faces at 1024×1024 resolution.  
  Source: [NVIDIA FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)

- **DALL-E 2**: AI-generated headshots locally produced using OpenAI's DALL-E 2 API.

```text
src/
└── data/
    ├── 140k-real-vs-fake/
    │   ├── test/
    │   │   ├── FAKE/
    │   │   └── REAL/
    │   ├── train/
    │   │   ├── FAKE/
    │   │   └── REAL/
    │   └── valid/
    │       ├── FAKE/
    │       └── REAL/
    ├── dalle/
    │   ├── test/
    │   │   ├── fake/
    │   │   └── real/
    │   ├── train/
    │   │   ├── fake/
    │   │   └── real/
    │   └── valid/
    │       ├── fake/
    │       └── real/
    └── tpdne/
        ├── test/
        │   ├── fake/
        │   └── real/
        ├── train/
        │   ├── fake/
        │   └── real/
        └── valid/
            ├── fake/
            └── real/
```
---

## Pipeline Improvements (Final Version)

- **Pretraining warm-up phase**:  
  Light augmentations for the first 5 epochs to stabilize feature extraction.

- **Strong augmentations after warm-up**:  
  Heavy RandAugment operations, color jitter, JPEG noise, Gaussian blur, random erasing, and additive noise.

- **Finetuning with MixUp**:  
  MixUp (α=0.2) regularization improves robustness over small, high-resolution datasets.

- **Exponential Moving Average (EMA)**:  
  EMA smoothing (decay=0.9999) applied throughout training to boost generalization.

- **Domain-balanced training**:  
  Mixed sampling across TPDNE, DALL-E, and 140K datasets during finetuning.

---

## Model Architecture

We build on an ImageNet-pretrained **ResNet-50**, replacing the final fully connected layer with:
- Dropout  
- Linear output head (Real vs Fake)

The model focuses on **mid-level textures** and **spatial face structure** rather than low-level generative artifacts.

---

## Results

| Dataset   | AUC    | Accuracy | Precision | Recall | F1-Score |
|-----------|--------|----------|-----------|--------|----------|
| TPDNE     | 0.9996 | 99.10%    | 0.99      | 0.99   | 0.99     |
| 140K      | 0.9950 | 96.44%    | 0.96      | 0.97   | 0.96     |
| DALL-E 2  | 0.9995 | 97.67%    | 0.96      | 0.99   | 0.98     |
| Combined  | 0.9958 | 96.79%    | 0.97      | 0.97   | 0.97     |

> **Key takeaway**: Precision, recall, and F1-scores consistently exceed 0.96 across domains.

---

## Presentation Slides

The slides used to present this project proposal for COMP 560 are available [here](https://docs.google.com/presentation/d/192OlYnVC1KzR5nTisA6muCPcoLXFrG7LRQWAPVUIRsU/edit?usp=sharing).

(Note: These slides were created during the initial project proposal phase and reflect our first model version, which experienced overfitting issues. The current pipeline described here includes major improvements addressing those challenges.)

---
