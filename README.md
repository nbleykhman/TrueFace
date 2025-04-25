# Fake vs Real Face Classifier

This project implements a two-stage deep-learning pipeline to distinguish AI-generated (“fake”) headshots from authentic (“real”) photographs using a ResNet-50 backbone.

## Motivation & Methodology  
Modern generative models (StyleGAN, diffusion networks) produce highly realistic faces that are increasingly hard to detect. We leverage transfer learning: first we **pre-train** on a large low-resolution binary dataset (140 K real vs. GAN fakes) to learn general facial features, then **fine-tune** jointly on high-resolution TPDNE diffusion images and held-out 140 K splits. Strong augmentations (random crops, blur, color jitter, erasing) and weight decay enforce robustness against trivial artifact cues.

## Data Collected  
- **FFHQ**: ∼70 000 high‑quality, aligned real human face images at 1024×1024 used to augment the real class headshots.  
  Source: [NVlabs FFHQ dataset](https://github.com/NVlabs/ffhq-dataset?tab=readme-ov-file)  

- **140K real-vs-fake**: 140 000 images at 256×256 (50% real, 50% GAN-generated).  
  Source: [Kaggle – 140k real and fake faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

- **TPDNE**: 20 000 headshots at 1024×1024 Style-GAN generated images.  
  Source: [Kaggle – ThisPersonDoesNotExist](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist/data) (sourced from https://thispersondoesnotexist.com).  

## Model Architecture Derivation  
We build on an ImageNet-pretrained ResNet-50, replacing its final layer with a dropout + linear head for two classes. Pre-training warms up broad mid-level features; fine-tuning adapts to domain-specific diffusion and GAN artifacts under a low learning rate and balanced sampling.

## Abstract  
This work presents a robust two‑stage pipeline for detecting AI‑generated versus authentic human headshots, leveraging a ResNet‑50 backbone pretrained on ImageNet. In the first “pretraining” phase, we warm up the network on a large low‑resolution corpus (140 K real-vs‑fake images) using aggressive augmentations—including random resized crops, color jitter, Gaussian blur, and erasing—to force the model to learn mid‑level facial textures rather than trivial artifacts. The second “fine‑tuning” phase combines high‑resolution TPDNE diffusion‑generated portraits with held‑out 140 K samples under a balanced sampling scheme and stronger noise augmentations. We adopt weight decay (1e‑3), a two‑epoch warm‑up scheduler, and a ReduceLROnPlateau policy to stabilize convergence.  

Extensive evaluation demonstrates near‑perfect separation across test splits: on the TPDNE set, the model achieved an AUC of 1.0000 and 99.87 % accuracy; on the 140K set, an AUC of 0.9999 and 99.14 % accuracy; and on the combined split, an AUC of 0.9999 and 99.23 % accuracy. Precision, recall, and F1‑scores consistently exceed 0.98 on both classes, confirming balanced performance. These results indicate the network’s capacity to generalize to diverse synthetic face generation methods (StyleGAN and diffusion) and resist shortcut learning.  

Our methodology demonstrates that judicious use of transfer learning, domain‑balanced sampling, and tailored augmentations can yield a highly accurate, generalizable fake‑face detector suitable for deployment in forensic and media‑authenticity applications.

## Presentation Slides

The slides used to present this project for COMP 560 are located [here](https://docs.google.com/presentation/d/192OlYnVC1KzR5nTisA6muCPcoLXFrG7LRQWAPVUIRsU/edit?usp=sharing).
