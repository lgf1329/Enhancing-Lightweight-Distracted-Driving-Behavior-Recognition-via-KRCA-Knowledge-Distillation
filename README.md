# Enhancing-Lightweight-Distracted-Driving-Behavior-Recognition-via-KRCA-Knowledge-Distillation
🚀 ​​**KRCA-Distilled Lightweight Model for Real-time Distracted Driving Recognition​​**

​​**Revolutionizing edge deployment with multi-knowledge distillation and light robustness optimization​​**

This repository presents a groundbreaking lightweight framework for driver distraction recognition, achieving ​​99.83% accuracy​​ with only ​​187.43K parameters​​. Our novel KRCA (Knowledge Relation Comparison with Attention) distillation method synergistically combines four complementary knowledge types for the first time in driving behavior recognition, overcoming traditional distillation's limitations in handling light variations and behavioral similarities.

![FIG1](https://github.com/user-attachments/assets/7423dfef-c3dc-4f2f-b7e7-65a6da380ef4)

🔬 ​​**Core Innovation: Multi-Knowledge Distillation Framework​​**

The KRCA distillation framework simultaneously optimizes:

1.**​​Hard labels​​** (L_cls) for final prediction alignment

2.​​**Soft labels​​** (L_TC + L_NC) with target/non-target class decomposition

3.**Sample relations​​** (L_RC) preserving cross-sample dependencies

4.​​**Inter-layer attention​​** (L_SC + L_AT) transferring spatial focus patterns

This integrated approach solves the critical problems of ​​behavioral preference bias​​ and ​​feature mutual exclusion​​ observed in single-knowledge distillation methods, particularly enhancing recognition of hand-occluded actions  in low-light conditions.

![FIG2](https://github.com/user-attachments/assets/a57d5f69-e033-46ef-9a46-9fc48697940b)
🌟 ​​**Key Performance Highlights​**

Modified SqueezeNet_LW

•Parameters reduced from 740.55K → 187.43K (**74.7%↓**)

•FLOPs reduced from 750.32M → 254.3M (**66.1%↓**)

•Inference speed: 120.88 FPS (**2.5× faster than baseline**)
