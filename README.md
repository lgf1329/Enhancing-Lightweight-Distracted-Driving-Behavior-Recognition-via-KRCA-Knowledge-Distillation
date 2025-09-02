# Enhancing-Lightweight-Distracted-Driving-Behavior-Recognition-via-KRCA-Knowledge-Distillation
🚀 ​​**KRCA-Distilled Lightweight Model for Real-time Distracted Driving Recognition​​**

​​**Revolutionizing edge deployment with multi-knowledge distillation and light robustness optimization​​**

This repository presents a groundbreaking lightweight framework for driver distraction recognition, achieving ​​99.83% accuracy​​ with only ​​187.43K parameters​​. Our novel KRCA (Knowledge Relation Comparison with Attention) distillation method synergistically combines four complementary knowledge types for the first time in driving behavior recognition, overcoming traditional distillation's limitations in handling light variations and behavioral similarities.

![FIG2](https://github.com/user-attachments/assets/a57d5f69-e033-46ef-9a46-9fc48697940b)
🔬 ​​**Core Innovation: Multi-Knowledge Distillation Framework​​**

The KRCA distillation framework simultaneously optimizes:

1.**​​Hard labels​​** (L_cls) for final prediction alignment

2.​​**Soft labels​​** (L_TC + L_NC) with target/non-target class decomposition

3.**Sample relations​​** (L_RC) preserving cross-sample dependencies

4.​​**Inter-layer attention​​** (L_SC + L_AT) transferring spatial focus patterns

This integrated approach solves the critical problems of ​​behavioral preference bias​​ and ​​feature mutual exclusion​​ observed in single-knowledge distillation methods, particularly enhancing recognition of hand-occluded actions  in low-light conditions.


🌟 ​​**Key Performance Highlights​**

Modified SqueezeNet_LW

•Parameters reduced from 740.55K → 187.43K (**74.7%↓**)

•FLOPs reduced from 750.32M → 254.3M (**66.1%↓**)

•Inference speed: 120.88 FPS (**2.5× faster than baseline**)
📁 ​​**Datasets​**
AUC (American University in Cairo) Dataset​ and SFD3 (State Farm Distracted Driver Detection)​
## Installation
a. Clone this repository.
```shell
git clone https://github.com/lgf1329/Enhancing-Lightweight-Distracted-Driving-Behavior-Recognition-via-KRCA-Knowledge-Distillation.git
```

b. Install the dependent libraries as follows:

```
pip install -r requirements.txt 
```

## Data Preparation

You can put the downloaded data here:
```
data
├── SFD3
|   |── train
|   |   |── drinking
|   |   |── makeup
|   |   |── normal driving
|   |   |── operating the interactive interface
|   |   |── reaching behind
|   |   |── talking on the phone - left
|   |   |── talking on the phone - right
|   |   |── talking to passenger
|   |   |── texting - left
|   |   |── texting - right
|   |── test
|   |   |── drinking
|   |   |── makeup
|   |   |── normal driving
|   |   |── operating the interactive interface
|   |   |── reaching behind
|   |   |── talking on the phone - left
|   |   |── talking on the phone - right
|   |   |── talking to passenger
|   |   |── texting - left
|   |   |── texting - right
|   |── train_labels.csv
|   |── test_labels.csv
├── AUC
|   |── train
|   |   |── drinking
|   |   |── makeup
|   |   |── normal driving
|   |   |── operating the interactive interface
|   |   |── reaching behind
|   |   |── talking on the phone - left
|   |   |── talking on the phone - right
|   |   |── talking to passenger
|   |   |── texting - left
|   |   |── texting - right
|   |── test
|   |   |── drinking
|   |   |── makeup
|   |   |── normal driving
|   |   |── operating the interactive interface
|   |   |── reaching behind
|   |   |── talking on the phone - left
|   |   |── talking on the phone - right
|   |   |── talking to passenger
|   |   |── texting - left
|   |   |── texting - right
|   |── train_labels.csv
|   |── test_labels.csv
```

## Training

a. First, run Convnet_try.py to train the teacher network.

b. Second, run SQ_LW_try.py to train the student network.

c.Next, run convnext_KRAC_SQ_LW.py to perform KRAC knowledge distillation.

d.Finally, run SqueezeNet_LW_result.py to analyze the results of the SqueezeNet_LW model after KRAC knowledge distillation.


## Citation 
If you find this project useful in your research, please consider citing:

```
@article{agentdriver,
  title={Enhancing Lightweight Distracted Driving Behavior Recognition via KRCA Knowledge Distillation},
  author={Guofeng, Luo and Baicang, Guo and Lisheng, Jin and Ye, Zhang and Xincheng, Liu and Jie, Liu and Wenjun, Sh and Hang, Yao},
  year={2025}
}
```
