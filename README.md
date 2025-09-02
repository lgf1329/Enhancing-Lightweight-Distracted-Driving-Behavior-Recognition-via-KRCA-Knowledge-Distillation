# Enhancing-Lightweight-Distracted-Driving-Behavior-Recognition-via-KRCA-Knowledge-Distillation
🚀 ​​**KRCA-Distilled Lightweight Model for Real-time Distracted Driving Recognition​​**

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

You can use the following structure to prepare the data.
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

```
python3 Convnet_try.py
```

b. Second, run SQ_LW_try.py to train the student network.
```
python3 SQ_LW_try.py
```
c.Next, run convnext_KRAC_SQ_LW.py to perform KRAC knowledge distillation.
```
python3 convnext_KRAC_SQ_LW.py
```
d.Finally, run SqueezeNet_LW_result.py to analyze the results of the SqueezeNet_LW model after KRAC knowledge distillation.The AUC and SF folders contain SqueezeNet_LW model parameters before and after distillation using various methods. These parameters can be directly modified in SqueezeNet_LW_result.py to alter the model's output results. SqueezeNet.pth holds the trained parameters of the SqueezeNet_LW model, while SqueezeNet_HL.pth contains parameters distilled using Hard labels. SqueezeNet_SL.pth contains parameters after distillation using soft labels, SqueezeNet_SR.pth after distillation using sample relations, SqueezeNet_IA.pth after distillation using inter-layer attention, and SqueezeNet_HL_SL.pth after distillation using both soft labels and sample relations. The remaining files follow a similar naming convention.
```
python3 SqueezeNet_LW_result.py
```

## Citation 
If you find this project useful in your research, please consider citing:

```
@article{agentdriver,
  title={Enhancing Lightweight Distracted Driving Behavior Recognition via KRCA Knowledge Distillation},
  author={Guofeng, Luo and Baicang, Guo and Lisheng, Jin and Ye, Zhang and Xincheng, Liu and Jie, Liu and Wenjun, Sh and Hang, Yao},
  year={2025}
}
```
