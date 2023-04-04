# Graph-LTVU
This repository contains an open-source codebase for graph-based long-term video understanding (Graph-LTVU). It is designed to serve as a spatial-temporal graph learning framework for multiple video understanding tasks. In the current version, it supports training and evaluating one of the state-of-the-art models, [SPELL](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950367.pdf), for the tasks of active speaker detection and action localization.

In the near future, we will release more advanced graph-based approaches for other tasks, including action segmentation and the [Ego4D challenge](https://ego4d-data.org/workshops/eccv22).

![](docs/images/graphltvu_teaser.jpg?raw=true)

## Use Cases and Performance
|  Model  |         Dataset         |            Task           |     validation mAP (%)     |
|:--------|:-----------------------:|:-------------------------:|:--------------------------:|
|  SPELL  |  AVA-ActiveSpeaker v1.0 |  Active Speaker Detection |   **94.2** (up from 88.0)  |
|  SPELL+ |  AVA-ActiveSpeaker v1.0 |  Action Speaker Detection |   **94.9** (up from 89.3)  |
|  SPELL  |  AVA-Actions v2.2       |  Action Localization      |   **36.8** (up from 29.4)  |
> Numbers in parentheses indicate the mAP scores without using the suggested graph learning method.

## Requirements
Preliminary requirements:
- Python>=3.7
- CUDA 11.3

Run the following command if you have CUDA 11.3:
```
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyYAML, pandas, and [PyG](https://www.pyg.org)>=2.0.3 with CUDA>=11.1

## Installation
After confirming the above requirements are met, run the following commands:
```
https://github.com/IntelLabs/Graph-LTVU
cd Graph-LTVU
pip3 install -e .
```

## Data Preparation
### Annotations
1) Download the annotations from the official site:
```
DATA_DIR="data/annotations"

wget https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2 -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_v2.2.csv.zip -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.2_for_activitynet_2019.pbtxt -P ${DATA_DIR}

tar -xf ${DATA_DIR}/ava_activespeaker_val_v1.0.tar.bz2 -C ${DATA_DIR}
unzip ${DATA_DIR}/ava_val_v2.2.csv.zip -d ${DATA_DIR}
mv ${DATA_DIR}/research/action_recognition/ava/website/www/download/ava_val_v2.2.csv ${DATA_DIR}
```

2) Preprocess the annotations:
```
python data/annotations/merge_ava_activespeaker.py
```

### Features
Download the features from [Google Drive](https://drive.google.com/drive/folders/1bX0cTHYLcBDc9ArmWps17F55goj3hgek?usp=share_link) and unzip under `data/features`.
> We used the features from the thirdparty repositories. RESNET18-TSM-AUG and RESNET50-TSM-AUG are directly from [SPELL](https://github.com/SRA2/SPELL). SLOWFAST-64x2-R101 is obtained from using [SlowFast](https://github.com/facebookresearch/SlowFast).

### Directory Structure
The data directories should look like as follows:
```
|-- data
    |-- annotations
        |-- ava_activespeaker_val_v1.0.csv
        |-- ava_val_v2.2.csv
        |-- ava_action_list_v2.2_for_activitynet_2019.pbtxt
    |-- features
        |-- RESNET18-TSM-AUG
            |-- train
            |-- val
        |-- RESNET50-TSM-AUG
            |-- train
            |-- val
        |-- SLOWFAST-64x2-R101
            |-- train
            |-- val
```

## Getting Started
We can perform the experiments on active speaker detection with the default configuration by following the three steps below.

### Step 1: Graph Generation
Run the following command to generate spatial-temporal graphs from the features:
```
python data/generate_graph.py --features RESNET18-TSM-AUG --ec_mode csi --time_span 90 --tau 0.9
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video, which spans about 90 seconds (specified by `--time_span`).

### Step 2: Training
Next, run the training script by passing the default configuration file:
```
python tools/train_context_reasoning.py --cfg configs/active-speaker-detection/ava_active-speaker/SPELL_default.yaml
```
The results and logs will be saved under `results`.

### Step 3: Evaluation
Now, we can evaluate the trained model's performance:
```
python tools/evaluate.py --exp_name SPELL_ASD_default --eval_type AVA_ASD
```
This will print the evaluation score.

In a similar way, we can perform the experiments on the task of action localization with the default configuration:
```
# Step 1
python data/generate_graph.py --features SLOWFAST-64x2-R101 --ec_mode cdi --time_span 90 --tau 3
# Step 2
python tools/train_context_reasoning.py --cfg configs/action-localization/ava_actions/SPELL_default.yaml
# Step 3
python tools/evaluate.py --exp_name SPELL_AL_default --eval_type AVA_AL
```

## Note
- For RESNET18-TSM-AUG and RESNET50-TSM-AUG, we used the same features used in [SPELL](https://github.com/SRA2/SPELL).
- For SLOWFAST-64x2-R101, we used the official code of [SlowFast](https://github.com/facebookresearch/SlowFast). We used the pretrained checkpoint ([`SLOWFAST_64x2_R101_50_50.pkl`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl)) in [SlowFast Model Zoo](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md).

## Contributor
Graph-LTVU is written and maintained by [Kyle Min](https://sites.google.com/view/kylemin)

## Citation
ECCV 2022 paper:
```bibtex
@inproceedings{min2022learning,
  title={Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  booktitle={European Conference on Computer Vision},
  pages={371--387},
  year={2022},
  organization={Springer}
}
```

Technical report for Ego4D challenge 2022:
```bibtex
@article{min2022intel,
  title={Intel Labs at Ego4D Challenge 2022: A Better Baseline for Audio-Visual Diarization},
  author={Min, Kyle},
  journal={arXiv preprint arXiv:2210.07764},
  year={2022}
}
```

## Disclaimer

> This “research quality code”  is for Non-Commercial purposes and provided by Intel “As Is” without any express or implied warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.

> AVA-ActiveSpeaker, AVA-Actions: Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.
