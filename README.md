# GraVi-T
This repository contains an open-source codebase for Graph-based long-term Video undersTanding (GraVi-T). It is designed to serve as a spatial-temporal graph learning framework for multiple video understanding tasks. In the current version, it supports training and evaluating one of the state-of-the-art models, [SPELL](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950367.pdf), for the tasks of active speaker detection and action localization.

In the near future, we will release more advanced graph-based approaches for other tasks, including action segmentation and the [Ego4D challenge](https://ego4d-data.org/workshops/eccv22). 
Our Spatio-temporal graph based method recently won many challenges - Ego4D audio-video diarization @ECCV22, @CVPR23. 

![](docs/images/gravit_teaser.jpg?raw=true)

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
git clone https://github.com/IntelLabs/GraVi-T.git
cd GraVi-T
pip3 install -e .
```

## Getting Started (Active Speaker Detection)
### Annotations
1) Download the annotations from the official site:
```
DATA_DIR="data/annotations"

wget https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2 -P ${DATA_DIR}
tar -xf ${DATA_DIR}/ava_activespeaker_val_v1.0.tar.bz2 -C ${DATA_DIR}
```

2) Preprocess the annotations:
```
python data/annotations/merge_ava_activespeaker.py
```

### Features
Download `RESNET18-TSM-AUG.zip` from the Google Drive link from [SPELL](https://github.com/SRA2/SPELL#code-usage) and unzip under `data/features`.
> We use the features from the thirdparty repositories.

### Directory Structure
The data directories should look like as follows:
```
|-- data
    |-- annotations
        |-- ava_activespeaker_val_v1.0.csv
    |-- features
        |-- RESNET18-TSM-AUG
            |-- train
            |-- val
```

### Experiments
We can perform the experiments on active speaker detection with the default configuration by following the three steps below.

#### Step 1: Graph Generation
Run the following command to generate spatial-temporal graphs from the features:
```
python data/generate_graph.py --features RESNET18-TSM-AUG --ec_mode csi --time_span 90 --tau 0.9
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video, which spans about 90 seconds (specified by `--time_span`).

#### Step 2: Training
Next, run the training script by passing the default configuration file:
```
python tools/train_context_reasoning.py --cfg configs/active-speaker-detection/ava_active-speaker/SPELL_default.yaml
```
The results and logs will be saved under `results`.

#### Step 3: Evaluation
Now, we can evaluate the trained model's performance:
```
python tools/evaluate.py --exp_name SPELL_ASD_default --eval_type AVA_ASD
```
This will print the evaluation score.

## Getting Started (Action Localization)
Please refer to the instructions in [GETTING_STARTED_AL.md](docs/GETTING_STARTED_AL.md).

## Contributor
GraVi-T is written and maintained by [Kyle Min](https://sites.google.com/view/kylemin)

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
