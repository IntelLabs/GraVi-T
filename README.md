# GraVi-T
This repository contains an open-source codebase for Graph-based long-term Video undersTanding (GraVi-T). It is designed to serve as a spatial-temporal graph learning framework for multiple video understanding tasks. In the current version, it supports training and evaluating one of the state-of-the-art models, [SPELL](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950367.pdf), for the tasks of active speaker detection, action localization, and action segmentation.

In the near future, we will release more advanced graph-based approaches (e.g. [STHG](https://arxiv.org/abs/2306.10608)) for other tasks, including video summarization and audio-visual diarization.

![](docs/images/gravit_teaser.jpg?raw=true)

## Ego4D Challenges and ActivityNet
We want to note that our method has recently won many challenges, including the Ego4D challenges [@ECCV22](https://ego4d-data.org/workshops/eccv22/), [@CVPR23](https://ego4d-data.org/workshops/cvpr23/) and ActivityNet [@CVPR22](https://research.google.com/ava/challenge.html). We summarize ASD (active speaker detection) and AVD (audio-visual diarization) performance comparisons on the validation set of the Ego4D dataset:
|  ASD Model  |  ASD mAP(%)&#8593;  |  ASD mAP@0.5(%)&#8593;  | AVD DER(%)&#8595;  |
|:------------|:-------------------:|:-----------------------:|:------------------:|
|  RegionCls  |   -                 |  24.6                   |  80.0              |
|  TalkNet    |   -                 |  50.6                   |  79.3              |
|  [SPELL](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950367.pdf) (Ours)      |   71.3  |  60.7  |  66.6  |
|  [STHG](https://arxiv.org/abs/2306.10608) (Ours)  |   **75.7**   |  **63.7**  |  **59.4**  |

:bulb:In this table, We report two metrics to evaluate ASD performance: mAP quantifies the ASD results by assuming that the face bound-box detections are the ground truth (i.e. assuming the perfect face detector), whereas mAP@0.5 quantifies the ASD results on the detected face bounding boxes (i.e. a face detection is considered positive only if the IoU between a detected face bounding box and the ground-truth exceeds 0.5). For AVD, we report DER (diarization error rate): a lower DER value indicates a better AVD performance. For more information, please refer to our technical reports for the challenge.

:bulb:We computed mAP@0.5 by using [Ego4D's official evaluation tool](https://github.com/EGO4D/audio-visual/tree/main/active-speaker-detection/active_speaker/active_speaker_evaluation)

## Use Cases and Performance
### Active Speaker Detection (Dataset: AVA-ActiveSpeaker v1.0)
|  Model         |      Feature       |     validation mAP (%)     |
|:---------------|:------------------:|:--------------------------:|
|  SPELL (Ours)  |  RESNET18-TSM-AUG  |   **94.2** (up from 88.0)  |
|  SPELL (Ours)  |  RESNET50-TSM-AUG  |   **94.9** (up from 89.3)  |
> Numbers in parentheses indicate the mAP scores without using the suggested graph learning method.

### Action Localization (Dataset: AVA-Actions v2.2)
|  Model         |         Feature        |     validation mAP (%)     |
|:---------------|:----------------------:|:--------------------------:|
|  SPELL (Ours)  |   SLOWFAST-64x2-R101   |   **36.8** (up from 29.4)  |
> Number in parentheses indicates the mAP score without using the suggested graph learning method.

### Action Segmentation (Dataset: 50Salads - split2)
|  Model         |   Feature    |         F1@0.1 (%)        |           Acc (%)         |
|:---------------|:------------:|:-------------------------:|:-------------------------:|
|  SPELL (Ours)  |   MSTCN++    |  **84.7** (up from 83.4)  |  **85.0** (up from 84.6)  |
|  SPELL (Ours)  |   ASFORMER   |  **89.8** (up from 86.1)  |  **88.2** (up from 87.8)  |
> Numbers in parentheses indicate the scores without using the suggested graph learning method.

### Video Summarization (Datasets: SumMe & TVSum)
|  Model         |              Feature              | [Kendall's Tau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau)* | [Spearman's Rho](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html#scipy.stats.spearmanr)* |
|:---------------|:---------------------------------:|:-------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|  SPELL (Ours)  | eccv16_dataset_summe_google_pool5 |                                                  **0.12** (up from 0.09)                                                  |                                                 **0.16** (up from 0.12)                                                  |
|  SPELL (Ours)  | eccv16_dataset_tvsum_google_pool5 |                                                  **0.30** (up from 0.27)                                                  |                                                 **0.42** (up from 0.39)                                                  |
> Numbers in parentheses indicate the scores without using the suggested graph learning method.\
>  *Correlation metric between predicted frame importance and ground truth. 

## Requirements
Preliminary requirements:
- Python>=3.7
- CUDA 11.6

Run the following command if you have CUDA 11.6:
```
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyYAML, pandas, and [PyG](https://www.pyg.org)>=2.0.3 with CUDA>=11.1

## Installation
After confirming the above requirements, run the following commands:
```
git clone https://github.com/IntelLabs/GraVi-T.git
cd GraVi-T
pip3 install -e .
```

## Getting Started (Active Speaker Detection)
### Annotations
1) Download the annotations of AVA-ActiveSpeaker from the official site:
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
The data directories should look as follows:
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
python data/generate_spatial-temporal_graphs.py --features RESNET18-TSM-AUG --ec_mode csi --time_span 90 --tau 0.9
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

## Getting Started (Action Segmentation)
Please refer to the instructions in [GETTING_STARTED_AS.md](docs/GETTING_STARTED_AS.md).

## Getting Started (Video Summarization)
Please refer to the instructions in [GETTING_STARTED_VS.md](docs/GETTING_STARTED_VS.md).

## Contributor
GraVi-T is written and maintained by [Kyle Min](https://github.com/kylemin) (from version 1.0.0 to 1.1.0). Please contact me if you want to become a contributor to this library.

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

Technical report for Ego4D challenge 2023:
```bibtex
@article{min2023sthg,
  title={STHG: Spatial-Temporal Heterogeneous Graph Learning for Advanced Audio-Visual Diarization},
  author={Min, Kyle},
  journal={arXiv preprint arXiv:2306.10608},
  year={2023}
}
```

## Disclaimer

> This “research quality code”  is for Non-Commercial purposes and provided by Intel “As Is” without any express or implied warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.

> AVA-ActiveSpeaker, AVA-Actions, 50Salads, TVSum, SumMe: Please see the dataset's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.

## Datasets & Models Disclaimer

> To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s), or using a model trained on those datasets, you agree to the terms associated with those datasets and that your use complies with the applicable license.

> Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data.  Intel is not liable for any liability or damages relating to your use of public datasets.
