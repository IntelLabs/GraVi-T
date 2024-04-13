## Getting Started (Action Segmentation)
### Annotations
We suggest using the same set of annotations used by [MS-TCN++](https://github.com/sj-li/MS-TCN2) and [ASFormer](https://github.com/ChinaYi/ASFormer). Download the 50Salads dataset from the links provided by either of the two repositories.

### Features
We suggest extracting the features using [ASFormer](https://github.com/ChinaYi/ASFormer). Please use their repository and the pre-trained model checkpoints ([link](https://github.com/ChinaYi/ASFormer/tree/main#reproduce-our-results)) to extract the frame-wise features for each split of the dataset. Please extract the features from each of the four refinement layers and concatenate them. To be more specific, you can concatenate the 64-dimensional features from this [line](https://github.com/ChinaYi/ASFormer/blob/main/model.py#L315), which will give you 256-dimensional (frame-wise) features. Similarly, you can also extract MS-TCN++ features from this [line](https://github.com/sj-li/MS-TCN2/blob/master/model.py#L23).
> We use the features from the thirdparty repositories.

### Directory Structure
The data directories should look as follows:
```
|-- data
    |-- annotations
        |-- 50salads
            |-- groundTruth
            |-- splits
            |-- mapping.txt
    |-- features
        |-- ASFORMER
            |-- split1
            |-- split2
            |-- split3
            |-- split4
            |-- split5
```

### Experiments
We can perform the experiments on action segmentation with the default configuration by following the three steps below.

#### Step 1: Graph Generation
Run the following command to generate temporal graphs from the features:
```
python data/generate_temporal_graphs.py --features ASFORMER --tauf 10
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video.

#### Step 2: Training
Next, run the training script by passing the default configuration file. You also need to specify which split to perform the experiments on:
```
python tools/train_context_reasoning.py --cfg configs/action-segmentation/50salads/SPELL_default.yaml --split 2
```
The results and logs will be saved under `results`.

#### Step 3: Evaluation
Now, we can evaluate the trained model's performance. You also need to specify which split to evaluate the experiments on:
```
python tools/evaluate.py --dataset 50salads --exp_name SPELL_AS_default --eval_type AS --split 2
```
This will print the evaluation scores.
