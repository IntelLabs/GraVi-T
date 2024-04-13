## Getting Started (Video Summarization)
### Datasets with annotations and features
We suggest using the same set of datasets used by [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) or [A2Summ](https://github.com/boheumd/A2Summ). Download the TVSum & SumMe datasets from the links provided by either of the two repositories.

### Directory Structure
The data directories should look as follows:
```
|-- data
    |-- annotations
        |-- SumMe
            |-- eccv16_dataset_summe_google_pool5.h5
        |-- TVSum
            |-- eccv16_dataset_tvsum_google_pool5.h5
```

### Experiments
We can perform the experiments on video summarization with the default configuration by following the three steps below.

#### Step 1: Graph Generation
Run the following command to generate temporal graphs from the features:

On SumMe:
```
python data/generate_temporal_graphs.py --dataset SumMe --features eccv16_dataset_summe_google_pool5 --tauf 10 --skip_factor 0
```
On TVSum:
```
python data/generate_temporal_graphs.py --dataset TVSum --features eccv16_dataset_tvsum_google_pool5 --tauf 5 --skip_factor 0
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video.

#### Step 2: Training
Next, run the training script by passing the default configuration file. You also need to specify which split to perform the experiments on:

On SumMe:
```
python tools/train_context_reasoning.py --cfg configs/video-summarization/SumMe/SPELL_default.yaml --split 4
```
On TVSum:
```
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 4
```
The results and logs will be saved under `results`.

#### Step 3: Evaluation
Now, we can evaluate the trained model's performance, You also need to specify which split to perform the evaluation on:

On SumMe:
```
python tools/evaluate.py --exp_name SPELL_VS_SumMe_default --eval_type VS_max --split 4
```
On TVSum:
```
python tools/evaluate.py --exp_name SPELL_VS_TVSum_default --eval_type VS_avg --split 4
```

This will print the evaluation scores.

#### Step 3: Evaluation Alternative
You can also get average results from all splits:

On SumMe:
```
python tools/evaluate.py --exp_name SPELL_VS_SumMe_default --eval_type VS_max --all_splits
```
On TVSum:
```
python tools/evaluate.py --exp_name SPELL_VS_TVSum_default --eval_type VS_avg --all_splits
```
#### Note:

You can use bash scripts from `gravit/utils/vs/` to train models on all the splits and get evaluation metrics for TVSum and SumMe.