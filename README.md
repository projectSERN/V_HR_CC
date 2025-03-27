# RhythmNet--CC_final

## Table of Contents
[Overview](#overview)\
[Code changes](#code-changes)\
[Cross-corpus study details](#cross-corpus-study-details)\
[Expected file directories](#expected-file-directories)

## Overview
This repo includes the code for the cross-corpus study using RhythmNet. It is an extension of the `RhythmNet--VIPL_final` repo, with most changes already implemented, except for the definitions of dataset-specific paths for certain datasets.

## Code changes
Any remaining modifications to be made are highlighted as **`# NOTE:` comments** in the code.
1. Place the downloaded `skin_u2netp.onnx` file into the `src/u2net/` folder of this repo (do not rename it)
2. Install the `onnxruntime-gpu` Python package using `pip install onnxruntime-gpu` for U2-Net's GPU support
3. Rename folders containing datasets to `VIPL-HR-V1`, `UBFC-rPPG`, `PURE` and `MPSC-rPPG` (if needed) and ensure these are all in the same parent folder
4. Change `"/scratch/zce*/"` and `"/home/zce*/"` in `src/config.py` and `src/st_maps.py` to datasets' parent folder and repo folder, respectively
5. Define appropriate paths for UBFC-rPPG, PURE and MPSC-rPPG datasets in `src/st_maps.py` and `src/main.py`
6. Conduct studies detailed below (find an unused or relatively free GPU to reduce comuptation times)

## Cross-corpus study details
First, generate clip-wise ST maps and ground truth data for videos in all datasets, by running `python src/st_maps.py --GPU <#> --DATASET <VIPL-HR-V1/UBFC-rPPG/PURE/MPSC-rPPG>`. These are saved to the same folder where the dataset is stored (refer to [expected file directories](#expected-file-directories))

### 1-dataset studies
For VIPL-HR-V1 (already done), VIPL-HR-V1_mini (a 50-video subset of VIPL), UBFC-rPPG, PURE, and MPSC-rPPG datasets, train model on 70% of the dataset, validate it on 10%, and test it on remaining 20%.
- these generate the baseline results
- for study with full VIPL-HR-V1, run `python src/main.py --GPU <#> --DATASET_1 <VIPL-HR-V1/UBFC-rPPG/PURE/MPSC-rPPG>`
- for study with VIPL-HR-V1_mini, run `python src/main.py --GPU <#> --DATASET_1 VIPL-HR-V1 --CC_MINI`
- results and plots are saved to `.../RhythmNet--CC_final/results_<VIPL-HR-V1/VIPL-HR-V1_mini/UBFC-rPPG/PURE/MPSC-rPPG>`

### 2-dataset studies
For each dataset, train model on (70% of this dataset + 10%, 30%, 50% and 70% of each of the remaining datasets), validate it on (10% of this dataset + 10% of each of the remaining datasets), and test it on (20% of this dataset + 20% of each of the remaining datasets).
- for study with full VIPL-HR-V1, run `python src/main.py --GPU <#> --DATASET_1 <xxx> --DATASET_2 <yyy> --CC_PROP <0.1/0.3/0.5/0.7>` with all combinations of VIPL-HR-V1, UBFC-rPPG, PURE, and MPSC-rPPG
- for study with VIPL-HR-V1_mini, run `python src/main.py --GPU <#> --DATASET_1 VIPL-HR-V1 --DATASET_2 <yyy> --CC_MINI --CC_PROP <0.1/0.3/0.5/0.7>` and `python src/main.py --GPU <#> --DATASET_1 <xxx> --DATASET_2 VIPL-HR-V1 --CC_MINI --CC_PROP <0.1/0.3/0.5/0.7>` with all combinations of VIPL-HR-V1_mini, UBFC-rPPG, PURE, and MPSC-rPPG
- results and plots are saved to `.../RhythmNet--CC_final/results_<xxx>_<0.1/0.3/0.5/0.7>_of_<yyy>`

### Full studies
After identifying the datasets with "transferability", incorporate all datasets into a full corpus. Form the training set based on the optimal proportions (`op_<xxx>` = 0.1/0.3/0.5/0.7) determined in previous studies. Validate model on 10% of all datasets and test on 20% of all datasets, as done in previous two studies.
- if any datasets have been identified to not contribute to learning at a satisfactory level, these can be omitted from this full cross-corpus study
- for study with full VIPL-HR-V1, run `python src/main.py --GPU <#> --CC_FULL --CC_FULL_PROPS <op_VIPL-HR-V1> <op_UBFC-rPPG> <op_PURE> <op_MPSC-rPPG>`
- for study with VIPL-HR-V1_mini, run `python src/main.py --GPU <#> --CC_MINI --CC_FULL --CC_FULL_PROPS <op_VIPL-HR-V1_mini> <op_UBFC-rPPG> <op_PURE> <op_MPSC-rPPG>`
- results and plots are saved to `.../RhythmNet--CC_final/results_<full/full_mini>`

All of this enables us to conduct a qualitative analysis of model performance with respect to the quality and/or content of the datasets.

## Expected file directories
A dataset folder, e.g., VIPL-HR-V1, after generating ST maps:
```
/scratch/zce*/VIPL-HR-V1/
|-- p1/
|   |-- v1/
|   |   |-- source1/
|   |   |   |-- video.avi
|   |   |   |-- gt_HR.csv
|   |   |-- ...
|   |   |-- source4/
|   |-- ...
|   |-- v9/
|-- ...
|-- p107/
|
|-- st_maps_125c_10s/
|   |-- p1/
|   |   |-- v1/
|   |   |   |-- source1/
|   |   |   |   |-- st_maps.npy             -> clip-wise ST maps of shape (num of clips in video, 125, 25, 3)
|   |   |   |   |-- gt_HRs/                 -> clip-wise ground truth data
|   |   |   |   |   |-- gt_HR_clip_0.txt
|   |   |   |   |   |-- gt_HR_clip_1.txt
|   |   |   |   |   |-- ...
|   |   |-- ...
|   |   |-- v6/
|   |-- ...
|   |-- p107/
```

`RhythmNet--CC_final` repo after all cross-corpus studies:
```
/home/zce*/RhythmNet--CC_final/
|-- results_<xxx>/
|   |-- best_model.pth
|   |-- config_args.txt
|   |-- last_model.pth
|   |-- test_results.txt
|   |-- train_val_stats/
|       |-- all_metrics.png
|       |-- bland_altman [optional]
|       |-- epoch_stats.txt
|       |-- loss.png
|       |-- mae.png
|       |-- mape.png
|       |-- rmse.png
|       |-- GT_vs_pred [optional]
|-- results_<xxx>_<0.1/0.3/0.5/0.7>_of_<yyy>/
|   |-- ...
|-- results_full/
|   |-- ...
|
|-- src/
|   |-- rtgene/
|   |   |-- ...
|   |-- u2net/
|   |   |-- inference.py
|   |   |-- skin_u2netp.onnx
|   |-- utils/
|   |   |-- custom_loss.py
|   |   |-- metrics.py
|   |   |-- plots.py
|   |   |-- rhythmnet_loss.py
|   |-- config.py
|   |-- dataset.py
|   |-- main.py
|   |-- model.py
|   |-- st_maps.py
|   |-- trainer.py
|
|-- README.md
|-- requirements.txt
```