import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import config
from dataset import DatasetHR
from trainer import RhythmNetTrainer

def main():
    """ Training and evaluation routines for RhythmNet model. """

    ## Set up environment ##
    # GPU and CUDA settings
    if config.GPU >= 0:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
        torch.cuda.empty_cache()

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # random seed settings
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # results directory
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    # !!! relevant part for cross-corpus study !!! #

    ## Load data ##
    if config.DATASET_2 is None and config.CC_PROP is None and not config.CC_FULL:                                          # for 1-dataset cross-corpus study
        # get all ST map files for the dataset
        if config.DATASET_1 == 'VIPL-HR-V1':
            all_st_map_files = glob.glob(os.path.join(config.ST_MAPS_PATH, 'p*', 'v*', 'source1', 'st_maps.npy'), recursive=True)
            if config.CC_MINI:
                np.random.shuffle(all_st_map_files)
                random_start = np.random.randint(0, len(all_st_map_files) - 50)
                all_st_map_files = all_st_map_files[random_start:random_start + 50]
        elif config.DATASET_1 == 'UBFC-rPPG':
            # NOTE: add appropriate code here for UBFC-rPPG dataset's file structure
            # all_st_map_files = ...
            pass
        elif config.DATASET_1 == 'PURE':
            # NOTE: add appropriate code here for PURE dataset's file structure
            # all_st_map_files = ...
            pass
        elif config.DATASET_1 == 'MPSC-rPPG':
            # NOTE: add appropriate code here for MPSC-rPPG dataset's file structure
            # all_st_map_files = ...
            pass
    
        # split video samples into train (70%), val (10%) and test (20%) sets
        video_files_train, video_files_test = train_test_split(all_st_map_files, test_size=0.2, random_state=config.SEED)   # 0.2 x 1 = 20% test
        video_files_train, video_files_val = train_test_split(video_files_train, test_size=0.125, random_state=config.SEED) # 0.125 x 0.8 = 10% val
    
        # print num of video samples in each set
        if config.DATASET_1 == 'VIPL-HR-V1' and config.CC_MINI:
            print(f"----\n1-DATASET CROSS-CORPUS STUDY: {config.DATASET_1}_mini")
        else:
            print(f"----\n1-DATASET CROSS-CORPUS STUDY: {config.DATASET_1}")
        print(f"Training set: {len(video_files_train)} video samples")
        print(f"Val set: {len(video_files_val)} video samples")
        print(f"Test set: {len(video_files_test)} video samples")

    elif config.DATASET_2 is not None and config.CC_PROP is not None:                                                       # for 2-dataset cross-corpus study
        # get all ST map files for dataset 1
        if config.DATASET_1 == 'VIPL-HR-V1':
            all_d1_st_map_files = glob.glob(os.path.join(config.ST_MAPS_PATH_1, 'p*', 'v*', 'source1', 'st_maps.npy'), recursive=True)
            if config.CC_MINI:
                np.random.shuffle(all_d1_st_map_files)
                random_start = np.random.randint(0, len(all_d1_st_map_files) - 50)
                all_d1_st_map_files = all_d1_st_map_files[random_start:random_start + 50]
        elif config.DATASET_1 == 'UBFC-rPPG':
            # NOTE: add appropriate code here for UBFC-rPPG dataset's file structure
            # all_d1_st_map_files = ...
            pass
        elif config.DATASET_1 == 'PURE':
            # NOTE: add appropriate code here for PURE dataset's file structure
            # all_d1_st_map_files = ...
            pass
        elif config.DATASET_1 == 'MPSC-rPPG':
            # NOTE: add appropriate code here for MPSC-rPPG dataset's file structure
            # all_d1_st_map_files = ...
            pass

        # get all ST map files for dataset 2
        if config.DATASET_2 == 'VIPL-HR-V1':
            all_d2_st_map_files = glob.glob(os.path.join(config.ST_MAPS_PATH_2, 'p*', 'v*', 'source1', 'st_maps.npy'), recursive=True)
            if config.CC_MINI:
                np.random.shuffle(all_d2_st_map_files)
                random_start = np.random.randint(0, len(all_d2_st_map_files) - 50)
                all_d2_st_map_files = all_d2_st_map_files[random_start:random_start + 50]
        elif config.DATASET_2 == 'UBFC-rPPG':
            # NOTE: add appropriate code here for UBFC-rPPG dataset's file structure
            # all_d2_st_map_files = ...
            pass
        elif config.DATASET_2 == 'PURE':
            # NOTE: add appropriate code here for PURE dataset's file structure
            # all_d2_st_map_files = ...
            pass
        elif config.DATASET_2 == 'MPSC-rPPG':
            # NOTE: add appropriate code here for MPSC-rPPG dataset's file structure
            # all_d2_st_map_files = ...
            pass

        # split dataset 1 into train (70%), val (10%) and test (20%) sets
        d1_video_files_train, d1_video_files_test = train_test_split(all_d1_st_map_files, test_size=0.2, random_state=config.SEED)
        d1_video_files_train, d1_video_files_val = train_test_split(d1_video_files_train, test_size=0.125, random_state=config.SEED)

        # split dataset 2 into train (config.CC_PROP * 100%), val (10%) and test (20%) sets
        d2_video_files_train, d2_video_files_val_test = train_test_split(all_d2_st_map_files, test_size=0.3, train_size=config.CC_PROP, random_state=config.SEED)
        d2_video_files_val, d2_video_files_test = train_test_split(d2_video_files_val_test, test_size=2/3, random_state=config.SEED)

        # combine train/val/test sets and shuffle
        video_files_train = d1_video_files_train + d2_video_files_train
        video_files_val = d1_video_files_val + d2_video_files_val
        video_files_test = d1_video_files_test + d2_video_files_test
        np.random.shuffle(video_files_train)
        np.random.shuffle(video_files_val)
        np.random.shuffle(video_files_test)

        # print num of video from each dataset and in each set
        if config.DATASET_1 == 'VIPL-HR-V1' and config.CC_MINI:
            print(f"----\n2-DATASET CROSS-CORPUS STUDY: {config.DATASET_1}_mini AND {config.DATASET_2}")
        elif config.DATASET_2 == 'VIPL-HR-V1' and config.CC_MINI:
            print(f"----\n2-DATASET CROSS-CORPUS STUDY: {config.DATASET_1} AND {config.DATASET_2}_mini")
        else:
            print(f"----\n2-DATASET CROSS-CORPUS STUDY: {config.DATASET_1} AND {config.DATASET_2}")
        print(f"{config.DATASET_1}: {len(d1_video_files_train)} train, {len(d1_video_files_val)} val, {len(d1_video_files_test)} test set samples")
        print(f"{config.DATASET_2}: {len(d2_video_files_train)} train, {len(d2_video_files_val)} val, {len(d2_video_files_test)} test set samples")
        print(f"Total corpus: {len(video_files_train)} train, {len(video_files_val)} val, {len(video_files_test)} test set samples")
        
    elif config.CC_FULL and config.CC_FULL_PROPS is not None:                                                               # for full cross-corpus study
        # comment out any dataset that is not used in the study (e.g., if MPSC-rPPG did not yield satisfactory results in 1-dataset and 2-dataset studies, ignore it in the full study)
        # get all ST map files for all datasets
        all_vipl_st_map_files = glob.glob(os.path.join(config.ST_MAPS_PATH_1, 'p*', 'v*', 'source1', 'st_maps.npy'), recursive=True)
        if config.CC_MINI:
            np.random.shuffle(all_vipl_st_map_files)
            random_start = np.random.randint(0, len(all_vipl_st_map_files) - 50)
            all_vipl_st_map_files = all_vipl_st_map_files[random_start:random_start + 50]
        # NOTE: add appropriate code here for other datasets' file structures
        # all_ubfc_st_map_files = ...
        # all_pure_st_map_files = ...
        # all_mpsc_st_map_files = ...

        # Split each dataset into train (config.CC_FULL_PROPS[i] * 100%), val (10%), and test (20%) of the full dataset
        # VIPL-HR-V1 or VIPL-HR-V1_mini
        vipl_video_files_train, vipl_video_files_val_test = train_test_split(all_vipl_st_map_files, test_size=0.3, train_size=config.CC_FULL_PROPS[0], random_state=config.SEED)
        vipl_video_files_val, vipl_video_files_test = train_test_split(vipl_video_files_val_test, test_size=2/3, random_state=config.SEED)

        # UBFC-rPPG
        ubfc_video_files_train, ubfc_video_files_val_test = train_test_split(all_ubfc_st_map_files, test_size=0.3, train_size=config.CC_FULL_PROPS[1], random_state=config.SEED)
        ubfc_video_files_val, ubfc_video_files_test = train_test_split(ubfc_video_files_val_test, test_size=2/3, random_state=config.SEED)

        # PURE
        pure_video_files_train, pure_video_files_val_test = train_test_split(all_pure_st_map_files, test_size=0.3, train_size=config.CC_FULL_PROPS[2], random_state=config.SEED)
        pure_video_files_val, pure_video_files_test = train_test_split(pure_video_files_val_test, test_size=2/3, random_state=config.SEED)

        # MPSC-rPPG
        mpsc_video_files_train, mpsc_video_files_val_test = train_test_split(all_mpsc_st_map_files, test_size=0.3, train_size=config.CC_FULL_PROPS[3], random_state=config.SEED)
        mpsc_video_files_val, mpsc_video_files_test = train_test_split(mpsc_video_files_val_test, test_size=2/3, random_state=config.SEED)

        # combine train/val/test sets and shuffle
        video_files_train = vipl_video_files_train + ubfc_video_files_train + pure_video_files_train + mpsc_video_files_train
        video_files_val = vipl_video_files_val + ubfc_video_files_val + pure_video_files_val + mpsc_video_files_val
        video_files_test = vipl_video_files_test + ubfc_video_files_test + pure_video_files_test + mpsc_video_files_test
        np.random.shuffle(video_files_train)
        np.random.shuffle(video_files_val)
        np.random.shuffle(video_files_test)

        # print num of video from each dataset and in each set
        print(f"----\nFULL CROSS-CORPUS STUDY")
        if config.CC_MINI:
            print(f"VIPL-HR-V1_mini: {len(vipl_video_files_train)} train, {len(vipl_video_files_val)} val, {len(vipl_video_files_test)} test set samples")
        else:
            print(f"VIPL-HR-V1: {len(vipl_video_files_train)} train, {len(vipl_video_files_val)} val, {len(vipl_video_files_test)} test set samples")
        print(f"UBFC-rPPG: {len(ubfc_video_files_train)} train, {len(ubfc_video_files_val)} val, {len(ubfc_video_files_test)} test set samples")
        print(f"PURE: {len(pure_video_files_train)} train, {len(pure_video_files_val)} val, {len(pure_video_files_test)} test set samples")
        print(f"MPSC-rPPG: {len(mpsc_video_files_train)} train, {len(mpsc_video_files_val)} val, {len(mpsc_video_files_test)} test set samples")
        print(f"Total corpus: {len(video_files_train)} train, {len(video_files_val)} val, {len(video_files_test)} test set samples")

    else:
        raise ValueError("Invalid combination of arguments for cross-corpus study. Refer to README.md for details.")

    # !!! end of relevant part for cross-corpus study !!! #

    ## Create DataLoaders ##
    # split train/val/test set video samples and their ground truth labels into clips and create train/val/test DataLoaders
    train_dl = DataLoader(
        DatasetHR(video_files_train, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,                                                                                  # shuffle=False for GRU/smooth loss computation
    )

    val_dl = DataLoader(
        DatasetHR(video_files_val, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,
    )

    test_dl = DataLoader(
        DatasetHR(video_files_test, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,
    )

    ## Train and evaluate model ##
    trainer = RhythmNetTrainer(train_dl, val_dl, test_dl)

    print(f"----\nTRAINING & VAL")
    with open(f"{config.RESULTS_PATH}/config_args.txt", 'w') as results_file:
        for arg in vars(config):
            results_file.write(f"{arg:<30}: {getattr(config, arg)}\n")
    trainer.train()

    print(f"----\nTESTING")
    trainer.test()

if __name__ == '__main__':
    main()