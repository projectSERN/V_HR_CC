import argparse

parser = argparse.ArgumentParser()

parser.add_argument('DATASET_1', type=str, default='VIPL-HR-V1', help='Dataset for 1-dataset study or first dataset for 2-dataset cross-corpus study', choices=['VIPL-HR-V1', 'UBFC-rPPG', 'PURE', 'MPSC-rPPG'])
parser.add_argument('--DATASET_2', type=str, help='Second dataset for 2-dataset cross-corpus study', choices=['VIPL-HR-V1', 'UBFC-rPPG', 'PURE', 'MPSC-rPPG'])
parser.add_argument('--CC_MINI', action='store_true', help='Use a 50-sample version of VIPL-HR-V1 in place of full dataset for cross-corpus study')
parser.add_argument('--CC_FULL', action='store_true', help='Perform full cross-corpus study using all datasets')
parser.add_argument('--CC_PROP', type=int, help='Proportion of dataset 2 to use for train+val set in 2-dataset cross-corpus study')
parser.add_argument('--CC_FULL_PROPS', type=float, nargs='+', help='Proportion of each dataset to use for train+val set in full cross-corpus study, in order: VIPL-HR-V1, UBFC-rPPG, PURE, MPSC-rPPG')

parser.add_argument('--FRAME_RATE_VIPL', type=int, default=25, help='Frame rate of VIPL-HR-V1 dataset')
parser.add_argument('--FRAME_RATE_UBFC', type=int, default=30, help='Frame rate of UBFC-rPPG dataset')
parser.add_argument('--FRAME_RATE_PURE', type=int, default=30, help='Frame rate of PURE dataset')
parser.add_argument('--FRAME_RATE_MPSC', type=int, default=30, help='Frame rate of MPSC-rPPG dataset')

parser.add_argument('--CLIP_SIZE', type=int, default=125, help='Number of frames per clip')
parser.add_argument('--STRIDE', type=int, default=10, help='Stride length in frames')
parser.add_argument('--SMOOTH_LOSS_WINDOW', type=int, default=6, help='Window size for smooth loss')
parser.add_argument('--ROI_GRID_SIZE', type=int, default=5, help='Side length of square ROI grid for ST map generation')
parser.add_argument('--SKIN_SEG_MODEL', type=str, default='skin_u2netp.onnx', help='Skin segmentation model to use in ST map generation')

parser.add_argument('--LR', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size')
parser.add_argument('--NUM_WORKERS', type=int, default=8, help='Number of workers for data loader')
parser.add_argument('--EPOCHS', type=int, default=20, help='Number of epochs')
parser.add_argument('--PATIENCE', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--PLOT_VIDEO', type=str, help='Path to video for bland altman and GT vs pred plots')

parser.add_argument('--GPU', type=int, default=0, help='ID of GPU to use')
parser.add_argument('--SEED', type=int, default=42, help='Random seed for reproducibility')
config = parser.parse_args()

config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'

# NOTE: ensure to name the dataset folders as 'VIPL-HR-V1', 'UBFC-rPPG', 'PURE', and 'MPSC-rPPG'
# NOTE: find and replace all "/scratch/zce*/" to where your datasets are located
# NOTE: find and replace all "/home/zce*/" to where your RhythmNet--CC_final repo is located

# first and/or the only dataset paths
config.SOURCE_PATH_1 = f"/scratch/zce*/{config.DATASET_1}/"
config.ST_MAPS_PATH_1 = f"{config.SOURCE_PATH_1}st_maps_{config.CLIP_SIZE}c_{config.STRIDE}s/"

# second dataset paths
if config.DATASET_2 is not None:
    config.SOURCE_PATH_2 = f"/scratch/zce*/{config.DATASET_2}/"
    config.ST_MAPS_PATH_2 = f"{config.SOURCE_PATH_2}st_maps_{config.CLIP_SIZE}c_{config.STRIDE}s/"

# results directory paths
if config.DATASET_2 is None and config.CC_PROP is None and not config.CC_FULL:
    if config.DATASET_1 == 'VIPL-HR-V1' and config.CC_MINI:
        config.RESULTS_PATH = f"/home/zce*/RhythmNet--CC_final/results_{config.DATASET_1}_mini/"
    else:
        config.RESULTS_PATH  = f"/home/zce*/RhythmNet--CC_final/results_{config.DATASET_1}/"
elif config.DATASET_2 is not None and config.CC_PROP is not None:
    if config.DATASET_1 == 'VIPL-HR-V1' and config.CC_MINI:
        config.RESULTS_PATH = f"/home/zce*/RhythmNet--CC_final/results_{config.DATASET_1}_mini_{config.CC_PROP}_of_{config.DATASET_2}/"
    elif config.DATASET_2 == 'VIPL-HR-V1' and config.CC_MINI:
        config.RESULTS_PATH = f"/home/zce*/RhythmNet--CC_final/results_{config.DATASET_1}_{config.CC_PROP}_of_{config.DATASET_2}_mini/"
    else:
        config.RESULTS_PATH  = f"/home/zce*/RhythmNet--CC_final/results_{config.DATASET_1}_{config.CC_PROP}_of_{config.DATASET_2}/"
elif config.CC_FULL and config.CC_FULL_PROPS is not None:
    if config.DATASET_1 == 'VIPL-HR-V1' and config.CC_MINI:
        config.RESULTS_PATH = f"/home/zce*/RhythmNet--CC_final/results_full_mini/"
    else:
        config.RESULTS_PATH  = f"/home/zce*/RhythmNet--CC_final/results_full/"