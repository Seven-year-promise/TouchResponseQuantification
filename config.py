from pathlib import Path


FILE_PATH = Path(__file__).parent.resolve()

UNET_MODEL_PATH = FILE_PATH / "Methods/UNet_tf/ori_UNet/models_update/UNet14000.pb"
QUANTIFY_DATA_PATH =FILE_PATH / "HTS/data/20220826-hts-4larvae-12mm-C0-11/"
TRACKING_SAVE_PATH = FILE_PATH / "tracking_saved/"
QUANTIFY_SAVE_PATH = FILE_PATH / "HTS/QuantificationResults/20220826-hts-4larvae-12mm-C0-11/"