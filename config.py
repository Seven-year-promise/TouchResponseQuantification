from pathlib import Path


FILE_PATH = Path(__file__).parent.resolve()

UNET_MODEL_PATH = FILE_PATH / "Methods/UNet_tf/ori_UNet/models_update/UNet14000.pb"
#QUANTIFY_DATA_PATH =FILE_PATH / "HTS/data/20220910-hts-4larvae-12mm-C60-76/"
QUANTIFY_DATA_PATH =FILE_PATH / "HTS_harish/BASF_New/10.03.2023/"
TRACKING_SAVE_PATH = FILE_PATH / "tracking_saved/"
#QUANTIFY_SAVE_PATH = FILE_PATH / "HTS/QuantificationResults/20220910-hts-4larvae-12mm-C60-76/"
QUANTIFY_SAVE_PATH = FILE_PATH / "HTS_harish/BASF_New/QuantificationResults/10.03.2023/"