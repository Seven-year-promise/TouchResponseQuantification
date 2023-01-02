from pathlib import Path


FILE_PATH = Path(__file__).parent.resolve()

QUANTIFY_DATA_PATH = FILE_PATH / "QuantificationResults/"
ACTION_DATA_PATH = FILE_PATH / "OldCompoundsMoA.csv"
RESULT_PATH = FILE_PATH / "results/"