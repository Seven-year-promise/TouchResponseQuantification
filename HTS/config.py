from pathlib import Path


FILE_PATH = Path(__file__).parent.resolve()

QUANTIFY_DATA_PATH = FILE_PATH / "QuantificationResults/"
ACTION_DATA_PATH = FILE_PATH / "OldCompoundsMoA.csv"
RESULT_PATH = FILE_PATH / "results/"
DATASET_PATH = FILE_PATH / "dataset/"


TEST_COMPOUNDS = ["C8", "C46", "C4", "C20", "C10", "C17", "C25", "C41", "C74", "C51", "C42", "C70", "C31", "C2", "C3"]