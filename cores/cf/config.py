from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "cf"

TRAIN_BEHAVIORS = RAW_DIR / "MINDsmall_train" / "behaviors.tsv"
DEV_BEHAVIORS = RAW_DIR / "MINDsmall_dev" / "behaviors.tsv"
TRAIN_NEWS = RAW_DIR / "MINDsmall_train" / "news.tsv"

USER_ITEM_MATRIX = PROCESSED_DIR / "user_item.npz"
USER2IDX_PATH = PROCESSED_DIR / "user2idx.pkl"
NEWS2IDX_PATH = PROCESSED_DIR / "news2idx.pkl"

TOP_K = 10
EVAL_BATCH_SIZE = 256
