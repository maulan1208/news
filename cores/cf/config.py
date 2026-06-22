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

# Matrix Factorization (model-based) — factors học được trong mf.py
MF_FACTORS_PATH = PROCESSED_DIR / "mf_factors.npz"
MF_DIM = 64            # số chiều không gian latent (vector ẩn k chiều)
MF_EPOCHS = 20         # số vòng lặp qua toàn bộ tương tác dương
MF_LR = 0.05           # learning rate cho SGD (gradient tự tính tay)
MF_REG = 1e-2          # hệ số phạt L2 (chống overfit)
MF_BATCH_SIZE = 8192   # số cặp (user, item) mỗi bước SGD

TOP_K = 10
EVAL_BATCH_SIZE = 256
