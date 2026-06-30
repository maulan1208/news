import sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent
CF_DIR  = ROOT / "cores" / "cf"
CBF_DIR = ROOT / "cores" / "cbf"

# Các tên module bị trùng giữa hai world, cần dọn giữa các lần nạp.
_SHARED = ("config", "preprocess", "encoders", "mf", "item", "recommend", "recommender")

def _purge():
    for name in _SHARED:
        sys.modules.pop(name, None)

@dataclass
class AppState:
    item_cf: object = None
    mf_cf: object = None

    cbf_recommend: object = None          
    cbf_encoders: tuple  = ()           
    cbf_cfg: object = None
    news_dict: dict = field(default_factory=dict)

    # Trending: 
    popularity: list = field(default_factory=list)   # [(news_id, count), ...]

def _load_cf(state: AppState):
    """Nạp ItemCF, MFCF và tính bảng trending từ ma trận user×item."""
    _purge()
    sys.path.insert(0, str(CF_DIR))
    try:
        import config as cf_config
        from item import ItemCF
        from mf import MatrixFactorizationCF

        state.item_cf = ItemCF().fit()
        state.mf_cf = MatrixFactorizationCF().fit()

        # Trending = số user đã đọc mỗi tin = tổng theo cột của ma trận user×news.
        matrix = load_npz(str(cf_config.USER_ITEM_MATRIX)).tocsc()
        counts = np.asarray(matrix.sum(axis=0)).ravel()      # (n_items,)
        idx2news = state.item_cf.idx2news
        order = np.argsort(counts)[::-1]                    # phổ biến -> hiếm
        state.popularity = [(idx2news[i], int(counts[i])) for i in order]
        print(f"Trending: {len(state.popularity)} tin xếp theo độ phổ biến")
    finally:
        sys.path.remove(str(CF_DIR))


def _load_cbf(state: AppState):
    """Nạp news_dict + 3 encoder (tfidf/qwen/entity) đã cache và hàm recommend."""
    _purge()
    sys.path.insert(0, str(CBF_DIR))
    try:
        from config import Config
        from preprocess import load_and_cache
        from recommender import recommend, _load_encoders

        cfg = Config()
        state.cbf_cfg = cfg
        state.news_dict = load_and_cache(cfg)
        state.cbf_encoders = _load_encoders(cfg)  
        state.cbf_recommend = recommend
    finally:
        sys.path.remove(str(CBF_DIR))

def load_all() -> AppState:
    state = AppState()
    _load_cf(state)
    _load_cbf(state)
    _purge()
    return state
