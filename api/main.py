from contextlib import asynccontextmanager
from fastapi import FastAPI
from loaders import load_all
from schemas import RecommendRequest, RecommendResponse, NewsItem

POOL_SIZE = 500   
TOP_K = 10
STATE = None      

@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    STATE = load_all()       
    yield

app = FastAPI(title="News Recommender", lifespan = lifespan)

def _to_items(pairs) -> list[NewsItem]:
    out = []
    for news_id, score in pairs:
        meta = STATE.news_dict.get(news_id, {})
        out.append(NewsItem(
            news_id=news_id,
            title=meta.get("title", "?"),
            category=meta.get("category", "?"),
            score=round(float(score), 4),
        ))
    return out

def _candidate_pool(history_set) -> list[str]:
    """Top-N tin trending, loại bỏ tin user đã đọc."""
    pool = []
    for news_id, _ in STATE.popularity:
        if news_id in history_set:
            continue
        pool.append(news_id)
        if len(pool) >= POOL_SIZE:
            break
    return pool

def _rank_cf(model, history, candidates) -> list:
    """ItemCF / MFCF: score_one -> dict {news_id: score} -> top-K."""
    scores = model.score_one(history, candidates)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:TOP_K]

@app.post("/recommend", response_model = RecommendResponse)
def recommend(req: RecommendRequest):
    history = req.history
    history_set = set(history)
    cold_start = len(history) == 0
    pool = _candidate_pool(history_set)

    # Trending
    trending_pairs = [(nid, cnt) for nid, cnt in STATE.popularity
                      if nid not in history_set][:TOP_K]

    if cold_start:
        return RecommendResponse(
            user_id=req.user_id, cold_start=True,
            trending=_to_items(trending_pairs),
            cbf=[], item_cf=[], mf_cf=[],
        )
    # CBF
    tfidf, qwen, entity = STATE.cbf_encoders
    cfg = STATE.cbf_cfg
    cbf_pairs = STATE.cbf_recommend(
        user_history = history, all_candidates = pool, news_dict = STATE.news_dict,
        tfidf_enc = tfidf, qwen_enc=qwen, entity_enc = entity,
        mode = "hybrid", top_k = TOP_K,
        alpha_tfidf = cfg.alpha_tfidf, alpha_qwen = cfg.alpha_qwen, alpha_entity = cfg.alpha_entity,
    )

    item_pairs = _rank_cf(STATE.item_cf, history, pool)
    mf_pairs = _rank_cf(STATE.mf_cf,   history, pool)

    return RecommendResponse(
        user_id = req.user_id, cold_start=False,
        trending = _to_items(trending_pairs),
        cbf = _to_items(cbf_pairs),
        item_cf = _to_items(item_pairs),
        mf_cf = _to_items(mf_pairs),
    )
