import numpy as np
from sklearn.metrics import roc_auc_score


def ndcg(labels, scores, k):
    order  = np.argsort(scores)[::-1][:k]
    gains  = np.array(labels)[order]
    disc   = np.log2(np.arange(2, len(gains) + 2))
    ideal  = sorted(labels, reverse=True)[:k]
    ideal_dcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return (gains / disc).sum() / (ideal_dcg + 1e-8)


def _norm(s):
    std = s.std()
    return (s - s.mean()) / (std + 1e-8)


def _compute_scores(mode, history, candidates, news_dict,
                    tfidf_enc, qwen_enc, entity_enc,
                    alpha_tfidf, alpha_qwen, alpha_entity):
    if mode == 'tfidf':
        return tfidf_enc.score(history, candidates, news_dict)

    if mode == 'qwen':
        return qwen_enc.score(history, candidates)

    if mode == 'entity':
        return entity_enc.score(history, candidates)

    if mode == 'hybrid':
        parts, weights = [], []
        if tfidf_enc is not None and alpha_tfidf > 0:
            parts.append(_norm(tfidf_enc.score(history, candidates, news_dict)))
            weights.append(alpha_tfidf)
        if qwen_enc is not None and alpha_qwen > 0:
            parts.append(_norm(qwen_enc.score(history, candidates)))
            weights.append(alpha_qwen)
        if entity_enc is not None and alpha_entity > 0:
            parts.append(_norm(entity_enc.score(history, candidates)))
            weights.append(alpha_entity)

        total = sum(weights)
        return sum(w / total * s for w, s in zip(weights, parts))

    raise ValueError(f'mode không hợp lệ: {mode}')


def _diverse_rerank(candidates, scores, news_dict, top_k, penalty=0.8):
    scores_map = dict(zip(candidates, scores))
    remaining  = list(candidates)
    results    = []
    seen_cats  = set()

    while len(results) < top_k and remaining:
        best = max(
            remaining,
            key=lambda nid: scores_map[nid] * (
                penalty if news_dict.get(nid, {}).get('category', '') in seen_cats else 1.0
            ),
        )
        results.append((best, scores_map[best]))
        seen_cats.add(news_dict.get(best, {}).get('category', ''))
        remaining.remove(best)

    return results


def recommend(user_history, all_candidates, news_dict,
              tfidf_enc=None, qwen_enc=None, entity_enc=None,
              mode='hybrid', top_k=10,
              alpha_tfidf=0.1, alpha_qwen=0.6, alpha_entity=0.3):
    # Novelty: loại bài đã đọc khỏi candidates
    history_set    = set(user_history)
    all_candidates = [c for c in all_candidates if c not in history_set]

    scores = _compute_scores(
        mode, user_history, all_candidates, news_dict,
        tfidf_enc, qwen_enc, entity_enc,
        alpha_tfidf, alpha_qwen, alpha_entity,
    )

    # Diversity: greedy rerank với penalty category trùng
    return _diverse_rerank(all_candidates, scores, news_dict, top_k)


def evaluate(behaviors, news_dict,
             tfidf_enc=None, qwen_enc=None, entity_enc=None,
             mode='hybrid',
             alpha_tfidf=0.1, alpha_qwen=0.6, alpha_entity=0.3,
             max_samples=None):
    aucs, ndcg5s, ndcg10s = [], [], []

    subset = behaviors[:max_samples] if max_samples else behaviors
    for sample in subset:
        history    = sample['history']
        candidates = sample['candidates']
        labels     = sample['labels']

        pairs = [(c, l) for c, l in zip(candidates, labels) if c in news_dict]
        if not pairs:
            continue
        candidates, labels = zip(*pairs)

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        scores = _compute_scores(
            mode, history, list(candidates), news_dict,
            tfidf_enc, qwen_enc, entity_enc,
            alpha_tfidf, alpha_qwen, alpha_entity,
        )

        aucs.append(roc_auc_score(labels, scores))
        ndcg5s.append(ndcg(labels, scores, 5))
        ndcg10s.append(ndcg(labels, scores, 10))

    return {
        'AUC':     round(np.mean(aucs),    4),
        'nDCG@5':  round(np.mean(ndcg5s),  4),
        'nDCG@10': round(np.mean(ndcg10s), 4),
        'samples': len(aucs),
    }



def _load_encoders(cfg):
    """Load 3 encoder từ cache; báo lỗi rõ nếu thiếu (cần chạy main.py để build)."""
    import os
    from encoders import TFIDFEncoder, QwenEncoder, EntityEncoder

    missing = [p for p in (cfg.cache_tfidf, cfg.cache_qwen, cfg.cache_entity)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Thiếu cache encoder:\n  ' + '\n  '.join(missing) +
            '\n=> Chạy `python main.py` một lần để build encoder trước.')

    tfidf = TFIDFEncoder.load(cfg.cache_tfidf)
    qwen = QwenEncoder(cfg); qwen.load(cfg.cache_qwen)
    entity = EntityEncoder(cfg); entity.load(cfg.cache_entity)
    return tfidf, qwen, entity


def _show(news_dict, history, recs):
    print(f'\nHistory:')
    for nid in history[:5]:
        print(f'  - {nid}: {news_dict.get(nid, {}).get("title", "?")}')
    if not recs:
        print('Không có gợi ý.')
        return
    print(f'\nTop {len(recs)} gợi ý:')
    for rank, (nid, score) in enumerate(recs, 1):
        n = news_dict.get(nid, {})
        print(f'  {rank}. [{score:.3f}] ({n.get("category", "?")}) {n.get("title", "?")}')


def main():
    import argparse
    from config import Config
    from preprocess import load_and_cache, load_behaviors

    parser = argparse.ArgumentParser(description='CBF recommender (chạy riêng, không evaluate)')
    parser.add_argument('--user-index', type=int, default=None,
                        help='Index user trong dev behaviors để gợi ý.')
    parser.add_argument('--history', default=None,
                        help='Tự nhập history, vd "N1 N2 N3". Candidate = toàn bộ news.')
    parser.add_argument('--mode', default='hybrid',
                        choices=['tfidf', 'qwen', 'entity', 'hybrid'])
    parser.add_argument('--top-k', type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    top_k = args.top_k or cfg.top_k
    news_dict = load_and_cache(cfg)
    tfidf, qwen, entity = _load_encoders(cfg)

    def rec(history, candidates):
        return recommend(
            user_history=history, all_candidates=candidates, news_dict=news_dict,
            tfidf_enc=tfidf, qwen_enc=qwen, entity_enc=entity,
            mode=args.mode, top_k=top_k,
            alpha_tfidf=cfg.alpha_tfidf, alpha_qwen=cfg.alpha_qwen, alpha_entity=cfg.alpha_entity,
        )

    # Chế độ tự nhập history -> candidate là toàn bộ news.
    if args.history:
        if args.mode in ('tfidf', 'hybrid'):
            print('[Cảnh báo] mode tfidf/hybrid chấm trên TOÀN BỘ news rất tốn RAM; '
                  'nên dùng --mode qwen, hoặc dùng --user-index để giới hạn candidate.')
        history = args.history.split()
        _show(news_dict, history, rec(history, list(news_dict.keys())))
        return

    # Chế độ theo user trong dev: candidate = danh sách impression của user (nhỏ, an toàn).
    behaviors = load_behaviors(f'{cfg.dev_dir}/behaviors.tsv')
    print(f'Đã load {len(behaviors)} behaviors (dev). mode={args.mode}, top_k={top_k}')

    def run_one(idx):
        s = behaviors[idx]
        print(f'\n{"=" * 60}\nUser [{idx}]: {s["user_id"]}')
        _show(news_dict, s['history'], rec(s['history'], s['candidates']))
        print('=' * 60)

    if args.user_index is not None:
        run_one(args.user_index)
        return

    print(f'Nhập index user (0..{len(behaviors) - 1}), "q" để thoát.')
    while True:
        raw = input('\nIndex user: ').strip()
        if raw.lower() in {'q', 'quit', 'exit'}:
            break
        if not raw.isdigit() or not (0 <= int(raw) < len(behaviors)):
            print('Index không hợp lệ.')
            continue
        run_one(int(raw))


if __name__ == '__main__':
    main()
