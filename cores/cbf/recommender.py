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
