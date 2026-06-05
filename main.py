import os
import json
from datetime import datetime
from preprocess import load_and_cache, load_behaviors
from config import Config
from encoders import TFIDFEncoder, QwenEncoder, EntityEncoder
from recommender import evaluate, recommend

def main():
    cfg = Config()
    os.makedirs(cfg.processed_dir, exist_ok=True)

    #  1. Load & cache news
    all_news  = load_and_cache(cfg)
    behaviors = load_behaviors(f'{cfg.dev_dir}/behaviors.tsv')
    print(f'News: {len(all_news)} | Behaviors: {len(behaviors)}')

    #  2. TF-IDF 
    if os.path.exists(cfg.cache_tfidf):
        tfidf = TFIDFEncoder.load(cfg.cache_tfidf)
    else:
        tfidf = TFIDFEncoder(cfg)
        tfidf.fit_transform(all_news)
        tfidf.save(cfg.cache_tfidf)

    # 3. Qwen
    qwen = QwenEncoder(cfg)
    if os.path.exists(cfg.cache_qwen):
        qwen.load(cfg.cache_qwen)
    else:
        qwen.encode_news(all_news)
        qwen.save(cfg.cache_qwen)

    #4. Entity
    entity = EntityEncoder(cfg)
    if os.path.exists(cfg.cache_entity):
        entity.load(cfg.cache_entity)
    else:
        entity.encode_news()
        entity.save(cfg.cache_entity)

    #  5. Evaluate 
    enc_kwargs = dict(
        tfidf_enc=tfidf, qwen_enc=qwen, entity_enc=entity,
        alpha_tfidf=cfg.alpha_tfidf,
        alpha_qwen=cfg.alpha_qwen,
        alpha_entity=cfg.alpha_entity,
    )

    results = {}
    for mode in ['tfidf', 'qwen', 'entity', 'hybrid']:
        metrics = evaluate(behaviors, all_news, mode=mode, max_samples=5000, **enc_kwargs)
        results[mode] = metrics
        print(f'[{mode.upper():8}] AUC={metrics["AUC"]} '
              f'nDCG@5={metrics["nDCG@5"]} '
              f'nDCG@10={metrics["nDCG@10"]} '
              f'({metrics["samples"]} samples)')

    run_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'alpha_tfidf':  cfg.alpha_tfidf,
            'alpha_qwen':   cfg.alpha_qwen,
            'alpha_entity': cfg.alpha_entity,
            'max_samples':  5000,
        },
        'results': results,
    }
    log_path = os.path.join('results.json')
    history  = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append(run_log)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'\nKết quả đã lưu: {log_path} ({len(history)} lần chạy)')

    #  6. Demo recommend
    print(f'\nTổng số behaviors: {len(behaviors)}')
    print('Nhập index user để xem gợi ý (0 đến {}), hoặc "q" để thoát.'.format(len(behaviors) - 1))

    while True:
        raw = input('\nNhập index user: ').strip()
        if raw.lower() == 'q':
            break
        if not raw.isdigit():
            print('Vui lòng nhập số nguyên hợp lệ.')
            continue
        idx = int(raw)
        if idx < 0 or idx >= len(behaviors):
            print(f'Index phải từ 0 đến {len(behaviors) - 1}.')
            continue

        sample = behaviors[idx]
        print(f'\n{"="*60}')
        print(f'User [{idx}]: {sample["user_id"]}')
        print(f'History:')
        for nid in sample['history'][:5]:
            n = all_news.get(nid, {})
            print(f'  - {nid}: {n.get("title", "?")}')
            abstract = n.get("abstract", "").strip()
            if abstract:
                print(f'    {abstract[:120]}...' if len(abstract) > 120 else f'    {abstract}')

        results = recommend(
            user_history=sample['history'],
            all_candidates=sample['candidates'],
            news_dict=all_news,
            mode='hybrid',
            top_k= cfg.top_k,
            **enc_kwargs,
        )

        print('Top 5 gợi ý (hybrid):')
        for rank, (nid, score) in enumerate(results, 1):
            news = all_news.get(nid, {})
            print(f'  {rank}. [{score:.3f}] {news.get("title", "?")}')
        print('='*60)

if __name__ == '__main__':
    main()
