import os
import pickle
import pandas as pd

def load_news(news_file):
    df = pd.read_csv(
        news_file, sep='\t', header=None,
        names=['id', 'category', 'subcategory', 'title',
               'abstract', 'url', 'title_entities', 'abstract_entities']
    )
    df['abstract'] = df['abstract'].fillna('')
    df['title']    = df['title'].fillna('')

    # Ghép title + abstract thành 1 text duy nhất
    df['text'] = df['title'] + ' ' + df['abstract']

    news = {}
    for _, row in df.iterrows():
        news[row['id']] = {
            'title':    row['title'],
            'abstract': row['abstract'],
            'text':     row['text'],
            'category': row['category'],
        }
    return news

def load_behaviors(behaviors_file):
    df = pd.read_csv(
        behaviors_file, sep='\t', header=None,
        names=['imp_id', 'user_id', 'time', 'history', 'impressions']
    )
    records = []
    for _, row in df.iterrows():
        history = (str(row['history']).split()
                   if pd.notna(row['history']) else [])
        impressions = str(row['impressions']).split()
        candidates = [i.split('-')[0] for i in impressions]
        labels     = [int(i.split('-')[1]) for i in impressions]
        records.append({
            'user_id':    row['user_id'],
            'history':    history,
            'candidates': candidates,
            'labels':     labels,
        })
    return records

def load_and_cache(cfg):
    cache_path = os.path.join(cfg.processed_dir, 'news_processed.pkl')
    os.makedirs(cfg.processed_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print('Load news từ cache...')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print('Đọc và xử lý news...')
    train_news = load_news(f'{cfg.train_dir}/news.tsv')
    dev_news   = load_news(f'{cfg.dev_dir}/news.tsv')
    all_news   = {**train_news, **dev_news}

    with open(cache_path, 'wb') as f:
        pickle.dump(all_news, f)

    print(f'Đã lưu {len(all_news)} news vào {cache_path}')
    return all_news