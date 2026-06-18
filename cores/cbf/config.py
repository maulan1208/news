import os

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_CBF = os.path.join(DATA_DIR, 'processed', 'cbf')

class Config:
    # Đường dẫn data (dùng chung ở root, không phụ thuộc thư mục chạy)
    raw_dir       = os.path.join(DATA_DIR, 'raw')
    processed_dir = PROCESSED_CBF
    train_dir     = os.path.join(DATA_DIR, 'raw', 'MINDsmall_train')
    dev_dir       = os.path.join(DATA_DIR, 'raw', 'MINDsmall_dev')

    # Cache paths
    cache_news    = os.path.join(PROCESSED_CBF, 'news_processed.pkl')
    cache_tfidf   = os.path.join(PROCESSED_CBF, 'tfidf_model.pkl')
    cache_qwen    = os.path.join(PROCESSED_CBF, 'qwen_embeddings.npy')
    cache_entity  = os.path.join(PROCESSED_CBF, 'entity_news_embeddings.npy')

    # TF-IDF
    tfidf_max_features = 30000

    # Qwen
    qwen_model_name = 'Qwen/Qwen3-0.6B'
    qwen_max_length = 256
    qwen_batch_size = 16
    device          = 'cuda'

    # Recommend
    top_k           = 5
    alpha_tfidf     = 0.3   # trọng số TF-IDF trong hybrid
    alpha_qwen      = 0.6   # trọng số Qwen trong hybrid
    alpha_entity    = 0.1   # trọng số Entity trong hybrid