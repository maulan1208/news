import os

class Config:
    # Đường dẫn data
    raw_dir       = './data/raw'
    processed_dir = './data/processed'
    train_dir     = './data/raw/MINDsmall_train'
    dev_dir       = './data/raw/MINDsmall_dev'

    # Cache paths
    cache_news    = './data/processed/news_processed.pkl'
    cache_tfidf   = './data/processed/tfidf_model.pkl'
    cache_qwen    = './data/processed/qwen_embeddings.npy'
    cache_entity  = './data/processed/entity_news_embeddings.npy'

    # TF-IDF
    tfidf_max_features = 30000

    # Qwen
    qwen_model_name = 'Qwen/Qwen3-0.6B'
    qwen_max_length = 128
    qwen_batch_size = 16
    device          = 'cuda'

    # Recommend
    top_k           = 5
    alpha_tfidf     = 0.1   # trọng số TF-IDF trong hybrid
    alpha_qwen      = 0.6   # trọng số Qwen trong hybrid
    alpha_entity    = 0.3   # trọng số Entity trong hybrid