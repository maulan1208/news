import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def cosine_similarity(mat, vec):
    """
    mat: [N, D]  — N vectors
    vec: [D]     — 1 vector
    Trả về: [N]  — cosine similarity giữa từng hàng của mat với vec
    """
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    return mat_norm @ vec_norm

class TFIDFEncoder:
    def __init__(self, cfg):
        self.cfg        = cfg
        self.vectorizer = None
        self.news_ids   = []
        self.matrix     = None   # sparse [N, vocab]

    def fit_transform(self, news_dict):
        self.news_ids = list(news_dict.keys())
        self._id2row  = {nid: i for i, nid in enumerate(self.news_ids)}
        texts = [news_dict[nid]['text'] for nid in self.news_ids]

        self.vectorizer = TfidfVectorizer(
            max_features=self.cfg.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            stop_words='english',
            max_df=0.95,
        )
        self.matrix = self.vectorizer.fit_transform(texts)  # sparse [N, vocab]
        print(f'TF-IDF: {self.matrix.shape[0]} news, vocab {self.matrix.shape[1]}')

    def _lookup(self, news_ids):
        rows = [self._id2row[nid] for nid in news_ids]
        return self.matrix[rows].toarray()                 

    def get_user_vector(self, history_ids):
        valid = [nid for nid in history_ids if nid in self._id2row]
        if not valid:
            return None
        vecs = self._lookup(valid)                         

        # bài cuối history = đọc gần nhất → trọng số cao hơn
        weights = np.arange(1, len(valid) + 1, dtype=np.float32)
        weights /= weights.sum()
        return weights @ vecs                              

    def score(self, history_ids, candidate_ids, news_dict):
        user_vec = self.get_user_vector(history_ids)
        if user_vec is None:
            return np.zeros(len(candidate_ids))

        valid_cands = [nid for nid in candidate_ids if nid in self._id2row]
        if not valid_cands:
            return np.zeros(len(candidate_ids))

        cand_vecs  = self._lookup(valid_cands)             
        scores_map = dict(zip(valid_cands, cosine_similarity(cand_vecs, user_vec)))
        return np.array([scores_map.get(nid, 0.0) for nid in candidate_ids])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'TF-IDF đã lưu: {path}')

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            enc = pickle.load(f)
        print(f'TF-IDF đã load: {path}')
        return enc
    
class QwenEncoder:
    def __init__(self, cfg):
        self.cfg     = cfg
        self.embeds  = {}   

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = self.cfg.device
        print(f'Load {self.cfg.qwen_model_name}...')
        tok   = AutoTokenizer.from_pretrained(self.cfg.qwen_model_name)
        model = AutoModel.from_pretrained(
            self.cfg.qwen_model_name,
            torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        ).to(device).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok, model

    def encode_news(self, news_dict):
        import torch
        from tqdm import tqdm

        tok, model = self._load_model()
        device     = self.cfg.device

        news_ids = list(news_dict.keys())
        texts    = [news_dict[nid]['text'] for nid in news_ids]
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.cfg.qwen_batch_size),
                          desc='Encoding news'):
                batch = texts[i : i + self.cfg.qwen_batch_size]
                enc   = tok(batch, max_length=self.cfg.qwen_max_length,
                            padding=True, truncation=True,
                            return_tensors='pt').to(device)
                out  = model(**enc)
                # Mean pooling (bỏ pad tokens)
                hidden = out.last_hidden_state             
                mask   = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                all_embs.append(pooled.cpu().float().numpy())

        embeddings = np.concatenate(all_embs, axis=0)     
        self.embeds = dict(zip(news_ids, embeddings))
        print(f'Qwen: {len(self.embeds)} news encoded, dim={embeddings.shape[1]}')

    def score(self, history_ids, candidate_ids):
        valid_hist = [nid for nid in history_ids if nid in self.embeds]
        if not valid_hist:
            return np.zeros(len(candidate_ids))

        hist_vecs = np.stack([self.embeds[nid] for nid in valid_hist])
        # user_vec  = hist_vecs.mean(axis=0)
        # user_vec  = user_vec / (np.linalg.norm(user_vec) + 1e-8) 
        weights   = np.arange(1, len(valid_hist) + 1, dtype=np.float32)
        weights   /= weights.sum()
        user_vec  = (weights[:, None] * hist_vecs).sum(axis=0)
        valid_mask = np.array([nid in self.embeds for nid in candidate_ids])
        valid_cands = [nid for nid, m in zip(candidate_ids, valid_mask) if m]
        if not valid_cands:
            return np.zeros(len(candidate_ids))

        cand_vecs = np.stack([self.embeds[nid] for nid in valid_cands])  

        scores = np.zeros(len(candidate_ids))
        scores[valid_mask] = cosine_similarity(cand_vecs, user_vec)     
        return scores

    def save(self, path):
        np.save(path, self.embeds)
        print(f'Qwen embeddings đã lưu: {path}')

    def load(self, path):
        self.embeds = np.load(path, allow_pickle=True).item()
        print(f'Qwen embeddings đã load: {len(self.embeds)} news')


class EntityEncoder:
    def __init__(self, cfg):
        self.cfg          = cfg
        self.entity_vecs  = {}   
        self.embeds       = {}   

    def _load_entity_vecs(self):
        for split_dir in [self.cfg.train_dir, self.cfg.dev_dir]:
            path = f'{split_dir}/entity_embedding.vec'
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    self.entity_vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
        print(f'Entity vecs: {len(self.entity_vecs)} entities, dim=100')

    def _parse_entities(self, news_file):
        import json
        import pandas as pd
        df = pd.read_csv(
            news_file, sep='\t', header=None,
            names=['id', 'category', 'subcategory', 'title',
                   'abstract', 'url', 'title_entities', 'abstract_entities']
        )
        news_entities = {}
        for _, row in df.iterrows():
            ids = []
            for col in ['title_entities', 'abstract_entities']:
                try:
                    ents = json.loads(row[col]) if pd.notna(row[col]) else []
                    ids += [e['WikidataId'] for e in ents]
                except Exception:
                    pass
            news_entities[row['id']] = ids
        return news_entities

    def encode_news(self):
        self._load_entity_vecs()

        all_news_entities = {}
        for split_dir in [self.cfg.train_dir, self.cfg.dev_dir]:
            all_news_entities.update(self._parse_entities(f'{split_dir}/news.tsv'))

        covered = 0
        for news_id, entity_ids in all_news_entities.items():
            valid = [self.entity_vecs[eid] for eid in entity_ids if eid in self.entity_vecs]
            if valid:
                self.embeds[news_id] = np.mean(valid, axis=0).astype(np.float32)
                covered += 1
            else:
                self.embeds[news_id] = np.zeros(100, dtype=np.float32)

        self._valid = {nid for nid, v in self.embeds.items() if np.any(v)}
        print(f'Entity: {covered}/{len(all_news_entities)} news có entity')

    def score(self, history_ids, candidate_ids):
        valid_set  = getattr(self, '_valid', None) or {
            nid for nid, v in self.embeds.items() if np.any(v)
        }
        valid_hist = [nid for nid in history_ids if nid in valid_set]
        if not valid_hist:
            return np.zeros(len(candidate_ids))

        hist_vecs = np.stack([self.embeds[nid] for nid in valid_hist])
        user_vec  = hist_vecs.mean(axis=0)                              

        valid_mask = np.array([nid in valid_set for nid in candidate_ids])
        valid_cands = [nid for nid, m in zip(candidate_ids, valid_mask) if m]
        if not valid_cands:
            return np.zeros(len(candidate_ids))

        cand_vecs = np.stack([self.embeds[nid] for nid in valid_cands]) 

        scores = np.zeros(len(candidate_ids))
        scores[valid_mask] = cosine_similarity(cand_vecs, user_vec)    
        return scores

    def save(self, path):
        np.save(path, self.embeds)
        print(f'Entity embeddings đã lưu: {path}')

    def load(self, path):
        self.embeds = np.load(path, allow_pickle=True).item()
        self._valid = {nid for nid, v in self.embeds.items() if np.any(v)}
        print(f'Entity embeddings đã load: {len(self.embeds)} news')