# News Recommender System

---

## Kiến trúc

```
Bài báo (title + abstract)
        │
        ├─ TFIDFEncoder   → sparse vector 30,000 chiều
        ├─ QwenEncoder    → dense vector 768 chiều  (Qwen3-0.6B)
        └─ EntityEncoder  → dense vector 100 chiều  (Wikidata KG)
                │
                ▼
        Hybrid score = α₁×TF-IDF + α₂×Qwen + α₃×Entity
                │
                ▼
        Diversity reranking (penalty category trùng)
                │
                ▼
        Top-K gợi ý
```

---

## Cấu trúc thư mục

```
news/
├── data/
│   ├── raw/
│   │   ├── MINDsmall_train/     # behaviors.tsv, news.tsv, entity_embedding.vec
│   │   └── MINDsmall_dev/
│   └── processed/               # cache sau khi chạy lần đầu
├── config.py                    # cấu hình toàn hệ thống
├── preprocess.py                # load và cache dữ liệu
├── encoders.py                  # TFIDFEncoder, QwenEncoder, EntityEncoder
├── recommender.py               # evaluate + recommend
└── main.py                      # entry point
```

---


## Chạy

1. Load và cache 65,238 bài báo
2. Fit TF-IDF matrix
3. Download và encode toàn bộ news bằng Qwen3-0.6B
4. Tính Entity embeddings từ Wikidata


---


## Cấu hình (`config.py`)

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-----------------|---------|
| `tfidf_max_features` | 30,000 | Kích thước vocabulary |
| `qwen_model_name` | `Qwen/Qwen3-0.6B` | Model embedding |
| `qwen_max_length` | 128 | Độ dài token tối đa |
| `qwen_batch_size` | 16 | Batch size khi encode |
| `alpha_tfidf` | 0.1 | Trọng số TF-IDF trong hybrid |
| `alpha_qwen` | 0.6 | Trọng số Qwen trong hybrid |
| `alpha_entity` | 0.3 | Trọng số Entity trong hybrid |
| `top_k` | 5 | Số bài gợi ý |

---

## 3 Encoders

### TFIDFEncoder
Biểu diễn bài báo bằng vector TF-IDF (unigram + bigram). User vector được tính bằng weighted mean với recency weighting — bài đọc gần nhất có trọng số cao hơn.

### QwenEncoder
Dùng Qwen LLM để tạo dense embedding qua mean pooling. Hiểu được ngữ nghĩa và từ đồng nghĩa. User vector cũng dùng recency weighting.

### EntityEncoder
Dựa vào Wikidata Knowledge Graph có sẵn trong MIND dataset. Mỗi bài báo được biểu diễn bằng trung bình embedding của các thực thể (người, địa danh, tổ chức) xuất hiện trong title và abstract.

---
s
## Metrics
- **AUC**: tỉ lệ cặp (bài click, bài không click) được xếp đúng thứ tự.
- **nDCG@K**: đo chất lượng xếp hạng top-K — bài click ở vị trí càng cao càng tốt.
