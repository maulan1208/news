# Collaborative Filtering (CF)
Những tin được cùng nhóm người đọc thì liên quan đến nhau". Không dùng nội dung bài viết, chỉ dùng lịch sử tương tác user–tin.

Có 2 thuật toán cùng chung interface

| File | Thuật toán | Kiểu |
|------|-----------|------|
| `item.py` | Item-based CF | memory-based (không train) |
| `mf.py` | Matrix Factorization | model-based (học vector ẩn) |

## Mô tả

1. **`config.py`**: đường dẫn dữ liệu + tham số. Đọc trước để biết file nào nằm ở đâu.
2. **`preprocess.py`**: biến `behaviors.tsv` thành ma trận thưa user × news. Hiểu dữ liệu đầu vào.
3. **`mf.py`** (hoặc `item.py`): cách model học và chấm điểm.
4. **`recommend.py`** / **`evaluate.py`**: dùng model để gợi ý và đo chất lượng.

- docs/cf/cf.md (tổng quan CF) và docs/cf/mf.md (Matrix Factorization).

## Luồng chạy

```
behaviors.tsv
    │  preprocess.py  
    ▼
user_item.npz + *.pkl   ma trận thưa user × news
    │  mf.py            học vector ẩn P, Q
    ▼
mf_factors.npz
    │  recommend.py / evaluate.py
    ▼
gợi ý top-K   /   đánh giá  → results.json
```

## Cấu trúc

```
cores/cf/
├── config.py        đường dẫn dữ liệu + tham số
├── preprocess.py    behaviors.tsv → ma trận thưa user × news
├── item.py          ItemCF (memory-based)
├── mf.py            Matrix Factorization (model-based)
├── recommend.py     demo gợi ý theo history
├── evaluate.py      đánh giá: AUC, MRR, nDCG@5, nDCG@10
└── results.json     lịch sử kết quả đánh giá

data/                
├── raw/             
└── processed/cf/    user_item.npz, *.pkl, mf_factors.npz
```
