# News Recommendation — CF vs CBF

Hệ gợi ý tin tức trên bộ dữ liệu **MIND** (MINDsmall), cài đặt **hai hướng tiếp cận** độc lập để so sánh:

- `cores/cbf/` — **Content-Based Filtering** (lọc theo nội dung bài báo)
- `cores/cf/`  — **Collaborative Filtering** (lọc cộng tác theo hành vi đọc)

Dữ liệu dùng chung ở `data/raw` (xem [data/](#cấu-trúc-thư-mục)).

---

## TL;DR — khác nhau ở đâu?

| | **CBF** (content-based) | **CF** (collaborative) |
|---|---|---|
| **Dựa vào** | Nội dung bài: tiêu đề, abstract, entity | Hành vi: ai đã đọc tin nào |
| **Câu hỏi trả lời** | "Tin nào *giống về nội dung* với tin bạn đã đọc?" | "Tin nào được *những người giống bạn* đọc?" |
| **Biểu diễn tin** | Vector nội dung (TF-IDF / Qwen / Entity) | Vector trong không gian user (ai đọc tin đó) |
| **Cold-start tin mới** | ✅ Tốt — có nội dung là gợi ý được ngay | ❌ Kém — chưa ai đọc → điểm 0 |
| **Cold-start dữ liệu thưa** | ✅ Không cần tương tác | ❌ Cần đủ lượng tương tác |
| **Tính bất ngờ (serendipity)** | Thấp — dễ trùng chủ đề (cần rerank) | Cao — khám phá qua cộng đồng |
| **Chi phí** | Nặng — cần LLM embeddings + GPU | Nhẹ — chỉ phép nhân ma trận thưa |
| **Phụ thuộc** | Chất lượng model ngôn ngữ / knowledge base | Lượng & chất tương tác người dùng |

> Một câu: **CBF nhìn vào *bài báo*, CF nhìn vào *người đọc*.**

---

## CBF — Content-Based Filtering (`cores/cbf`)

Gợi ý dựa trên **độ giống về nội dung** giữa tin ứng viên và các tin trong lịch sử đọc. Gồm **3 encoder**, có thể chạy riêng hoặc kết hợp (hybrid):

| Encoder | Nội dung dùng | Cách biểu diễn |
|---|---|---|
| **TF-IDF** | title + abstract | Bag-of-words, n-gram (1,2), `max_features=30000`, sublinear TF |
| **Qwen** | title + abstract | Embedding ngữ nghĩa từ `Qwen3-0.6B` (mean pooling), bắt được ý nghĩa sâu |
| **Entity** | thực thể Wikidata trong bài | Trung bình vector entity (MIND `entity_embedding.vec`, 100 chiều) |

**Luồng tính điểm:**
1. **User vector** = trung bình *có trọng số* các vector tin trong history (bài đọc gần nhất → trọng số cao hơn).
2. **Score** = `cosine(vector_ứng_viên, user_vector)`.
3. **Hybrid**: chuẩn hoá z-score điểm của từng encoder rồi cộng theo trọng số `alpha_tfidf / alpha_qwen / alpha_entity`.
4. **Re-rank** (`recommender.py`):
   - *Novelty* — loại bỏ tin đã đọc khỏi ứng viên.
   - *Diversity* — greedy rerank, phạt (`penalty=0.8`) tin trùng category với tin đã chọn.

**Chạy:**
bash
cd cores/cbf
python main.py          # build/load encoders → evaluate (tfidf/qwen/entity/hybrid) → demo

---

## CF — Collaborative Filtering (`cores/cf`)

**Item-based CF**: hai tin được *cùng một nhóm user* đọc thì giống nhau — không dùng nội dung bài.

**Luồng tính điểm:**
1. `preprocess.py` — dựng ma trận thưa **user × news** (nhị phân, 1 = đã đọc) từ `behaviors.tsv`.
2. `item.py` (`ItemCF`) — chuyển vị thành **news × user**, mỗi tin là 1 vector trong không gian user; L2-normalize để *dot product = cosine*; đưa lên GPU (torch sparse).
3. Điểm của tin *i* = Σ `cosine(i, h)` với mọi tin *h* trong history → "tin *i* giống tổng cộng bao nhiêu với những gì bạn đã đọc". Toàn bộ là 2 phép nhân ma trận thưa.

**Chạy:**
```bash
cd cores/cf
python preprocess.py    # behaviors.tsv → data/processed/cf/ (user_item.npz, *2idx.pkl)
python recommend.py --history "N1 N2 N3"   # demo gợi ý
python evaluate.py --limit 5000            # đo AUC / MRR / nDCG trên dev
```

---

## Đánh giá

Cả hai dùng chung bộ chỉ số trên tập `dev` (impression có cả tin click & không click):

- **AUC** — phân biệt tin click vs không click.
- **nDCG@5 / @10** — chất lượng xếp hạng top-K.
- **MRR** (CF) — thứ hạng tin click đúng đầu tiên.

Kết quả ghi ra `results/cbf.json` và `results/cf.json`.

---

## Cấu trúc thư mục

```
news/
├── cores/
│   ├── cbf/   # config, preprocess, encoders, recommender, main
│   └── cf/    # config, preprocess, item (ItemCF), recommend, evaluate
├── data/                      # KHÔNG commit (đã .gitignore)
│   ├── raw/MINDsmall_{train,dev}/    # nguồn dùng chung
│   └── processed/{cbf,cf}/           # cache theo từng model
├── docs/{cbf,cf}/             # tài liệu & notebook giải thích
├── results/{cbf,cf}.json      # kết quả đánh giá
└── notebook/                  # notebook thử nghiệm
```

---

## Khi nào dùng cái nào?

- **Tin mới liên tục, ít dữ liệu hành vi** → CBF (không bị cold-start tin mới).
- **Nhiều người dùng, nhiều tương tác, muốn gợi ý bất ngờ** → CF.
- **Thực tế** → kết hợp cả hai (hybrid system): CF cho tin phổ biến, CBF lấp chỗ cold-start.
