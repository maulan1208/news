# Content-based
Thuật toán content-based recommendation dựa trên nội dung của các item (bài báo) để đưa ra gợi ý. Ý tưởng là nếu người dùng đã click vào một bài báo nào đó, thì những bài báo có nội dung tương tự sẽ có khả năng được người dùng quan tâm.

## Ưu điểm
- Không cần dữ liệu người dùng khác (không bị cold-start user)
- Dễ giải thích: có thể chỉ ra lý do tại sao một bài báo được gợi ý (dựa trên nội dung tương tự)
- Có thể gợi ý cho người dùng mới chỉ dựa trên nội dung bài báo mà họ đã click

## Nhược điểm
- Gợi ý thiếu đa dạng: nếu người dùng chỉ click vào một loại bài báo, thì sẽ chỉ được gợi ý những bài báo tương tự, thiếu sự đa dạng
- Không thể gợi ý những bài báo có nội dung khác nhưng vẫn phù hợp với sở thích của người dùng 

## Cách hoạt động
1. Biểu diễn bài báo dưới dạng vector: Mỗi bài báo được biểu diễn bằng một vector đặc trưng. Ví dụ, có thể dùng TF-IDF để biểu diễn bài báo dưới dạng vector sparse, hoặc dùng một mô hình ngôn ngữ lớn (LLM) như Qwen để tạo dense embedding.
2. Xây dựng user profile: Dựa trên lịch sử click của người dùng, xây dựng một vector đại diện cho sở thích của người dùng bằng cách trung bình hóa các vector bài báo mà người dùng đã click.
3. Tính độ tương đồng: Sử dụng cosine similarity
4. Xếp hạng và gợi ý: Sắp xếp các bài báo theo độ tương đồng với user profile và gợi ý top-K.

## Thuật toán sử dụng
- TF-IDF: Biểu diễn bài báo bằng vector TF-IDF (unigram + bigram). User vector được tính bằng weighted mean với recency weighting — bài đọc gần nhất có trọng số cao hơn.
- QwenEncoder: Dùng Qwen LLM để tạo dense embedding qua mean pooling. Hiểu được ngữ nghĩa và từ đồng nghĩa. User vector cũng dùng recency weighting.
- EntityEncoder: Dựa vào Wikidata Knowledge Graph có sẵn trong MIND dataset. Mỗi bài báo được biểu diễn bằng trung bình embedding của các thực thể (người, địa danh, tổ chức) xuất hiện trong title và abstract.
- Cosine similarity: Đo độ tương đồng giữa user vector và news vector để xếp hạng và gợi ý.

## Cấu trúc thư mục

news/
├── data/
│   ├── raw/
│   │   ├── MINDsmall_train/     # behaviors.tsv, news.tsv, entity_embedding.vec
│   │   └── MINDsmall_dev/
│   └── processed/               # cache sau khi chạy
├── config.py                    # cấu hình toàn hệ thống
├── preprocess.py                # load và cache dữ liệu
├── encoders.py                  # TFIDFEncoder, QwenEncoder, EntityEncoder
├── recommender.py               # evaluate + recommend
└── main.py                      # entry point

## Data
MIND dataset: 65,238 bài báo với title, abstract và entity (thực thể) được trích xuất từ Wikidata. Có sẵn train/dev split với lịch sử click của người dùng.
- Mindsmall_train: 100,000 users, 65,238 news
- Mindsmall_dev: 10,000 users, 65,238 news

## Kết quả
Đánh giá mô hình 
| Encoder        | AUC   |
|----------------|-------|
| TF-IDF         | 0.5796  |
| Qwen3-0.6B     | 0.6057  |
| Entity         | 0.5478  |          
| Hybrid         | 0.6133  |  

Từ kết quả trên có thể thấy rằng Qwen3-0.6B có hiệu suất tốt hơn TF-IDF và Entity, cho thấy khả năng hiểu ngữ nghĩa của LLM giúp cải thiện chất lượng gợi ý. Kết hợp cả 3 encoder trong mô hình hybrid mang lại hiệu suất tốt nhất, chứng tỏ rằng mỗi encoder đóng góp thông tin bổ sung giúp cải thiện kết quả cuối cùng.

