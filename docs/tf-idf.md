
# TF-IDF
Là phương pháp biểu diễn văn bản dưới dạng vector, đánh giá tầm quan trọng của từ trong document dựa trên tần suất xuất hiện (TF) và độ hiếm của từ trên toàn bộ kho tài liệu (IDF). TF-IDF giúp tìm kiếm document tương tự bằng cách so sánh vector TF-IDF của chúng.

### Token
Đơn vị từ sau khi tiền xử lý văn bản (lowercase, bỏ ký tự đặc biệt, tách theo khoảng trắng).

### DF — Document Frequency
Số lượng tài liệu có chứa từ `t` (mỗi tài liệu chỉ đếm 1 lần dù từ xuất hiện nhiều lần):

```
df(t) = số document chứa từ t
```

Từ phổ biến (stopword) có DF rất cao — ví dụ `the` xuất hiện trong 39.330/51.282 document; ngược lại tên riêng hiếm gặp có DF = 1.

### TF — Term Frequency
Tần suất xuất hiện của từ `t` trong document `d`, chuẩn hóa theo tổng số từ của document đó:

```
tf(t, d) = count(t trong d) / tổng số từ trong d
```

Chuẩn hóa giúp document dài không lấn át document ngắn.

### IDF — Inverse Document Frequency

Đo độ "hiếm" (mức độ mang thông tin) của từ trên toàn bộ kho tài liệu. Dùng công thức **smooth IDF** (giống scikit-learn):

```
idf(t) = ln((N + 1) / (df(t) + 1)) + 1
```

Trong đó `N` là tổng số document. Việc cộng 1 vào tử và mẫu (smoothing) tránh chia cho 0 với từ chưa từng xuất hiện; cộng 1 ở cuối đảm bảo IDF luôn dương (từ xuất hiện ở mọi document vẫn không bị triệt tiêu hoàn toàn).

- Từ hiếm (df = 1) → IDF cao nhất ≈ **11,15**
- Từ phổ biến nhất (`the`, df = 39.330) → IDF thấp nhất ≈ **1,27**

### TF-IDF

Trọng số cuối cùng của từ `t` trong document `d`:

```
tfidf(t, d) = tf(t, d) × idf(t)
```

Ý nghĩa: từ có giá trị TF-IDF cao khi nó **xuất hiện nhiều trong document này** nhưng **hiếm trong các document khác** → đặc trưng cho nội dung của document.

### Biểu diễn sparse (thưa)

Vì mỗi document chỉ chứa vài chục từ trong số ~55.000 từ của vocabulary, vector TF-IDF được lưu dạng **dict thưa** `{word_id: tfidf_value}` thay vì mảng dày 55.000 chiều — tiết kiệm bộ nhớ và tăng tốc tính toán.


## Luồng hoạt động

```
 Ghép title + abstract ──► Tiền xử lý (tokenize)
                            │
                            ▼
              Đếm DF ──► Lọc vocabulary ──► word_to_index
                            │
                            ▼
                       Tính IDF
                            │
 Tính TF (từng document) ───┤
                            ▼
              TF-IDF sparse cho từng document
```



## 4. Kết quả minh họa

### 4.1. TF-IDF của một document (Document 5)

> *"Should NFL be able to fine players for criticizing officiating? ..."*

Top từ có TF-IDF cao nhất:

| word | tfidf |
|------|-------|
| criticizing | 0.5516 |
| officiating | 0.5201 |
| players | 0.3381 |
| nfl | 0.3089 |
| fines | 0.2593 |

