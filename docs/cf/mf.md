# Matrix Factorization (MF)
Matrix Factorization là phương pháp model-based của Collaborative Filtering: thay vì dùng trực tiếp ma trận user–item khổng lồ (như memory-based), nó học một biểu diễn cô đọng cho mỗi user và mỗi item, rồi dùng biểu diễn đó để dự đoán.

Ý tưởng: phân tách ma trận tương tác `R` (user × item, rất thưa) thành tích của hai ma trận chiều thấp `P` và `Q`. Mỗi user và item được mô tả bằng một vector yếu tố ẩn (latent factors) `k` chiều.

```
R  ≈  P @ Qᵀ
P: (n_users × k)   — mỗi DÒNG là 1 user
Q: (n_items × k)   — mỗi DÒNG là 1 item
```

Vì `k` nhỏ (ở đây `k = 64`, so với ~40.000 tin), mô hình không thể "nhớ vẹt" từng ô mà buộc phải tìm ra cấu trúc chung — nhờ đó tự điền được các ô trống (tin user chưa đọc) một cách hợp lý.

---

## Các khái niệm

### Yếu tố ẩn (latent factor)
Mỗi user và item được biểu diễn bằng một vector `k` chiều. Có thể hình dung mỗi chiều là một "đặc trưng ẩn" (ví dụ với phim: mức hành động, mức lãng mạn...). Mô hình tự khám phá các chiều này từ dữ liệu, không gán nhãn trước.

```
user u -> p_u  (gu của user với từng yếu tố)
item i -> q_i  (item chứa từng yếu tố tới đâu)
```

### Dự đoán = tích vô hướng
Mức độ user `u` hợp với item `i` = độ khớp giữa hai vector:

```
score(u, i) = p_u · q_i = Σ_f p_u[f] · q_i[f]
```

User mê yếu tố nào, gặp item giàu yếu tố đó -> tích vô hướng cao -> dự đoán "thích".

### Implicit feedback
Dữ liệu MIND không có điểm 1–5, chỉ có tín hiệu "đã đọc" (= 1). Nếu chỉ học trên các ô = 1, mô hình học mẹo lười: đoán 1 ở mọi nơi -> vô dụng.

### Negative sampling
Cách chữa implicit feedback: với mỗi cặp dương `(u, i⁺)` (đã đọc), bốc ngẫu nhiên một tin âm `i⁻` mà user chưa đọc, và dạy mô hình:

```
score(u, i⁺) -> 1 (kéo lên)
score(u, i⁻) -> 0 (đẩy xuống)
```

Có cả lực kéo lên lẫn đẩy xuống thì vector mới tách bạch được tin thích / không thích.

## Học P, Q

### Hàm mất mát
Chỉ so khớp trên các ô đã biết, cộng thêm phạt L2 (regularization) để chống overfit:

```
L = Σ e²  +  λ (‖p_u‖² + ‖q_i‖²)
```

với sai số tại một ô là `e = target − p_u · q_i`.

### Gradient (đạo hàm)
Đạo hàm của `L` theo từng vector:

```
∂L/∂p_u = −2 e q_i + 2λ p_u
∂L/∂q_i = −2 e p_u + 2λ q_i
```

### Gradient descent
Đi ngược hướng đạo hàm một bước nhỏ `lr` (gộp hằng số 2 vào `lr`):

```
p_u ← p_u + lr (e q_i − λ p_u)
q_i ← q_i + lr (e p_u − λ q_i)
```

Đọc cho có nghĩa:
- `e q_i`: đoán thiếu (`e > 0`) -> kéo `p_u` về phía `q_i` để lần sau đoán cao hơn.
- `−λ p_u`: kéo vector co lại, không cho phình to (regularization).

### Batch & Epoch
- Batch: một nhóm cặp xử lý chung trong 1 bước cập nhật (ở đây 8.192 cặp), tính bằng phép toán ma trận thay vì lặp Python từng cặp.
- Epoch: một lượt duyệt hết toàn bộ cặp dương. ~1,1 triệu cặp ÷ 8.192 ≈ 141 batch/epoch; lặp 20 epoch.


## Luồng hoạt động

```
 Khởi tạo P, Q ngẫu nhiên nhỏ
            │
   ┌──> mỗi EPOCH (×20) ─────────────────────────┐
   │     xáo trộn các cặp dương                  │
   │                                             │
   │   ┌──> mỗi BATCH (8192 cặp) ──────────┐     │
   │   │  1. negative sampling: bốc i⁻     │     │
   │   │  2. dự đoán   pred = p·q          │     │
   │   │  3. sai số    e = target − pred   │     │
   │   │  4. gradient  e·q − λp            │     │
   │   │  5. cập nhật  p ← p + lr·grad     │     │
   │   └───────────────────────────────────┘     │
   │     in RMSE (giảm dần)                      │
   └─────────────────────────────────────────────┘
            │
        P, Q  ->  lưu mf_factors.npz
```

### Chấm điểm (suy luận)
Pipeline chỉ đưa vào history (danh sách tin đã đọc), không có `user_id` -> không tra được `p_u`. Nên ta dựng "gu" user từ chính các tin trong history:

```
profile = trung bình Q của các tin trong history
score(candidate) = profile · Q[candidate]
```

`P` chỉ cần khi train để `Q` học ra không gian ẩn tốt; lúc chấm điểm chỉ dùng `Q`.

## Tham số (config.py)

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `MF_DIM` (k) | 64 | số chiều không gian ẩn |
| `MF_EPOCHS` | 20 | số vòng lặp qua toàn bộ tương tác |
| `MF_LR` | 0.05 | learning rate (độ dài mỗi bước cập nhật) |
| `MF_REG` (λ) | 0.01 | hệ số phạt L2 |
| `MF_BATCH_SIZE` | 8192 | số cặp mỗi bước cập nhật |

> Lưu ý `lr`: quá nhỏ -> hội tụ chậm; quá lớn -> nhảy vọt qua đáy, RMSE dao động/phân kỳ.

## Ưu / Nhược

Ưu điểm
- Nén chiều (`k` nhỏ) -> khái quát hóa tốt, đoán được cả ô trống.
- Vector nhẹ (64 chiều) -> chấm điểm nhanh, tiết kiệm bộ nhớ so với vector thưa của memory-based.

Nhược điểm
- Phải train, có nhiều siêu tham số (`k`, `lr`, `λ`, `epochs`) cần tinh chỉnh.
- Các chiều ẩn khó diễn giải ý nghĩa cụ thể.
- Cold start: user/tin mới chưa có trong dữ liệu train thì chưa có vector.


## So với Memory-Based (ItemCF)

| | ItemCF (memory-based) | MF (model-based) |
|---|---|---|
| Biểu diễn 1 tin | vector thưa ~50.000 chiều | vector đặc 64 chiều |
| Vector đến từ đâu | lấy thẳng từ ma trận, không học | học bằng gradient descent |
| Pha train | không | có (20 epoch) |
| Chấm điểm | profile · vector thưa (cosine) | profile · Q (dot) |

Cách chấm điểm gần như giống nhau (đều dựng profile từ history); khác biệt cốt lõi là MF có học biểu diễn nén, còn ItemCF dùng vector thô lấy sẵn.
