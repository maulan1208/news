## Collaborative Filtering
collaborative filtering đưa ra dự đoán dựa trên hành vi của người dùng khác có cùng sở thích tương tự

Sử dụng ma trận user-item để theo dõi những gì người dùng thích, xem hoặc mua. Sau đó, nó xác định sự tương đồng giữa người dùng hoặc sản phẩm và đề xuất dựa trên các mẫu này

Có hai loại chính:
- **User-Based (Dựa trên người dùng):** Gợi ý sản phẩm dựa trên những gì người dùng khác có gu tương tự đã thích. Nếu bạn và người dùng khác thích cùng loại phim, bạn sẽ được gợi ý những phim họ đánh giá cao.
- **Item-Based (Dựa trên sản phẩm):** Hệ thống gợi ý những sản phẩm tương tự với những gì bạn đã thích trước đó. Nếu bạn đã xem một bộ phim, nó có thể gợi ý các bộ phim khác có chủ đề hoặc thể loại tương tự.


## Các thuật toán CF
Chia thành hai loại chính: 
- memory-based
- model-based

### Memory-Based
Cách tiếp cận trực tiếp này đưa ra gợi ý sử dụng trực tiếp ma trận user-item. Được gọi là "memory-based" vì hệ thống dựa vào toàn bộ ma trận để tìm người dùng hoặc sản phẩm tương tự.
- user-user: Tìm người dùng tương tự với người dùng mục tiêu dựa trên sở thích chung.
- item-item: Xác định các sản phẩm tương tự với những sản phẩm người dùng đã tương tác.

### Model-Based
Xây dựng mô hình dự đoán dựa trên ma trận user-item. Hiệu quả và có khả năng mở rộng hơn, đặc biệt với tập dữ liệu lớn.
- Matrix Factorization: Phân tách ma trận user-item thành các ma trận chiều thấp hơn, xác định các yếu tố ẩn ảnh hưởng đến sở thích (ví dụ: Singular Value Decomposition - SVD).
- Neural Networks: Mạng nơ-ron và deep learning có thể nắm bắt các mẫu phức tạp hơn, cải thiện độ chính xác của dự đoán.


## Cách CF hoạt động
1. **Thu thập dữ liệu:** Hệ thống thu thập dữ liệu tương tác người dùng (đánh giá, lượt xem, mua hàng).
2. **Xây dựng ma trận user-item:** Tạo ma trận với người dùng ở hàng và sản phẩm ở cột, điền vào các giá trị tương tác.
3. **Tính toán sự tương đồng:** Sử dụng các phương pháp như cosine similarity hoặc Pearson correlation để đo lường sự tương đồng giữa người dùng hoặc sản phẩm.
4. **Dự đoán:** Dựa trên sự tương đồng, hệ thống dự đoán mức độ người dùng sẽ thích một sản phẩm chưa tương tác.
5. **Gợi ý:** Đưa ra danh sách các sản phẩm được gợi ý dựa trên dự đoán.

## Ưu điểm
- **Cá nhân hóa quy mô lớn:** Phân tích dữ liệu lịch sử để gợi ý nội dung phù hợp dựa trên mẫu của người dùng tương tự.
- **Không cần metadata sản phẩm:** Không cần hiểu biết sâu về sản phẩm được gợi ý; hoạt động thuần túy với dữ liệu tương tác người dùng.
- **Gợi ý động:** Hệ thống liên tục cập nhật gợi ý khi người dùng tương tác thêm với nội dung.
- **Linh hoạt đa ngành:** Có thể áp dụng trên nhiều lĩnh vực: bán lẻ, giải trí, truyền thông, y tế.

## Nhược điểm
- **Data Sparsity (Dữ liệu thưa):** Ma trận user-item thường có nhiều giá trị thiếu, làm giảm độ chính xác dự đoán.
- **Cold Start Problem:** Hệ thống có ít dữ liệu về người dùng hoặc sản phẩm mới, khiến gợi ý ban đầu kém chính xác.
- **Khả năng mở rộng:** Chi phí tính toán tăng khi số lượng người dùng và sản phẩm tăng, đặc biệt với memory-based methods.
- **Popularity Bias:** Hệ thống có xu hướng ưu tiên các sản phẩm phổ biến, bỏ qua các sản phẩm ngách có thể phù hợp hơn với người dùng cụ thể.
- **Lo ngại quyền riêng tư:** Cần theo dõi và phân tích lượng lớn dữ liệu người dùng, phải tuân thủ các quy định như GDPR và CCPA.
