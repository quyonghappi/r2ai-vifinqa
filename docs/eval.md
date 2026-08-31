Hiệu năng của hệ thống được đánh giá bằng ba tiêu chí tự động: Truy hồi thông tin, Độ chính xác kết quả, và Độ chính xác pandas query. Chúng tôi sử dụng trung bình macro (chỉ số đánh giá được tính cho từng truy vấn rồi lấy trung bình) để tính điểm đánh giá cuối cùng.

### **3.1 Truy hồi thông tin**

Hiệu suất hệ thống trên nhiệm vụ truy hồi bảng dữ liệu được đánh giá bằng các chỉ số Độ chính xác (Precision), Độ bao phủ (Recall) và điểm F2 macro. Chúng tôi sử dụng macro-average (tính chỉ số đánh giá cho từng truy vấn rồi lấy trung bình) để tính điểm đánh giá cuối cùng.

* Độ chính xác (Precision): Precision \= trung bình của (số bảng dữ liệu truy hồi đúng cho mỗi truy vấn) / (số bảng dữ liệu đã truy hồi cho mỗi truy vấn)  
* Độ bao phủ (Recall): Recall \= trung bình của (số bảng dữ liệu truy hồi đúng cho mỗi truy vấn) / (số bảng dữ liệu liên quan của mỗi truy vấn)  
* Độ đo F2: F2 \= (5 × Precision × Recall) / (4 × Precision \+ Recall)

### **3.2 Độ chính xác kết quả**

Độ chính xác của số liệu đầu ra so với đáp án chuẩn, tính trong ngưỡng sai số cho phép do Ban Tổ chức (BTC) công bố.

* Answer Accuracy \= (số query có kết quả khớp đáp án chuẩn, trong ngưỡng sai số) / (tổng số query)

### **3.3 Độ chính xác pandas query**

Hiệu suất hệ thống trên nhiệm vụ sinh mã truy vấn và tính toán trên bảng dữ liệu tài chính được đánh giá bằng chỉ số Execution Accuracy. Chúng tôi sử dụng macro-average để tính điểm đánh giá cuối cùng.

* Execution Accuracy \= (số code chạy được và cho kết quả đúng) / (tổng số query)

