**Nguồn dữ liệu chính thức của cuộc thi: https://huggingface.co/datasets/AIGuruTinix/ViFinQA** 

**Ban Tổ chức cung cấp:**

* Kho dữ liệu báo cáo tài chính: BCTC của 100 công ty niêm yết trong 10 năm (Bảng cân đối kế toán, Báo cáo kết quả kinh doanh, Báo cáo lưu chuyển tiền tệ, thuyết minh BCTC), làm nguồn dữ liệu gốc để truy hồi và tính toán.  
* **Bộ dữ liệu kiểm thử (test set):** tập câu hỏi về chỉ số tài chính, được sử dụng làm căn cứ chấm điểm và đánh giá hệ thống của các đội thi. **Không cung cấp bất kỳ tập dữ liệu huấn luyện (train) hay tập phát triển (dev) nào.**

**Bộ đáp án chuẩn:** được Ban Tổ chức giữ kín

Các đội thi được toàn quyền chủ động trong việc trích xuất, làm sạch và cấu trúc hoá dữ liệu, bao gồm:

* Trích xuất bảng dữ liệu từ file do BTC cung cấp.  
* Xây dựng schema chuẩn hoá (tên công ty, năm, tên bảng, tên cột, đơn vị tính).  
* Các tập dữ liệu mở (open dataset) khác về tài chính doanh nghiệp niêm yết Việt Nam.  
* Mọi nguồn dữ liệu hợp pháp khác mà đội thi có thể tiếp cận.

Cuộc thi khuyến khích các đội phát huy tối đa sự sáng tạo trong toàn bộ quy trình xây dựng giải pháp, chẳng hạn:

* Tiền xử lý dữ liệu bảng từ BCTC.  
* Thiết kế chiến lược biểu diễn dữ liệu và schema linking (ánh xạ câu hỏi ↔ tên bảng/cột).  
* Tối ưu hoá cơ chế truy hồi bảng dữ liệu liên quan.  
* Xây dựng pipeline sinh, kiểm tra và tự sửa lỗi câu lệnh pandas.

### **Kho báo cáo tài chính**

Ban Tổ chức cung cấp kho báo cáo tài chính dưới dạng các file văn bản có phần mở rộng `.txt`. Mỗi file chứa nội dung của một báo cáo tài chính, bao gồm các thông tin thuyết minh và các bảng số liệu liên quan.

Mỗi đội thi có nhiệm vụ khai thác dữ liệu từ các file `.txt` để truy hồi thông tin và tính toán đáp án cho bộ câu hỏi kiểm thử. Các đội được chủ động lựa chọn phương pháp nhận diện bảng, trích xuất, làm sạch, chuẩn hóa và cấu trúc hóa dữ liệu phù hợp với giải pháp của mình.

### **Bộ câu hỏi kiểm thử**

Mỗi câu hỏi trong bộ dữ liệu kiểm thử bao gồm:

{

  "id": \<integer\>,  
  "question": "\<string\>"  
}  
eg:   
{  
  "id": 1,  
  "question": "Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu VNĐ?"  
}

