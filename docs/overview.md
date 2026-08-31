## **Bối cảnh bài toán**

Nhà đầu tư, chuyên viên phân tích và doanh nghiệp tại Việt Nam thường mất nhiều thời gian tra cứu thủ công các chỉ số tài chính (doanh thu, lợi nhuận, ROE, ROA, tỉ lệ nợ/vốn chủ sở hữu, tăng trưởng theo giai đoạn...) nằm rải rác trong hàng trăm báo cáo tài chính (BCTC) dạng bảng của các công ty niêm yết qua nhiều năm. Trợ lý AI Text-to-Pandas được xây dựng nhằm hỗ trợ tự động hoá việc tra cứu, tổng hợp và tính toán các chỉ số này từ dữ liệu BCTC gốc.

Trong bối cảnh trí tuệ nhân tạo phát triển mạnh mẽ, đặc biệt với sự xuất hiện của các mô hình ngôn ngữ lớn như ChatGPT, DeepSeek và Qwen, nhu cầu xây dựng các hệ thống AI có khả năng chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn dữ liệu bảng (Text-to-Code/Text-to-Pandas) ngày càng trở nên quan trọng. Tuy nhiên, so với các bài toán Text-to-SQL trên dữ liệu tiếng Anh, nguồn tài nguyên và nghiên cứu về Text-to-Pandas trên dữ liệu tài chính tiếng Việt vẫn còn hạn chế.

Nhằm thúc đẩy nghiên cứu và phát triển trong lĩnh vực này, chúng tôi tổ chức cuộc thi về Truy hồi Bảng dữ liệu & Sinh truy vấn Pandas trên Báo cáo tài chính doanh nghiệp niêm yết (Financial Table Retrieval & Text-to-Pandas Query Generation). Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng xác định đúng bảng dữ liệu liên quan và tự động sinh, thực thi câu lệnh pandas để trả lời chính xác câu hỏi về chỉ số tài chính.

Truy hồi bảng dữ liệu (Table Retrieval) là nhiệm vụ cốt lõi đầu tiên, liên quan đến việc xác định bảng dữ liệu nào phù hợp nhất với một truy vấn cho trước. Nhiệm vụ có thể được hình thức hoá như sau: Cho một tập câu hỏi Q \= {q1, q2, ..., qn} và một kho báo cáo tài chính D \= {d1, d2, ..., dn} (mỗi báo cáo gồm nhiều bảng: Bảng cân đối kế toán, Báo cáo kết quả kinh doanh, Báo cáo lưu chuyển tiền tệ, thuyết minh), nhiệm vụ yêu cầu xác định một tập con bảng D′ ⊂ D trong đó mỗi bảng di ∈ D′ được coi là "liên quan" đến câu hỏi tương ứng q. Chúng tôi gọi một bảng dữ liệu là "Liên quan" nếu bảng đó chứa (một phần hoặc toàn bộ) số liệu cần thiết để tính ra câu trả lời.

Sinh truy vấn Pandas (Text-to-Pandas) dựa trên các bảng đã truy hồi, hệ thống cần sinh ra câu lệnh pandas thực thi được để tính toán và trả về đúng số liệu cho câu hỏi tài chính tương ứng. Mục tiêu của nhiệm vụ là xây dựng các hệ thống AI có khả năng không chỉ tìm đúng bảng dữ liệu căn cứ mà còn hiểu và chuyển hoá đúng logic tính toán tài chính thành code, đảm bảo kết quả có thể kiểm chứng và tái lập.

## **Mục tiêu cuộc thi**

Các đội thi cần xây dựng hệ thống AI có khả năng:

#### **1\. Truy hồi dữ liệu chính xác**

* Xác định đúng công ty, đúng năm, đúng bảng dữ liệu chứa số liệu cần thiết.  
* Tìm kiếm và truy xuất chính xác vị trí bảng dữ liệu từ kho BCTC được cung cấp.  
* Ưu tiên khả năng retrieval và grounding chính xác trên dữ liệu dạng bảng.

#### **2\. Hiểu truy vấn tài chính bằng tiếng Việt**

* Hiểu ngôn ngữ tự nhiên tiếng Việt về các chỉ số và thuật ngữ tài chính.  
* Xử lý được câu hỏi so sánh nhiều công ty, nhiều năm, hoặc chỉ số dẫn xuất (ROE, ROA, tăng trưởng...).

#### **3\. Sinh truy vấn pandas & tính toán chính xác**

* Sinh câu lệnh pandas chạy được, đúng logic, đúng schema dữ liệu.  
* Trả về đúng số liệu, đúng đơn vị, đúng kỳ báo cáo được hỏi.

#### **4\. Dẫn nguồn minh bạch**

* Trích dẫn công ty, năm, tên báo cáo, tên bảng và vị trí (trang/mục) chứa số liệu gốc.  
* Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng thông tin.  
* Hạn chế việc trả lời không có căn cứ dữ liệu.

#### **5\. Kiểm soát nội dung sai lệch**

* Hạn chế việc AI sinh ra số liệu sai lệch (hallucination).  
* Tránh bịa bảng dữ liệu hoặc nguồn tham chiếu không tồn tại.  
* Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

## **Quy định về dữ liệu bên ngoài và mô hình ngôn ngữ huấn luyện trước (PLMs)**

Người tham gia được phép sử dụng dữ liệu từ các nguồn bên ngoài, tuy nhiên phải trích dẫn rõ ràng và cung cấp đầy đủ thông tin về nguồn gốc dữ liệu để Ban tổ chức có thể kiểm tra, xác minh khi cần thiết. Bạn có thể sử dụng các mô hình ngôn ngữ huấn luyện trước và các LLM có dữ liệu huấn luyện và/hoặc mô hình được công khai (ví dụ: Hugging Face hoặc các trang tương tự), nhưng bạn không được sử dụng các LLM có mô hình đóng (ví dụ: GPT-4o, Gemini, ...). Ngoài ra, bạn chỉ được sử dụng các mô hình được phát hành trước ngày 1 tháng 6 năm 2026 (giờ Việt Nam) có kích thước nhỏ hơn hoặc bằng 14B. Vì mục đích tái lập kết quả, vui lòng đưa thông tin về cách thức lấy mô hình vào bài báo.

