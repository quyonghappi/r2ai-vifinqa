Các đội thi nộp kết quả dự đoán trực tiếp trên hệ thống Dashboard chính thức của cuộc thi. Mỗi lần nộp bài cần đảm bảo các yêu cầu sau:

* **Định dạng file:** kết quả được nộp dưới dạng file chuẩn theo mẫu do Ban Tổ chức quy định, với cấu trúc trường dữ liệu tuân thủ đúng đặc tả.  
* **Nội dung file:** bao gồm kết quả dự đoán cho toàn bộ câu hỏi trong bộ dữ liệu kiểm thử. Các câu hỏi bị thiếu hoặc sai định dạng sẽ bị tính là dự đoán không hợp lệ.  
* **Số lần nộp:** mỗi đội được giới hạn số lần nộp bài mỗi ngày (chi tiết sẽ được công bố trên Dashboard) nhằm đảm bảo tính công bằng và tránh hiện tượng dò đáp án.

## **Định dạng nộp bài**

Bạn phải nộp một file dự đoán duy nhất ở định dạng `.json`. File phải tuân theo cấu trúc sau:

\[  
  {  
    "id": \<integer\>,  
    "question": "\<string\>",  
    "answer": \<float\>,  
    "relevant\_docs": \["\<id\_báo\_cáo\>"\],  
    "relevant\_tables": \["\<id\_báo\_cáo\>|\<vị trí trong báo cáo\>"\],  
    "evidence": \[  
      {  
        "variable": "\<tên\_biến\_dataframe\>",  
        "csv\_path": "\<string\>"  
      }  
    \],  
    "pandas\_query": "\<string\>"  
  },  
  ...  
\]

**Giải thích:**

* **id**: Mã định danh của câu hỏi, kiểu số nguyên (integer).  
* **question**: Nội dung câu hỏi tài chính, kiểu chuỗi (string).  
* **relevant\_docs**: Danh sách mã định danh của các báo cáo hoặc tài liệu có liên quan đến câu hỏi. Mã báo cáo được xác định từ tên file cuối cùng trong đường dẫn tài liệu và loại bỏ phần mở rộng `.txt`. Ví dụ, với đường dẫn: `ocr_filter\AAA\2015\AAA_financial_statements_2015_consolidated` thì mã báo cáo được sử dụng là: `AAA_financial_statements_2015_consolidated`.  
* **relevant\_tables**: Danh sách các bảng dữ liệu có liên quan trực tiếp đến câu trả lời. Mỗi phần tử có định dạng `<id_báo_cáo>|<vị trí bảng trong báo cáo>`, trong đó:  
  * **id\_báo\_cáo**: Tên file cuối cùng trong đường dẫn tài liệu sau khi loại bỏ phần mở rộng `.txt`.  
  * **vị trí bảng trong báo cáo**: Vị trí dòng bắt đầu của bảng trong file báo cáo OCR tương ứng do Ban Tổ chức cung cấp.  
* Ví dụ: `AAA_financial_statements_2015_consolidated|350`.  
* **answer**: Kết quả số liệu kiểu số thực (float).  
* **evidence**: Danh sách các bảng dữ liệu được sử dụng để thực thi `pandas_query`. Mỗi phần tử gồm:  
  * **variable**: Tên biến DataFrame đại diện cho bảng và được sử dụng trực tiếp trong `pandas_query`. Tên biến phải hợp lệ trong Python và không được trùng nhau trong cùng một câu hỏi.  
  * **csv\_path**: Đường dẫn tương đối tới file CSV chứa dữ liệu mà `pandas_query` đã sử dụng để tính ra `answer`. Đường dẫn phải nằm trong thư mục `data/` của gói nộp bài.  
* **pandas\_query**: Câu lệnh pandas được sinh ra để trích xuất/tính toán ra đáp án, kiểu chuỗi (string), có thể chạy lại được trên dữ liệu đã chuẩn hoá.

**Ví dụ bài nộp:**

\[  
  {  
    "id": 1,  
    "question": "Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu?",  
    "answer": 63075000000,  
    "relevant\_docs": \["AAA\_financial\_statements\_2015\_consolidated"\],  
    "relevant\_tables": \["AAA\_financial\_statements\_2015\_consolidated|350"\],  
    "evidence": \[  
      {  
        "variable": "df1",  
        "csv\_path": "data/AAA\_financial\_statements\_2015\_consolidated\_table\_1.csv"  
      }  
    \],  
    "pandas\_query": "df1\[(df1.company=='VNM') & (df1.year==2023)\]\['net\_revenue'\].values\[0\]",  
  }  
\]

