# Báo cáo kỹ thuật: Dữ liệu và mô hình trong hệ thống ViFinQA

Tài liệu này mô tả dữ liệu và các mô hình thực sự được sử dụng trong pipeline hiện tại của dự
án, dựa trên việc rà soát mã nguồn (`src/`), notebook `notebooks/vifinqa.ipynb`, tài liệu đề bài (`docs/`), và lịch sử quyết định

## 1. Dữ liệu

### 1.1 Nguồn dữ liệu chính thức

Nguồn dữ liệu duy nhất được Ban Tổ chức cung cấp và được pipeline sử dụng là bộ dữ liệu
[`AIGuruTinix/ViFinQA`](https://huggingface.co/datasets/AIGuruTinix/ViFinQA) trên Hugging
Face. Bộ dữ liệu này gồm ba thành phần:

- Kho báo cáo tài chính (`financial_statements/`): 1.973 báo cáo dạng file `.txt`, kết quả
  OCR từ báo cáo tài chính (BCTC) của 100 công ty niêm yết trong giai đoạn 2015–2025. Mỗi file
  chứa cả văn bản thuyết minh và các bảng số liệu chèn trực tiếp trong text dưới dạng đoạn HTML
  `<table><tr><td>...</td></tr></table>` - dữ liệu không có cấu trúc bảng sẵn, việc trích xuất
  bảng là một bài toán xử lý, không phải bước tiền xử lý đơn giản (đúng như `docs/data.md` mô
  tả).
- `code_stock.csv`: ánh xạ mã cổ phiếu (ticker) sang tên công ty.
- Bộ câu hỏi kiểm thử (`questions/questions.jsonl`): các bản ghi `{"id": int, "question":
str}`, không kèm đáp án. Đây là tập được dùng để chấm điểm chính thức.

Dự án không được cung cấp tập huấn luyện (train) hay tập phát triển (dev) chính thức nào.
Đáp án chuẩn (ground truth) được Ban Tổ chức giữ kín hoàn toàn cho đến khi chấm điểm

### 1.2 Cấu trúc và định dạng dữ liệu

Cấu trúc thư mục của kho báo cáo (theo thẻ dữ liệu chính thức và xác nhận qua các file mẫu đã
tải về tại `data/raw/sample_reports/`):

```
financial_statements/
└── TICKER/
    └── YEAR/
        └── DOCUMENT_NAME/
            └── DOCUMENT_NAME_extracted.txt
```

Ví dụ: `financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/
AAA_financial_statements_2015_consolidated_extracted.txt`.

Tên file thường cho biết loại báo cáo là hợp nhất (`consolidated`) hay công ty mẹ/riêng lẻ
(`separate`); một số ít dùng `aggregated` hoặc tên không theo quy ước (khoảng 55/1.973 file,
theo thẻ dữ liệu chính thức). Mỗi file `.txt` là văn bản OCR có đánh dấu trang
(`===== PAGE N =====`) và các bảng nằm trọn trên một dòng vật lý duy nhất - điều này đã được
xác minh thực nghiệm trên 8 file mẫu (540 bảng, 0 ngoại lệ; xem docstring của
`src/extraction/parser.py`), cho phép lấy chính xác số dòng bắt đầu của bảng mà không cần suy
luận heuristic, đúng yêu cầu `relevant_tables` của định dạng nộp bài.

`code_stock.csv` có hai cột: `Mã CK` (ticker) và `Tên công ty` (tên đầy đủ).

### 1.3 Hệ thống tiêu thụ dữ liệu như thế nào

Pipeline xử lý dữ liệu qua các giai đoạn tách biệt, mỗi giai đoạn có ranh giới trách nhiệm rõ
ràng:

1. `extraction/parser.py` - đọc từng file `.txt`, tìm mọi đoạn `<table>`, dựng thành lưới ô
   (grid) đầy đủ (mở rộng `rowspan`/`colspan`), giữ nguyên số dòng gốc trong file OCR. Không lọc
   hay đánh giá độ liên quan của bảng - việc đó thuộc về các giai đoạn sau.
2. `normalization/build_artifacts.py` + `schema.py` - chuẩn hoá mỗi bảng thô thành một bản
   ghi có ticker, năm, loại báo cáo (`variant`), đơn vị tiền tệ phát hiện được, và một
   `retrieval_text` dùng cho tìm kiếm. Kết quả ghi ra hai file:
   - `data/processed/normalized_tables.csv` - catalog gọn dùng cho retrieval (BM25).
   - `data/processed/normalized_tables.jsonl` - bản ghi đầy đủ từng ô (grid), dùng cho schema
     linking và dựng DataFrame bằng chứng (~667 MB trên toàn bộ corpus).
3. `retrieval/*` - nhận câu hỏi, trả về tập bảng ứng viên (mặc định BM25 thưa; có tuỳ chọn
   kết hợp dense + rerank, xem mục 2).
4. `schema_linking/linker.py` - ánh xạ câu hỏi vào các ô cụ thể (row_index, column_index)
   trong các bảng đã truy hồi.
5. `query_generation/generator.py` - mô hình Qwen sinh một kế hoạch truy vấn dạng JSON chỉ
   dùng các toán hạng đã được grounding, sau đó dựng thành câu lệnh `pandas` thực thi được.
6. `execution/runner.py` - chạy câu lệnh, bắt lỗi, và có vòng lặp sửa lỗi tường minh
   (generate → validate → execute → feedback → repair), tối đa `max_retries` lần thử lại.

Toàn bộ quá trình được `submission/run_full_inference.py` điều phối trên toàn bộ tập câu hỏi
chính thức, có khả năng resumable sau khi bị gián đoạn.

### 1.4 Cách truy cập, tải và sử dụng dữ liệu

Tải toàn bộ bộ dữ liệu bằng `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AIGuruTinix/ViFinQA",
    repo_type="dataset",
    local_dir="data/raw/vifinqa_corpus",
)
```

Hoặc chỉ tải riêng file câu hỏi bằng thư viện `datasets`:

```python
from datasets import load_dataset

questions = load_dataset(
    "AIGuruTinix/ViFinQA", data_files="questions/questions.jsonl", split="train",
)
```

Sau khi tải, chạy bước chuẩn hoá (`normalization/build_artifacts.py`, xem README.md §7.2) để
sinh ra `normalized_tables.csv`/`.jsonl` mà các giai đoạn retrieval/schema linking sử dụng.

Bản sao câu hỏi và `code_stock.csv` chính thức đã có sẵn trong repo tại
`data/raw/hf_meta/questions.jsonl` và `data/raw/hf_meta/code_stock.csv`; chỉ cần tải riêng phần
`financial_statements/` (kho báo cáo, dung lượng lớn) để chạy đầy đủ. 8 báo cáo mẫu đã được
đính kèm tại `data/raw/sample_reports/` phục vụ phát triển/test cục bộ không cần tải toàn bộ
corpus.

### 1.5 Link chia sẻ dữ liệu

- Nguồn dữ liệu gốc, công khai: `https://huggingface.co/datasets/AIGuruTinix/ViFinQA` (đã xác
  minh, dùng trực tiếp trong pipeline).
- Các artifact đã xử lý của riêng dự án (`normalized_tables.csv`, `normalized_tables.jsonl`,
  ~667 MB trên toàn corpus) hiện không được publish ở bất kỳ nơi lưu trữ chia sẻ nào
  (Google Drive/OneDrive/Hugging Face...); các file này được sinh cục bộ và không commit vào
  repo dưới dạng nhị phân. `TODO`: chưa có link chia sẻ công khai cho các artifact đã chuẩn
  hoá - cần publish nếu muốn tái lập nhanh mà không phải chạy lại bước normalization.

## 2. Mô hình và checkpoint

### 2.1 Mô hình sinh truy vấn (bắt buộc)

| Thuộc tính                | Giá trị                                                                                                                                                                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model ID                  | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`                                                                                                                                                                                                    |
| Revision đã pin           | `b56cc04415fac88c421533036e44149a5983dd2a`                                                                                                                                                                                              |
| Số tham số                | 7,61 tỷ (dưới ngưỡng 14B)                                                                                                                                                                                                               |
| License                   | Apache-2.0 (open-weight)                                                                                                                                                                                                                |
| Ngày phát hành            | 2024-09-19 (trước hạn 2026-06-01)                                                                                                                                                                                                       |
| Vai trò                   | Sinh kế hoạch truy vấn (query plan) dạng JSON từ ngữ cảnh đã schema-link; kế hoạch chỉ được phép dùng các toán hạng (bảng/dòng/cột) đã thực sự tồn tại - mọi toán hạng "bịa" bị `validate_plan` từ chối trước khi render thành `pandas` |
| Định nghĩa trong mã nguồn | `src/query_generation/generator.py` (`MODEL_ID`, `MODEL_REVISION`, `MODEL_ELIGIBILITY`, lớp `QwenAWQGenerator`)                                                                                                                         |

Cấu hình suy luận đã chốt (`docs/model_acquisition.md`): batch size 1, `do_sample=False`,
`max_new_tokens=768` (giới hạn thực thi trong code hiện tại là 256 - xem
`MAX_NEW_TOKENS` trong `generator.py`; số 768 trong tài liệu chuẩn bị mô hình là mức trần dự
kiến ban đầu, con số thực thi trong code là nguồn xác thực), ngữ cảnh đã schema-link giới hạn
gần 12K token (thực thi bằng `VIFINQA_MAX_INPUT_TOKENS`, mặc định 4096 token trong code hiện
tại). Suy luận chạy hoàn toàn offline (`local_files_only=True`), không có lệnh gọi API mô
hình đóng nào trong `src/`.

### 2.2 Mô hình dense retrieval

| Thuộc tính                | Giá trị                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| Model ID                  | `BAAI/bge-m3`                                                                                            |
| Số tham số                | 568 triệu                                                                                                |
| License                   | MIT                                                                                                      |
| Ngày phát hành            | 2024-02                                                                                                  |
| Vai trò                   | Encode câu hỏi và văn bản bảng thành embedding để tìm kiếm dense, kết hợp (fusion) với kết quả BM25      |
| Định nghĩa trong mã nguồn | `src/retrieval/dense.py` (`DENSE_MODEL_ID`, `DENSE_MODEL_ELIGIBILITY`, lớp `SentenceTransformerEncoder`) |

Đây là thành phần tuỳ chọn: retriever mặc định/được duyệt để đưa vào production là BM25
thưa (sparse), không cần mô hình học máy nào (`retrieval/sparse.py` là cài đặt inverted-index
thuần Python, không phụ thuộc thư viện BM25 ngoài). BGE-M3 chỉ được kích hoạt khi truyền
`--dense-index-dir` cho `submission/run_full_inference.py`, và khi đó được kết hợp với BM25
bằng backfill fusion (không phải Reciprocal Rank Fusion đối xứng) - quyết định này được ghi
lại trong `CHANGE_LOG.md` (mục 2026-08-31, "Hybrid fusion: RRF -> backfill"), sau khi RRF cho
kết quả kém hơn BM25 đơn thuần trên các câu hỏi mà row-label reranking đã cải thiện mạnh nhất.

### 2.3 Cách tải và sử dụng checkpoint

Cả hai checkpoint được tải một lần duy nhất trong giai đoạn chuẩn bị (có mạng), sau đó nạp
lại hoàn toàn offline lúc suy luận:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    revision="b56cc04415fac88c421533036e44149a5983dd2a",
    local_dir="qwen25-coder-7b-instruct-awq",
)

snapshot_download(repo_id="BAAI/bge-m3", local_dir="bge-m3")
```

Trên Kaggle (môi trường suy luận thực tế của dự án, theo `docs/model_acquisition.md`): mỗi
thư mục tải về được publish thành một Kaggle Dataset riêng (private), gắn (attach) vào notebook
suy luận, và truyền đường dẫn `/kaggle/input/...` đó cho `QwenAWQGenerator`/
`SentenceTransformerEncoder` - cả hai lớp này nạp mô hình với `local_files_only=True`, tức là
bắt buộc phải thành công ngay cả khi tắt mạng trên Kaggle.

Do `Qwen2.5-Coder-7B-Instruct-AWQ` dùng lượng tử hoá AWQ, môi trường chạy cần một backend
`transformers`/`accelerate` tương thích AWQ; notebook (`notebooks/vifinqa_pipeline.ipynb`) cài
đặt thêm `gptqmodel` và cài lại `transformers`/`accelerate` từ wheel đã kiểm thử trước
(`pip install -q gptqmodel`, cùng các lệnh `pip install --no-cache-dir --force-reinstall` khác
trong notebook).

### 2.4 Link chia sẻ checkpoint

- `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`: công khai, chính thức tại
  `https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`.
- `BAAI/bge-m3`: công khai, chính thức tại `https://huggingface.co/BAAI/bge-m3`.

Cả hai checkpoint được tải trực tiếp từ kho Hugging Face chính thức nêu trên - không cần link
chia sẻ riêng của dự án. Kaggle Dataset dùng để cache offline (ví dụ
`tofuwonion/qwen25-coder-7b-instruct-awq`, xuất hiện trong
`reports/generation_dev_v1/generation_dev_v1_report.json`) là dataset riêng tư, chỉ phục vụ
việc nạp offline trên Kaggle, không phải link chia sẻ công khai. `TODO`: chưa có mirror công
khai riêng của dự án cho hai checkpoint này - không cần thiết vì nguồn Hugging Face chính thức
đã công khai, nhưng nếu Ban Tổ chức yêu cầu một bản sao cố định (không đổi theo thời gian),
đây là phần cần bổ sung.

Ngoài ra, `docs/model_acquisition.md` ghi rõ yêu cầu: các phiên bản wheel (`transformers`,
`accelerate`, backend AWQ) dùng trong môi trường đã kiểm thử cần được ghi lại chính xác trong
báo cáo thực nghiệm cuối cùng để đảm bảo khả năng tái lập - `TODO`: các phiên bản wheel cụ
thể chưa được ghi lại dưới dạng manifest (`requirements.txt`/`environment.yml`) trong repo tại
thời điểm viết báo cáo này.
