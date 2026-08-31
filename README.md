# ViFinQA — Financial Table Retrieval & Text-to-Pandas Query Generation

Pipeline trả lời câu hỏi tài chính bằng tiếng Việt trên báo cáo tài chính của 100 công ty niêm
yết. Với một câu hỏi ngôn ngữ tự nhiên (ví dụ: _"Doanh thu thuần của VNM năm 2023 là bao
nhiêu?"_), hệ thống truy hồi (retrieval) bảng nguồn chứa đáp án, ánh xạ (grounding) câu hỏi vào
các ô dữ liệu cụ thể, sinh một câu lệnh `pandas` thực thi được, và trả về đáp án số kèm đường
dẫn truy vết đầy đủ về đúng dòng trong file OCR gốc.

## 1. Tổng quan pipeline

```
extraction/       báo cáo .txt  ->  bảng thô <table>, giữ nguyên số dòng gốc trong OCR
normalization/     bảng thô     ->  schema thống nhất (ticker/năm/loại BCTC/đơn vị) + retrieval text
retrieval/        câu hỏi       ->  tập bảng ứng viên (BM25, tuỳ chọn dense + hybrid fusion)
schema_linking/    câu hỏi       ->  toán hạng (ô dòng/cột) cụ thể trong các bảng đã truy hồi
query_generation/  ngữ cảnh      ->  kế hoạch truy vấn pandas đã grounding (Qwen2.5-Coder-7B-Instruct-AWQ)
execution/         truy vấn      ->  chạy, kiểm tra, và vòng lặp sửa lỗi khi thất bại
submission/        dự đoán từng câu hỏi -> submission.json chính thức + validate
```

Mỗi giai đoạn là một module độc lập dưới `src/`; `src/pipeline.py` nối
`schema_linking -> query_generation -> execution` cho một câu hỏi đã được truy hồi
(`answer_question`). Việc truy hồi và điều phối toàn bộ corpus nằm ở
`submission/run_full_inference.py`, chạy toàn bộ pipeline trên mọi câu hỏi kiểm thử chính thức.

`src-code/` và `eval-code/` là bản sao y hệt của `src/` và (một phần của) `eval/`, được giữ dưới
dạng snapshot phẳng, không phụ thuộc, để đính kèm làm Kaggle Dataset thứ hai khi môi trường
notebook Kaggle không thể `pip install` trực tiếp repo này (xem mục 7). Coi `src/`/`eval/` là
nguồn xác thực; `src-code/`/`eval-code/` chỉ là bản đóng gói của nó.

## 2. Cấu trúc repository

```
docs/
  overview.md            đề bài chính thức của cuộc thi (tiếng Việt)
  data.md                mô tả dữ liệu chính thức (tiếng Việt)
  eval.md                công thức đánh giá chính thức (tiếng Việt)
  submission_guide.md     định dạng nộp bài chính thức (tiếng Việt)
  model_acquisition.md    cách pin/chuẩn bị checkpoint Qwen dùng lúc runtime cho Kaggle
  technical_report_vi.md  báo cáo kỹ thuật (tiếng Việt) về dữ liệu và mô hình

src/
  common/                đường dẫn dùng chung, bộ nạp bảng có cấu trúc theo dạng stream, từ điển khái niệm tài chính
  extraction/             .txt -> bảng ứng viên thô (parser.py)
  normalization/          bảng thô -> schema chuẩn hoá + catalog (schema.py, build_artifacts.py)
  retrieval/              sparse.py (BM25), dense.py (BGE-M3), hybrid.py (fusion),
                          rerank.py (rerank theo row-label), decompose.py (phân rã câu hỏi),
                          full_corpus.py (điều phối có scope theo công ty)
  schema_linking/         linker.py — câu hỏi -> toán hạng ô đã grounding
  query_generation/       generator.py — sinh và render kế hoạch truy vấn bằng Qwen
  execution/              runner.py — vòng lặp execute/validate/repair
  pipeline.py             điều phối theo từng câu hỏi (schema_linking -> generation -> execution)

eval/
  metrics.py, generation_metrics.py   P/R/F2 cho retrieval và accuracy cho answer/execution
  run_eval.py, run_generation_eval.py, run_dense_retrieval.py    entry point đánh giá
  dev_questions/          tập dev tự xây dựng, đã kiểm chứng thủ công (KHÔNG phải ground truth chính thức)
  reports/                script chẩn đoán/audit dùng trong quá trình tinh chỉnh retrieval

submission/
  run_full_inference.py   suy luận có thể tiếp tục (resumable) trên toàn bộ câu hỏi kiểm thử chính thức
  build_submission.py     format các dự đoán thành công thành submission.json chính thức
  validate_submission.py  validator offline nghiêm ngặt (schema, đường dẫn, chạy lại truy vấn độc lập)

data/
  raw/hf_meta/            questions.jsonl, code_stock.csv, và README của bộ dữ liệu HF chính thức
  raw/sample_reports/     8 báo cáo OCR mẫu dùng để phát triển/test cục bộ
  interim/                chẩn đoán extraction/EDA (báo cáo bất thường, thống kê file, manifest)
  processed/               normalized_tables.csv (catalog cho BM25), normalized_tables.jsonl
                            (bảng có cấu trúc đầy đủ), row_label_index.csv (sidecar cho rerank)

notebooks/vifinqa.ipynb   notebook Kaggle chạy Checkpoint 1-4 từ đầu đến cuối

tests/          mirror 1:1 cấu trúc của src/, cộng thêm tests/eval/ và tests/submission/
```

## 3. Dữ liệu

### 3.1 Nguồn

Bộ dữ liệu chính thức:
**[`AIGuruTinix/ViFinQA`](https://huggingface.co/datasets/AIGuruTinix/ViFinQA)** trên Hugging
Face, gồm:

- `financial_statements/TICKER/YEAR/DOCUMENT/DOCUMENT_extracted.txt` — mỗi file là văn bản OCR
  của một báo cáo tài chính (bảng cân đối kế toán, báo cáo kết quả kinh doanh, báo cáo lưu
  chuyển tiền tệ, hoặc thuyết minh), 100 công ty niêm yết, giai đoạn 2015–2025, tổng cộng 1.973
  báo cáo. Các bảng được nhúng inline dưới dạng đoạn HTML
  `<table><tr><td>...</td></tr></table>`; không có định dạng bảng dựng sẵn.
- `code_stock.csv` — ánh xạ mã cổ phiếu (ticker) sang tên công ty.
- `questions/questions.jsonl` — bộ câu hỏi chính thức, `{"id": int, "question": str}`, không kèm
  đáp án.

### 3.2 Truy cập và tải dữ liệu

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AIGuruTinix/ViFinQA",
    repo_type="dataset",
    local_dir="data/raw/vifinqa_corpus",
)
```

Hoặc dùng thư viện `datasets`, chỉ tải riêng file câu hỏi:

```python
from datasets import load_dataset
questions = load_dataset("AIGuruTinix/ViFinQA", data_files="questions/questions.jsonl", split="train")
```

Trỏ `--questions`, `--companies`, và thư mục gốc corpus extraction về nơi bạn đặt bản tải về
(xem mục 7 để biết giá trị mặc định chính xác của các tham số CLI).

### 3.3 Link chia sẻ cho artifact đã xử lý của dự án

`data/processed/normalized_tables.csv` (catalog cho BM25) và
`data/processed/normalized_tables.jsonl` (bảng có cấu trúc đầy đủ, ~667 MB trên toàn corpus)
được sinh cục bộ bởi `normalization/build_artifacts.py` và không được commit vào repo dưới dạng
dữ liệu nhị phân.

## 4. Mô hình và checkpoint

| Vai trò                          | Model                                                                                                                                                      | Số tham số | License    | Ngày phát hành | Dùng bởi                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | -------------- | -------------------------------------------------------- |
| Sinh truy vấn (query generation) | [`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ), pin revision `b56cc04415fac88c421533036e44149a5983dd2a` | 7,61B      | Apache-2.0 | 2024-09-19     | `src/query_generation/generator.py` (`QwenAWQGenerator`) |
| Dense retrieval (tuỳ chọn)       | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)                                                                                                        | 568M       | MIT        | 2024-02        | `src/retrieval/dense.py` (`SentenceTransformerEncoder`)  |

Cả hai bản ghi tính hợp lệ (`MODEL_ELIGIBILITY` / `DENSE_MODEL_ELIGIBILITY`) được định nghĩa
ngay cạnh mã nạp mô hình, không chỉ ghi trong tài liệu.

- **Sinh truy vấn là bắt buộc** cho toàn bộ pipeline; mô hình chỉ nhận các toán hạng đã được
  `schema_linking` grounding sẵn và trả về một kế hoạch số học dạng JSON — không thể "bịa" bảng,
  ô, hay giá trị (bị `validate_plan` từ chối).
- **Dense retrieval là tuỳ chọn**, bật bằng cờ `--dense-index-dir` trên `run_full_inference.py`.
  Retriever mặc định cho production là BM25 (sparse), có thể kết hợp thêm kết quả dense từ
  BGE-M3 bằng backfill fusion cộng dồn (`retrieval/hybrid.py::backfill_fusion`) thay vì RRF đối
  xứng.

### 4.1 Tải và nạp checkpoint

Cả hai checkpoint được tải về **một lần duy nhất** (chuẩn bị offline), sau đó nạp lại hoàn toàn
offline (`local_files_only=True`) lúc suy luận — không truy cập mạng lúc runtime:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    revision="b56cc04415fac88c421533036e44149a5983dd2a",
    local_dir="qwen25-coder-7b-instruct-awq",
)

snapshot_download(repo_id="BAAI/bge-m3", local_dir="bge-m3")
```

Trên Kaggle, publish mỗi thư mục đã tải thành một Kaggle Dataset riêng tư và attach vào notebook
suy luận. Thiết lập:

- `VIFINQA_QWEN_MODEL_PATH` (hoặc truyền `model_path` trực tiếp cho `run_full_inference.py`)
  trỏ tới thư mục snapshot Qwen chỉ đọc.
- `--dense-model-path` trỏ tới thư mục BGE-M3 đã attach cục bộ, **hoặc** `--dense-hf-repo-id
BAAI/bge-m3` để `run_dense_retrieval.py` tự tải một lần vào `--dense-hf-cache-dir`.

## 5. Môi trường và dependency

- Python 3.11 (các file cache `.pyc` dưới `src/**/__pycache__` được build cho `cpython-311`).
- Repo này không có file `requirements.txt` cố định phiên bản. Các package sau được import trực
  tiếp trong mã nguồn và cần được cài đặt:

  | Package                 | Dùng bởi                                                                                 |
  | ----------------------- | ---------------------------------------------------------------------------------------- |
  | `pandas`                | normalization, retrieval, execution, submission                                          |
  | `numpy`                 | dense retrieval (`retrieval/dense.py`)                                                   |
  | `transformers`          | `query_generation/generator.py` (`QwenAWQGenerator`)                                     |
  | `torch`                 | suy luận sinh truy vấn, encode dense (fp16 trên CUDA)                                    |
  | `sentence-transformers` | dense retrieval (`retrieval/dense.py`), tuỳ chọn                                         |
  | `huggingface_hub`       | tải checkpoint/dataset một lần (`snapshot_download`)                                     |
  | `tqdm`                  | thanh tiến trình suy luận toàn corpus (`submission/run_full_inference.py`, encode dense) |
  | `pytest`                | bộ test                                                                                  |

  Cần một backend `transformers`/`accelerate`/quantization tương thích AWQ để thực sự chạy suy
  luận `Qwen2.5-Coder-7B-Instruct-AWQ` (notebook Kaggle cài thêm `gptqmodel` và cài lại
  `transformers`/`accelerate` từ wheel)

- BM25 sparse retrieval (`retrieval/sparse.py`) là cài đặt inverted-index tự viết từ đầu, không
  phụ thuộc thư viện ngoài — không cần `rank_bm25` hay package tương tự cho nhánh sparse.
- GPU: sinh truy vấn và encode dense được thiết kế cho GPU CUDA

Cài dependency cốt lõi:

```bash
pip install pandas numpy transformers torch sentence-transformers huggingface_hub tqdm pytest
```

## 6. Cấu hình

Không cần file `.env` hay secret nào — pipeline không gọi API bên ngoài lúc suy luận.

Các biến môi trường thực sự được mã nguồn đọc:

| Biến                       | Đọc bởi                                                                                 | Mục đích                                                                                               | Mặc định     |
| -------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------ |
| `VIFINQA_ROOT`             | `src/common/paths.py`                                                                   | Ghi đè repo root đã phát hiện tự động (nếu không sẽ dùng `/kaggle/working` trên Kaggle hoặc repo root) | tự phát hiện |
| `VIFINQA_MAX_INPUT_TOKENS` | `src/query_generation/generator.py`                                                     | Giới hạn kích thước ngữ cảnh (các toán hạng đã grounding) trong prompt                                 | `4096`       |
| `VIFINQA_QWEN_MODEL_PATH`  | quy ước trong `docs/model_acquisition.md` (cục bộ thì truyền trực tiếp qua tham số CLI) | Đường dẫn tới checkpoint Qwen offline đã attach                                                        | không có     |
| `VIFINQA_CODE_PATH`        | quy ước trong `docs/model_acquisition.md` (thiết lập notebook Kaggle)                   | Đường dẫn tới dataset snapshot `src-code`/`eval-code` đã attach                                        | không có     |

## 7. Cài đặt và tái lập từ môi trường sạch

### 7.1 Cài đặt và tải dữ liệu

```bash
git clone <this-repo> && cd r2ai-stage2
pip install pandas numpy transformers torch sentence-transformers huggingface_hub tqdm pytest

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='AIGuruTinix/ViFinQA', repo_type='dataset', local_dir='data/raw/vifinqa_corpus')
"
```

Lệnh này cho bạn `data/raw/vifinqa_corpus/financial_statements/**/*_extracted.txt`,
`data/raw/vifinqa_corpus/code_stock.csv`, và
`data/raw/vifinqa_corpus/questions/questions.jsonl`. File câu hỏi và bảng mã công ty chính thức
đã có sẵn cục bộ tại `data/raw/hf_meta/questions.jsonl` và `data/raw/hf_meta/code_stock.csv`;
chỉ cần tải riêng corpus báo cáo `financial_statements/` để chạy đầy đủ. 8 báo cáo mẫu đã được
commit sẵn tại `data/raw/sample_reports/` để phát triển cục bộ mà không cần toàn bộ corpus.

### 7.2 Dựng artifact đã chuẩn hoá (bắt buộc một lần, có thể tiếp tục)

```bash
python -c "
from pathlib import Path
from normalization.build_artifacts import build
build(
    corpus_root=Path('data/raw/vifinqa_corpus'),
    processed_dir=Path('data/processed'),
    interim_dir=Path('data/interim'),
    companies_path=Path('data/raw/hf_meta/code_stock.csv'),
)
"
```

Sinh ra `data/processed/normalized_tables.csv` (catalog cho BM25) và
`data/processed/normalized_tables.jsonl` (bảng có cấu trúc đầy đủ theo từng ô, dùng cho schema
linking và evidence CSV). An toàn khi chạy lại: bước này bỏ qua nếu cả bốn artifact output đã
tồn tại và không rỗng (truyền `force=True` để dựng lại).

Tuỳ chọn dựng thêm sidecar rerank theo row-label
(`retrieval/rerank.py::build_row_label_index`) — `run_full_inference.py` tự phát hiện
`data/processed/row_label_index.csv` nếu có và dùng nó để cải thiện xếp hạng trong phạm vi đã
scope; nếu không có thì bỏ qua một cách an toàn.

### 7.3 Chạy test

```bash
pytest
```

`pyproject.toml` cấu hình `pythonpath = ["src", "."]` và `testpaths = ["tests"]`, nên chạy
`pytest` từ repo root sẽ tự động thu thập test của mọi giai đoạn mà không cần thêm flag. Test
chạy trên các báo cáo mẫu đã commit và fixture tổng hợp nhỏ — không cần tải toàn bộ corpus hay
bất kỳ checkpoint mô hình nào.

### 7.4 Chạy suy luận trên toàn bộ câu hỏi kiểm thử chính thức

Cần checkpoint Qwen đã tải (mục 4.1), khả dụng cục bộ và nạp được với
`local_files_only=True`:

```bash
python submission/run_full_inference.py <đường-dẫn-tới-qwen25-coder-7b-instruct-awq> \
    --questions data/raw/hf_meta/questions.jsonl \
    --catalog data/processed/normalized_tables.csv \
    --companies data/raw/hf_meta/code_stock.csv \
    --structured data/processed/normalized_tables.jsonl \
    --package-dir submission/package \
    --top-k 10 --max-retries 2
```

Bước này có thể tiếp tục (resumable): nó bỏ qua các câu hỏi đã có dự đoán hoàn chỉnh dưới
`submission/package/work/predictions/`, và ghi lại thất bại riêng dưới
`submission/package/work/failures/<id>.json` để kiểm tra. Thêm
`--dense-index-dir data/processed/dense_bge_m3_enriched_full_corpus --dense-hf-repo-id BAAI/bge-m3`
để bật retrieval hybrid (BM25 + dense, kết hợp bằng backfill) thay vì chỉ BM25.

### 7.5 Dựng và validate file nộp bài

```bash
python submission/build_submission.py \
    --questions data/raw/hf_meta/questions.jsonl \
    --predictions submission/package/work/predictions \
    --output submission/package/submission.json

python submission/validate_submission.py submission/package/submission.json \
    --questions data/raw/hf_meta/questions.jsonl \
    --package-dir submission/package
```

`build_submission.py` chỉ đơn thuần format: nó từ chối đưa vào bất kỳ câu hỏi nào có dự đoán bị
thiếu, thất bại, hoặc không phải số hữu hạn, thay vì bịa ra một giá trị.
`validate_submission.py` chạy lại độc lập từng `pandas_query` trên các evidence CSV đã đóng gói
và kiểm tra schema, containment của đường dẫn, và độ bao phủ đầy đủ theo ID chính thức trước khi
thoát với mã `0`.

## 8. Input, output, và artifact được sinh ra

| Artifact                                        | Sinh bởi                                                | Được dùng bởi                                       |
| ----------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| `data/processed/normalized_tables.csv`          | `normalization/build_artifacts.py`                      | `retrieval/*`, `eval/run_eval.py`                   |
| `data/processed/normalized_tables.jsonl`        | `normalization/build_artifacts.py`                      | `common/table_store.py` (schema linking / evidence) |
| `data/processed/row_label_index.csv`            | `retrieval/rerank.py::build_row_label_index` (tuỳ chọn) | `retrieval/full_corpus.py` tự phát hiện             |
| `submission/package/work/predictions/<id>.json` | `submission/run_full_inference.py`                      | `submission/build_submission.py`                    |
| `submission/package/work/failures/<id>.json`    | `submission/run_full_inference.py`                      | debug thủ công                                      |
| `submission/package/data/q<id>_<var>.csv`       | `submission/run_full_inference.py`                      | đóng gói làm evidence trong bài nộp                 |
| `submission/package/submission.json`            | `submission/build_submission.py`                        | nộp lên Dashboard chính thức                        |
