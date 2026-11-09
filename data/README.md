# Dữ liệu

## `raw/` — DỮ LIỆU TỔNG HỢP (synthetic)

32 văn bản Markdown trong `raw/` là **dữ liệu tổng hợp do AI (Claude) soạn**, mô phỏng bộ quy định nội bộ
của một công ty **giả định** tên "Công ty Cổ phần Công nghệ Vân Thạch". Các văn bản:

- không sao chép từ nguồn có bản quyền và không chứa dữ liệu cá nhân thật;
- dùng tên miền dành riêng `*.example` cho mọi địa chỉ email, không nêu tên người thật;
- có con số và thủ tục do AI tự đặt ra: **không phản ánh quy định của bất kỳ tổ chức có thật nào** và
  không phải tư vấn pháp lý, dù một số chi tiết được viết cho giống thực tế lao động ở Việt Nam;
- được cố ý viết thành các cặp chủ đề gần nhau (nghỉ phép / nghỉ ốm, công tác phí / hoàn trả chi phí,
  mật khẩu / bảo mật / sự cố an ninh...) để tạo "hard negative" cho bài toán truy hồi.

Định dạng: `# Tiêu đề`, dòng thông tin văn bản, các mục `## n. Tên mục`, đoạn văn và danh sách gạch đầu dòng.
Có thể thay bằng dữ liệu của bạn: đặt file `.txt` hoặc `.md` vào `raw/` rồi chạy lại `python -m src.cli build`.
File `.txt` không cần heading; khi đó tiêu đề lấy theo tên file.

## `index/` — artifact sinh ra (không commit)

Tạo bởi `python -m src.cli build`: `chunks.jsonl`, `bm25_tokens.json`, `faiss.index`, `manifest.json`.
