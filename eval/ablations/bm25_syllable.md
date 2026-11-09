# Kết quả đánh giá

> Sinh tự động bởi `python -m eval.evaluate` lúc 2026-11-09 10:29 — số liệu từ lần chạy thực tế, không chỉnh tay.

## Thiết lập

- Dữ liệu: 32 văn bản tổng hợp, 183 chunk (chunk_size=120, overlap=30, respect_sections=True).
- Bộ câu hỏi: 50 câu tổng hợp (26 keyword, 24 paraphrase), mỗi câu có đúng một văn bản nguồn.
- BM25: tokenizer `syllable`, stopword=True, k1=1.5, b=0.75.
- Dense: `intfloat/multilingual-e5-small` + FAISS IndexFlatIP (vector đã chuẩn hoá L2).
- Hybrid: RRF k=60, mỗi retriever lấy 50 ứng viên.
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (mặc định), `BAAI/bge-reranker-base` — xếp hạng lại top-20 ứng viên sau RRF.
- Máy: 12th Gen Intel(R) Core(TM) i5-12400 (12 luồng logic); Python 3.11.16; torch 2.15.0+cpu, 6 luồng, GPU: False; HF_HUB_OFFLINE=1.
- Batch đánh giá: 10 câu/batch.
- Ghi đè cấu hình: `sparse.tokenizer=syllable`.

## Kết quả chính

| Cấu hình | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | Độ trễ TB (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chỉ BM25 | 0.660 | 0.840 | 0.860 | 0.734 | 0.758 | 0.3 | 0.3 | 0.4 |

## Theo loại câu hỏi

| Cấu hình | keyword R@1 | keyword R@5 | keyword MRR | paraphrase R@1 | paraphrase R@5 | paraphrase MRR |
|---|---:|---:|---:|---:|---:|---:|
| Chỉ BM25 | 0.962 | 1.000 | 0.981 | 0.333 | 0.667 | 0.466 |

## Nạp model và bộ nhớ

Thời gian nạp model đo riêng, không tính vào độ trễ truy vấn ở trên. RSS = bộ nhớ thường trú của tiến trình, đo tại các mốc: đầu chế độ, sau khi nạp model, cuối mỗi batch và sau khi giải phóng (cột RSS cuối).

| Cấu hình | Model được nạp (số lần × TB giây) | RSS đầu (MiB) | RSS đỉnh (MiB) | RSS cuối (MiB) | Thời gian chạy (s) |
|---|---|---:|---:|---:|---:|
| Chỉ BM25 | — | 53 | 54 | 54 | 0.0 |

## Hạng của chunk đúng theo từng câu hỏi (— = không có trong top-10)

<details><summary>Mở bảng chi tiết</summary>

| Câu | Loại | Câu hỏi | Chỉ BM25 |
|---|---|---|---:|
| q01 | paraphrase | Tôi vào làm trễ mấy phút thì bị tính là vi phạm giờ giấc? | 4 |
| q02 | keyword | Khu vực hút thuốc của công ty nằm ở đâu? | 1 |
| q03 | keyword | Nhân viên chính thức được hưởng bao nhiêu ngày phép năm? | 1 |
| q04 | paraphrase | Khi cưới vợ, tôi được nghỉ mấy ngày mà vẫn có lương? | 2 |
| q05 | keyword | Nghỉ ốm từ 2 ngày liên tiếp trở lên thì phải nộp giấy tờ gì? | 1 |
| q06 | paraphrase | Chồng được nghỉ bao lâu khi vợ đẻ? | 2 |
| q07 | keyword | Mỗi tuần nhân viên được làm việc từ xa tối đa mấy ngày? | 1 |
| q08 | paraphrase | Làm ở nhà thì công ty có trả tiền mạng không? | — |
| q09 | keyword | Làm thêm giờ vào ngày lễ, Tết được trả bao nhiêu phần trăm lương? | 1 |
| q10 | paraphrase | Hôm qua tôi quên quẹt mặt lúc ra về thì phải làm gì? | 3 |
| q11 | paraphrase | Khi nào thì tôi nhận được tiền công hằng tháng? | 2 |
| q12 | keyword | Xếp loại B thì được thưởng hiệu suất quý bao nhiêu phần trăm? | 1 |
| q13 | keyword | Tiêu chuẩn khách sạn khi đi công tác ở Hà Nội tối đa bao nhiêu một đêm? | 1 |
| q14 | paraphrase | Đi gặp khách hàng ở Đà Nẵng, tôi có được đặt vé bay hay phải đi tàu? | 1 |
| q15 | keyword | Khoản chi trên 20 triệu đồng do ai phê duyệt? | 1 |
| q16 | paraphrase | Tôi bỏ tiền túi mua đồ cho dự án thì trong bao lâu phải gửi hóa đơn để được trả lại? | — |
| q17 | keyword | Thông tin của công ty được phân thành mấy cấp độ? | 1 |
| q18 | paraphrase | Nghỉ việc rồi thì tôi còn phải giữ bí mật cho công ty trong bao lâu? | — |
| q19 | keyword | Máy tính xách tay được thay mới theo chu kỳ bao nhiêu năm? | 1 |
| q20 | paraphrase | Tôi cắm thẻ nhớ vào máy công ty để chép tài liệu được không? | — |
| q21 | keyword | Mật khẩu tài khoản phải dài tối thiểu bao nhiêu ký tự? | 1 |
| q22 | paraphrase | Có được dùng mã OTP gửi qua tin nhắn điện thoại để đăng nhập không? | 2 |
| q23 | keyword | Phát hiện sự cố an ninh thông tin thì phải báo cáo trong vòng bao lâu? | 1 |
| q24 | paraphrase | Nghi máy tính bị dính virus thì có nên tắt máy ngay không? | 1 |
| q25 | keyword | Email của khách hàng cần được phản hồi trong bao lâu? | 1 |
| q26 | paraphrase | Sếp có được nhắn việc cho nhân viên lúc 10 giờ đêm không? | 4 |
| q27 | keyword | Thưởng giới thiệu ứng viên cho vị trí thông thường là bao nhiêu? | 1 |
| q28 | paraphrase | Ứng viên có mấy ngày để trả lời offer? | — |
| q29 | keyword | Nhân viên mới được phân công người đồng hành trong bao lâu? | 1 |
| q30 | keyword | Trong thời gian thử việc nhân viên được hưởng bao nhiêu phần trăm lương? | 1 |
| q31 | keyword | Hợp đồng không xác định thời hạn phải báo trước bao nhiêu ngày khi muốn nghỉ việc? | 1 |
| q32 | paraphrase | Sau khi nghỉ hẳn thì bao lâu tôi nhận được khoản tiền cuối cùng? | 1 |
| q33 | keyword | Ngân sách học tập cá nhân mỗi năm là bao nhiêu? | 1 |
| q34 | paraphrase | Thi chứng chỉ bị rớt thì ai trả tiền lệ phí? | 1 |
| q35 | keyword | Tỷ lệ nhân viên xếp loại A trong mỗi phòng ban tối đa là bao nhiêu? | 1 |
| q36 | keyword | Bảo hiểm sức khỏe bổ sung chi trả điều trị nội trú tối đa bao nhiêu? | 1 |
| q37 | paraphrase | Đám cưới của nhân viên thì công ty mừng bao nhiêu tiền? | — |
| q38 | paraphrase | Khi có hỏa hoạn trong tòa nhà thì có được đi thang máy xuống không? | 5 |
| q39 | keyword | Báo cáo hành vi quấy rối tình dục qua kênh nào? | 2 |
| q40 | paraphrase | Đối tác biếu tôi một giỏ quà Tết trị giá 1 triệu thì có được nhận không? | 1 |
| q41 | keyword | Đơn hàng từ 10 triệu đến 100 triệu đồng cần tối thiểu bao nhiêu báo giá? | 1 |
| q42 | paraphrase | Đặt phòng họp rồi mà không đến thì phòng có bị nhả cho người khác không? | 1 |
| q43 | keyword | Công ty hỗ trợ bao nhiêu phần trăm phí gửi xe máy? | 1 |
| q44 | paraphrase | Đội dev có được copy database production về máy để test không? | 7 |
| q45 | keyword | Mã mới phải đạt độ bao phủ kiểm thử tối thiểu bao nhiêu? | 1 |
| q46 | paraphrase | Con tôi 4 tuổi đi du lịch cùng công ty thì có mất tiền không? | 1 |
| q47 | paraphrase | Mỗi ngày đi làm công ty hỗ trợ tiền cơm trưa bao nhiêu? | 1 |
| q48 | keyword | Thời hạn giải quyết khiếu nại ở mỗi bước là bao lâu? | 1 |
| q49 | paraphrase | Tôi có thể dán mã nguồn dự án vào chatbot miễn phí trên mạng để nhờ sửa lỗi không? | — |
| q50 | keyword | Điều kiện để được xét thăng chức là gì? | 1 |

</details>

## Định nghĩa

- **Recall@k**: tỷ lệ câu hỏi có ít nhất một chunk đúng trong top-k (hit rate). Mỗi câu hỏi ở đây có đúng một chunk đúng nên đây cũng là recall thông thường.
- **MRR@10**: trung bình 1/hạng của chunk đúng đầu tiên trong top-10 (0 nếu không có).
- **nDCG@5**: nDCG với độ liên quan nhị phân.
- **Độ trễ**: thời gian `pipeline.search()` cho một câu hỏi (đã nạp sẵn model), gồm tách từ, mã hoá câu hỏi, tìm kiếm, gộp và xếp hạng lại.
- Chi tiết từng truy vấn: `eval/runs/<chế độ>.jsonl`.
