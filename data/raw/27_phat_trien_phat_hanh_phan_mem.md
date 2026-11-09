# Quy trình phát triển và phát hành phần mềm

Mã văn bản: QT-KT-27 · Đơn vị ban hành: Khối Kỹ thuật · Hiệu lực: 01/01/2026

## 1. Quản lý mã nguồn

Nhánh chính của mọi kho mã được bảo vệ; không ai được đẩy mã trực tiếp lên nhánh chính. Mọi thay đổi đi qua yêu cầu hợp nhất (pull request) và cần ít nhất 1 người duyệt. Thay đổi liên quan đến xác thực, thanh toán hoặc dữ liệu cá nhân cần 2 người duyệt, trong đó có 1 kỹ sư cấp cao.

## 2. Kiểm thử tự động

Hệ thống tích hợp liên tục phải chạy thành công trước khi hợp nhất, bao gồm kiểm thử đơn vị, kiểm tra định dạng mã và quét lỗ hổng bảo mật. Mã mới phải đạt độ bao phủ kiểm thử tối thiểu 70%. Lỗ hổng mức nghiêm trọng phát hiện bởi công cụ quét phải được khắc phục trước khi phát hành.

## 3. Bí mật trong mã nguồn

Không đưa khóa API, mật khẩu, chứng chỉ hoặc chuỗi kết nối cơ sở dữ liệu vào kho mã. Các bí mật được lưu trong hệ thống quản lý bí mật và nạp qua biến môi trường khi triển khai. Nếu lỡ đẩy bí mật lên kho mã, kỹ sư phải thu hồi và tạo lại bí mật đó ngay, đồng thời báo cáo cho bộ phận Bảo mật thông tin.

## 4. Lịch phát hành

Phiên bản mới được phát hành lên môi trường production vào thứ Ba và thứ Năm hằng tuần. Không phát hành từ chiều thứ Sáu đến hết Chủ nhật, và trong giai đoạn đóng băng 2 tuần trước Tết Nguyên đán. Mỗi lần phát hành được triển khai dần theo các bước 5%, 50% rồi 100% người dùng, kèm theo dõi chỉ số lỗi.

## 5. Sửa lỗi khẩn cấp

Bản sửa lỗi khẩn cấp ngoài lịch phát hành cần được Trưởng nhóm kỹ thuật và kỹ sư trực sự cố phê duyệt. Sau khi khắc phục, nhóm phụ trách viết báo cáo phân tích sau sự cố trong vòng 5 ngày làm việc, tập trung vào nguyên nhân gốc rễ thay vì đổ lỗi cá nhân.
