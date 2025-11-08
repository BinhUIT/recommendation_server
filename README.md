##**Server đơn giản dùng để xác định hình dạng cơ thể, độ tuổi, giới tính từ một hình ảnh**
- **Chạy server:**
      - Cài đặt Python 3.10.10
      - Cd tới thư mục dự án
      - Chạy lệnh "pip install -r requirement.txt"
      - Chạy lệnh "python server.py"
- Sau khi chạy thành công, server sẽ lắng nghe ở "localhost:8082/"
- Test
- Gửi 1 POST request ( với param “file” là ảnh )
- Kết quả
![Result]( https://github.com/BinhUIT/web_ban_hang/blob/master/project_images/Screenshot%202025-11-08%20144418.png?raw=true)
- Json trả về có dạng
    {
        "age": 24,
        "gender": 0,
        "shape": "thin",
        "body_shape": "Rectangle"
    }
