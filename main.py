import cv2
import csv
import time
import psutil
import os
from ultralytics import YOLO

# Khởi tạo mô hình YOLO11-pose Nano
model = YOLO("yolo11n-pose.pt")

# Tên file lưu dữ liệu
csv_filename = "emotion_dataset.csv"

# Khởi tạo file CSV và viết tiêu đề (Headers)
with open(csv_filename, mode='a', newline='') as f:
    writer = csv.writer(f)
    # Chỉ ghi header nếu file trống
    if f.tell() == 0:
        headers = ['label'] + [f'x{i}' for i in range(5)] + [f'y{i}' for i in range(5)]
        writer.writerow(headers)

# Mở webcam
cap = cv2.VideoCapture(0)

is_recording = False
record_count = 0
current_label = -1
MAX_FRAMES = 300

# Khởi tạo biến cho FPS
prev_time = 0

print("--- HỆ THỐNG THU THẬP DỮ LIỆU ---")
print("Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tính toán FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Tính toán CPU & RAM (Sử dụng psutil)
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # Đổi sang MB
    
    # Chạy YOLO11-pose để trích xuất không gian
    results = model(frame, verbose=False)
    keypoints_data = []
    
    # Trích xuất tọa độ điểm mốc và vẽ nhận dạng khuôn mặt
    if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 0:
        keypoints = results[0].keypoints.xy[0][:5].cpu().numpy()
        
        if len(keypoints) == 5:
            # Tách riêng mảng tọa độ X và Y
            xs = [kp[0] for kp in keypoints]
            ys = [kp[1] for kp in keypoints]
            keypoints_data = xs + ys
            
            # --- VẼ Ô VUÔNG "VỪA CÁI ĐẦU" TỪ ĐIỂM MỐC ---
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Chiều rộng khuôn mặt (khoảng cách xấp xỉ giữa 2 tai/mắt)
            face_width = max_x - min_x
            
            # Xác định tâm của khuôn mặt
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)
            
            # Tính toán kích thước hộp (Mở rộng 1.6 lần chiều rộng để lấy cả đầu)
            box_w = int(face_width * 1.6)
            box_h = int(face_width * 1.8) # Chiều cao dài hơn để bao trọn từ đỉnh trán xuống cằm
            
            # Đảm bảo tọa độ không bị tràn ra ngoài khung hình camera
            x1 = max(0, int(center_x - box_w / 2))
            y1 = max(0, int(center_y - box_h / 2 * 1.2)) # Đẩy trọng tâm lên trên một chút để lấy trán
            x2 = min(frame.shape[1], int(center_x + box_w / 2))
            y2 = min(frame.shape[0], int(center_y + box_h / 2 * 0.8)) # Cắt bớt phần thừa dưới cằm
            
            # Vẽ ô vuông vừa đầu màu xanh dương
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Vẽ các điểm nhận dạng khuôn mặt (màu xanh lá)
            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Xử lý Ghi dữ liệu vào file CSV
    if is_recording and len(keypoints_data) == 10:
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_label] + keypoints_data)
        
        record_count += 1
        # Hiển thị trạng thái đang ghi hình
        cv2.putText(frame, f"REC: {record_count}/{MAX_FRAMES} (Label: {current_label})", 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if record_count >= MAX_FRAMES:
            is_recording = False
            print(f"✅ Đã ghi xong {MAX_FRAMES} frames cho nhãn {current_label}.")
    
    # Hiển thị thông số phần cứng trên góc trái màn hình
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_usage}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"RAM: {ram_usage:.1f} MB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Hiển thị hướng dẫn phím tắt ở góc dưới cùng
    cv2.putText(frame, "Press 0-3 to Record. 'q' to quit.", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Data Collection Mode", frame)
    
    # Nhận diện phím bấm
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2'), ord('3')] and not is_recording:
        current_label = int(chr(key))
        is_recording = True
        record_count = 0
        print(f"🔴 Bắt đầu ghi hình cho nhãn {current_label}...")

cap.release()
cv2.destroyAllWindows()