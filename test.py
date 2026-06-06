import cv2
import torch
import numpy as np
import time
import psutil
import os
from collections import deque
from ultralytics import YOLO
from lstm_model import EmotionLSTM # Đảm bảo file lstm_model.py ở cùng thư mục

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_WEIGHTS = "yolo11n-pose.pt"
LSTM_WEIGHTS = "lstm_emotion.pth"
SEQ_LENGTH = 15 # Phải khớp với cấu hình lúc train

# Dictionary map label với tên cảm xúc
EMOTION_DICT = {
    0: "Binh thuong (Neutral)",
    1: "Cang thang (Stress)",
    2: "Met moi (Fatigue)",
    3: "Vui ve (Happy)"
}

# --- 1. KHỞI TẠO MÔ HÌNH ---
print("Đang tải mô hình YOLO và LSTM...")
# YOLO Model
yolo_model = YOLO(MODEL_WEIGHTS)

# LSTM Model
# Theo main.py, input là 5 điểm (10 tọa độ X, Y)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = EmotionLSTM(input_size=10, hidden_size=64, num_layers=1, num_classes=4).to(device)

# Load weights an toàn
try:
    lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=device))
    print("✅ Đã load weights LSTM thành công!")
except Exception as e:
    print(f"❌ Lỗi load weights LSTM: {e}")
    print("Vui lòng đảm bảo file lstm_emotion.pth tồn tại và khớp cấu trúc mạng.")
    exit()

lstm_model.eval() # Chế độ dự đoán

# --- 2. CHUẨN BỊ BUFFER & THÔNG SỐ ---
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=SEQ_LENGTH)
current_emotion = "Dang phan tich..."
prev_time = 0

print("--- HỆ THỐNG PHÂN TÍCH CẢM XÚC (DEMO) ---")
print("Nhấn 'q' để thoát.")

# --- 3. VÒNG LẶP XỬ LÝ (REAL-TIME) ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Tính toán FPS và Hardware Usage
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # Chạy YOLO11-pose
    results = yolo_model(frame, verbose=False)
    
    # --- XỬ LÝ NHẬN DIỆN & VẼ HÌNH TRỰC QUAN ---
    if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 0:
        # Lấy 5 điểm mốc (face keypoints)
        keypoints = results[0].keypoints.xy[0][:5].cpu().numpy()
        
        if len(keypoints) == 5:
            # 1. Vẽ khung UI nhận diện khuôn mặt (Sao chép từ main.py)
            xs = [kp[0] for kp in keypoints]
            ys = [kp[1] for kp in keypoints]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            face_width = max_x - min_x + 1e-6 # Tránh lỗi chia 0
            
            center_x, center_y = int((min_x + max_x) / 2), int((min_y + max_y) / 2)
            box_w, box_h = int(face_width * 1.6), int(face_width * 1.8)
            
            x1 = max(0, int(center_x - box_w / 2))
            y1 = max(0, int(center_y - box_h / 2 * 1.2))
            x2 = min(frame.shape[1], int(center_x + box_w / 2))
            y2 = min(frame.shape[0], int(center_y + box_h / 2 * 0.8))
            
            # Vẽ Box và Điểm mốc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # --- PREPROCESSING (CHUẨN HÓA DỮ LIỆU) ---
            # Lưu ý: Cần chuẩn hóa GIỐNG HỆT quy trình trong train.py
            # 1. Ghép X và Y thành các cặp tọa độ
            pts = np.column_stack((xs, ys))
            
            # 2. Lấy tọa độ mũi làm gốc (Index 0)
            nose_x, nose_y = pts[0][0], pts[0][1]
            
            # 3. Dời gốc tọa độ
            pts_shifted = pts - [nose_x, nose_y]
            
            # 4. Chuẩn hóa tỷ lệ theo chiều rộng khuôn mặt
            pts_normalized = pts_shifted / face_width
            
            # 5. Duỗi phẳng thành mảng 1D (10 phần tử) và đưa vào Buffer
            sequence_buffer.append(pts_normalized.flatten())

            # --- DỰ ĐOÁN CẢM XÚC (INFERENCE) ---
            if len(sequence_buffer) == SEQ_LENGTH:
                # Chuyển đổi Buffer thành Tensor: shape (Batch=1, Seq=15, Features=10)
                input_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    output = lstm_model(input_tensor)
                    _, predicted_class = torch.max(output.data, 1)
                    current_emotion = EMOTION_DICT[predicted_class.item()]

    # --- HIỂN THỊ THÔNG TIN LÊN MÀN HÌNH ---
    # 1. Thông số phần cứng (Góc trái trên)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_usage}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"RAM: {ram_usage:.1f} MB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # 2. Trạng thái cảm xúc (Góc trái dưới)
    # Xác định màu sắc theo cảm xúc (Vui = Xanh lá, Căng thẳng = Đỏ, Mệt = Cam)
    color = (0, 255, 0)
    if "Stress" in current_emotion: color = (0, 0, 255)
    elif "Fatigue" in current_emotion: color = (0, 165, 255)
    
    cv2.putText(frame, f"Cam xuc: {current_emotion}", (10, frame.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 3. Hiển thị % Buffer đầy (Để demo trực quan)
    buffer_percent = (len(sequence_buffer) / SEQ_LENGTH) * 100
    cv2.putText(frame, f"Buffer: {int(buffer_percent)}%", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("SIC Emotion Real-time Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()