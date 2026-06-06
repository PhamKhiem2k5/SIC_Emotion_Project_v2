import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from lstm_model import EmotionLSTM 

# 1. Thiết lập siêu tham số
SEQ_LENGTH = 15     
BATCH_SIZE = 32     
EPOCHS = 50         
LEARNING_RATE = 0.001

# 2. Xử lý dữ liệu (Data Preprocessing)
def load_and_preprocess_data(csv_file, seq_length):
    print("Đang đọc và chuẩn hóa dữ liệu từ CSV...")
    # Đọc không có header để tránh mất dòng dữ liệu đầu tiên
    df = pd.read_csv(csv_file, header=None)
    
    labels = df.iloc[:, 0].values # Cột đầu tiên là label
    raw_features = df.iloc[:, 1:].values # Các cột còn lại là features
    
    # Số lượng features = số điểm (num_points) * 2 (x, y)
    num_features = raw_features.shape[1]
    num_points = num_features // 2
    print(f"Phát hiện {num_points} điểm mốc (keypoints) -> Input Size = {num_features}")
    
    normalized_features = []
    
    for row in raw_features:
        # Tách X và Y theo cách dữ liệu được lưu trong main.py
        xs = row[:num_points]
        ys = row[num_points:]
        
        # Ghép thành các cặp (x_i, y_i)
        pts = np.column_stack((xs, ys))
        
        # 1. Lấy tọa độ mũi làm gốc (Điểm index 0)
        nose_x, nose_y = pts[0][0], pts[0][1]
        
        # 2. Dời gốc tọa độ
        pts_shifted = pts - [nose_x, nose_y]
        
        # 3. Tính chiều rộng khuôn mặt (Scale)
        face_width = np.max(xs) - np.min(xs) + 1e-6
        
        # 4. Chuẩn hóa tỷ lệ hóa
        pts_normalized = pts_shifted / face_width
        
        # Duỗi phẳng 1D và lưu lại
        normalized_features.append(pts_normalized.flatten())
        
    normalized_features = np.array(normalized_features)
    
    sequences = []
    seq_labels = []
    
    for i in range(len(normalized_features) - seq_length):
        if len(set(labels[i : i + seq_length])) == 1:
            sequences.append(normalized_features[i : i + seq_length])
            seq_labels.append(labels[i])
            
    return np.array(sequences), np.array(seq_labels), num_features

class EmotionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 3. Chuẩn bị luồng huấn luyện
X, y, input_size_dynamic = load_and_preprocess_data('emotion_dataset.csv', SEQ_LENGTH)
print(f"Tổng số chuỗi (sequences) hợp lệ thu được: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(EmotionDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 4. Khởi tạo Mô hình - Sử dụng input_size tự động
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionLSTM(input_size=input_size_dynamic, hidden_size=64, num_layers=1, num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Vòng lặp Huấn luyện (Training Loop)
print(f"Bắt đầu huấn luyện mô hình trên {device}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(sequences) 
        loss = criterion(outputs, labels) 
        
        loss.backward() 
        optimizer.step() 
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        
    train_acc = 100 * correct / len(y_train)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')

# 6. Lưu mô hình
torch.save(model.state_dict(), 'lstm_emotion.pth')
print("✅ Đã lưu trọng số mô hình vào file 'lstm_emotion.pth' thành công!")