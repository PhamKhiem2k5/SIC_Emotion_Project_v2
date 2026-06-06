import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, num_classes=4):
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Thiết lập mạng LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Lớp Fully Connected để đưa ra quyết định phân loại cảm xúc
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Khởi tạo trạng thái ẩn (hidden state) và trạng thái tế bào (cell state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Đẩy chuỗi dữ liệu thời gian qua mạng LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Chỉ lấy kết quả đầu ra của khung hình cuối cùng trong chuỗi để dự đoán
        out = self.fc(out[:, -1, :])
        return out