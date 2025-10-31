import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10, conv_channels=32, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, conv_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear((conv_channels * 2) * 7 * 7 if in_ch == 1 else (conv_channels * 2) * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
