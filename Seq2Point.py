import torch
import torch.nn as nn

class Seq2Point(nn.Module):
    def __init__(self, window_size):
        super(Seq2Point, self).__init__()
        self.window_size = window_size
        
        # Architecture based on Zhang et al. (2018) for Seq2Point
        # Input: (Batch, 1, Window)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(50, 50, kernel_size=5, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened features size
        # L_out = L_in - sum(kernel_size - 1)
        # 10-1=9, 8-1=7, 6-1=5, 5-1=4, 5-1=4
        # Total reduction = 29
        self.flatten_dim = 50 * (window_size - 29)
        
        if self.flatten_dim <= 0:
            raise ValueError(f"Window size {window_size} is too small for this architecture. Minimum 30 required.")

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        # x: (Batch, 1, Window)
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x
