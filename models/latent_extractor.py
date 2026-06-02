import torch
import torch.nn as nn

class LatentExtractor(nn.Module):
    """从 VLA 隐层状态提取压缩的 latent representation"""
    
    def __init__(self, hidden_dim: int = 1024, latent_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 编码器：将隐层状态压缩为 latent z
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 解码器：从 latent z 恢复特征（用于验证）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
    
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
        Returns:
            z: (batch_size, latent_dim)
        """
        # 取平均池化得到全局表示
        global_feature = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        z = self.encoder(global_feature)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从 latent z 重构特征"""
        return self.decoder(z)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.encode(hidden_states)