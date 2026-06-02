import torch
import torch.nn as nn

class DynamicsMLPModel(nn.Module):
    """简单 MLP Dynamics 模型：预测下一步状态"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
            action: (batch_size, action_dim)
        Returns:
            next_z: (batch_size, latent_dim)
        """
        x = torch.cat([z, action], dim=1)
        next_z = self.net(x)
        return next_z


class DynamicsEmu3Model(nn.Module):
    """使用 Emu3 VisionTokenizer 作为 Dynamics 模型"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 2, 
                 emu3_model_path: str = None):
        super().__init__()
        # 这里可以集成 Emu3 的预训练权重进行图像预测
        # latent z -> 预测下一帧图像 tokens
        self.projection = nn.Linear(latent_dim + action_dim, latent_dim)
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一状态或图像 tokens"""
        combined = torch.cat([z, action], dim=1)
        next_z = self.projection(combined)
        return next_z


class DynamicsModule(nn.Module):
    """Dynamics 模块选择器"""
    
    def __init__(self, model_type: str = "mlp", latent_dim: int = 256, 
                 action_dim: int = 2, emu3_path: Optional[str] = None):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "mlp":
            self.model = DynamicsMLPModel(latent_dim, action_dim)
        elif model_type == "emu3":
            self.model = DynamicsEmu3Model(latent_dim, action_dim, emu3_path)
        else:
            raise ValueError(f"Unknown dynamics model type: {model_type}")
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(z, action)