import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """Actor 网络：从 latent z 和历史动作预测下一步动作"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 动作归一化到 [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            action: (batch_size, action_dim)
        """
        action = self.net(z)
        return action


class CriticNetwork(nn.Module):
    """Critic 网络：评估状态-动作对的价值"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
            action: (batch_size, action_dim)
        Returns:
            q_value: (batch_size, 1)
        """
        x = torch.cat([z, action], dim=1)
        q_value = self.net(x)
        return q_value


class OfflineRLModule(nn.Module):
    """离线 RL 训练模块"""
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.actor = ActorNetwork(latent_dim, action_dim)
        self.critic = CriticNetwork(latent_dim, action_dim)
        self.target_critic = CriticNetwork(latent_dim, action_dim)
        
        # 初始化目标网络
        self._copy_weights(self.critic, self.target_critic)
    
    @staticmethod
    def _copy_weights(source_net, target_net):
        """复制网络权重"""
        for src_param, tgt_param in zip(source_net.parameters(), target_net.parameters()):
            tgt_param.data.copy_(src_param.data)
    
    def compute_critic_loss(self, z: torch.Tensor, action: torch.Tensor, 
                           reward: torch.Tensor, next_z: torch.Tensor, 
                           done: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """计算 Critic 损失（TD-error）"""
        # 当前 Q 值
        q_current = self.critic(z, action)
        
        # 目标 Q 值
        with torch.no_grad():
            next_action = self.actor(next_z)
            q_next = self.target_critic(next_z, next_action)
            q_target = reward + gamma * q_next * (1 - done)
        
        # TD 损失
        loss = F.mse_loss(q_current, q_target)
        return loss
    
    def compute_actor_loss(self, z: torch.Tensor) -> torch.Tensor:
        """计算 Actor 损失"""
        action = self.actor(z)
        q_value = self.critic(z, action)
        # Actor 最大化 Q 值
        loss = -q_value.mean()
        return loss
    
    def update_target_network(self, tau: float = 0.005):
        """软更新目标网络"""
        for src_param, tgt_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)