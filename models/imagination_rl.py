import torch
import torch.nn as nn
import torch.nn.functional as F

class ImaginationRLModule(nn.Module):
    """想象世界中的 RL 训练：利用 Dynamics 模型进行规划"""
    
    def __init__(self, actor: nn.Module, critic: nn.Module, 
                 dynamics: nn.Module, horizon: int = 5):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.dynamics = dynamics
        self.horizon = horizon  # 想象的步数
    
    def imagine_rollout(self, z: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        在想象世界中执行 rollout，计算折扣回报
        Args:
            z: 初始状态 (batch_size, latent_dim)
            horizon: 规划深度
        Returns:
            cumulative_reward: 累积回报
        """
        if horizon is None:
            horizon = self.horizon
        
        cumulative_reward = 0.0
        current_z = z
        gamma = 0.99
        
        for step in range(horizon):
            # Actor 选择动作
            action = self.actor(current_z)
            
            # Critic 评估状态-动作对
            q_value = self.critic(current_z, action)
            
            # 累积奖励
            cumulative_reward += (gamma ** step) * q_value
            
            # Dynamics 模型预测下一状态
            current_z = self.dynamics(current_z, action)
        
        return cumulative_reward
    
    def compute_imagination_loss(self, z: torch.Tensor, 
                                 real_reward: torch.Tensor = None) -> torch.Tensor:
        """
        想象 rollout 损失：最大化想象世界中的累积回报
        """
        imagined_reward = self.imagine_rollout(z, self.horizon)
        
        if real_reward is not None:
            # 结合真实奖励和想象奖励
            loss = -(0.5 * real_reward + 0.5 * imagined_reward).mean()
        else:
            # 仅使用想象奖励
            loss = -imagined_reward.mean()
        
        return loss