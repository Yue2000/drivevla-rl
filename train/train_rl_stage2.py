"""
阶段 2：加入 World Model (MLP Dynamics)
训练：Latent Extractor + Actor + Critic + Dynamics
不训练：Imagination RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RLTrainerStage2:
    """阶段 2：RL + Dynamics 模型训练"""
    
    def __init__(self, args, training_args):
        self.args = args
        self.training_args = training_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的 VLA 模型
        logger.info(f"Loading VLA model from {args.model_name_or_path}")
        from transformers import AutoModel
        self.vla_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 冻结 VLA Backbone
        for param in self.vla_model.parameters():
            param.requires_grad = False
        logger.info("✓ VLA Backbone frozen")
        
        # 初始化模块
        from models.latent_extractor import LatentExtractor
        from models.rl_policy import OfflineRLModule
        from models.dynamics_model import DynamicsModule
        
        self.latent_extractor = LatentExtractor(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        ).to(self.device)
        
        self.rl_module = OfflineRLModule(
            latent_dim=args.latent_dim,
            action_dim=args.action_dim
        ).to(self.device)
        
        # 关键：加入 Dynamics 模型（MLP 类型）
        self.dynamics = DynamicsModule(
            model_type="mlp",  # 使用 MLP Dynamics
            latent_dim=args.latent_dim,
            action_dim=args.action_dim
        ).to(self.device)
        
        # 初始化优化器
        self.actor_optim = torch.optim.Adam(
            self.rl_module.actor.parameters(),
            lr=args.actor_lr
        )
        self.critic_optim = torch.optim.Adam(
            self.rl_module.critic.parameters(),
            lr=args.critic_lr
        )
        self.latent_optim = torch.optim.Adam(
            self.latent_extractor.parameters(),
            lr=args.latent_lr
        )
        self.dynamics_optim = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=args.dynamics_lr
        )
        
        logger.info("✓ Stage 2 RL Pipeline initialized (Actor + Critic + MLP Dynamics)")
    
    def extract_latent(self, input_ids, attention_mask):
        """从 VLA 提取 latent z"""
        with torch.no_grad():
            outputs = self.vla_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        z = self.latent_extractor.encode(hidden_states)
        return z
    
    def training_step(self, batch):
        """
        阶段 2 训练步骤：RL + Dynamics 模型
        
        关键创新点：
        1. 使用 Dynamics 模型预测下一状态 z'
        2. 用真实的下一状态 z_real 监督 Dynamics
        3. 用预测的 z' 辅助 RL 训练（可选）
        """
        # 准备数据
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        action = batch['action'].to(self.device)
        
        reward = batch.get('reward', torch.zeros(action.size(0)).to(self.device))
        done = batch.get('done', torch.zeros(action.size(0)).to(self.device))
        
        next_input_ids = batch.get('next_input_ids', input_ids)
        next_attention_mask = batch.get('next_attention_mask', attention_mask)
        
        # ============ 提取 Latent ============
        z = self.extract_latent(input_ids, attention_mask)
        next_z_real = self.extract_latent(next_input_ids, next_attention_mask)
        
        # ============ Critic 损失 ============
        q_current = self.rl_module.critic(z, action)
        
        with torch.no_grad():
            next_action = self.rl_module.actor(next_z_real)
            q_next = self.rl_module.target_critic(next_z_real, next_action)
            q_target = reward.unsqueeze(1) + self.args.gamma * q_next * (1 - done.unsqueeze(1))
        
        critic_loss = F.mse_loss(q_current, q_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.rl_module.critic.parameters(), 1.0)
        self.critic_optim.step()
        
        # ============ Actor 损失 ============
        action_pred = self.rl_module.actor(z)
        q_value = self.rl_module.critic(z, action_pred)
        actor_loss = -q_value.mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.rl_module.actor.parameters(), 1.0)
        self.actor_optim.step()
        
        # ============ Dynamics 损失（关键 - 阶段 2 新增）============
        # 预测下一状态
        next_z_pred = self.dynamics(z, action)
        
        # 用真实的下一状态监督 Dynamics
        dynamics_loss = F.mse_loss(next_z_pred, next_z_real.detach())
        
        self.dynamics_optim.zero_grad()
        dynamics_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.dynamics_optim.step()
        
        # ============ Latent 提取器优化 ============
        z_reconstructed = self.latent_extractor.decode(z)
        latent_loss = F.mse_loss(z_reconstructed, z.detach())
        
        self.latent_optim.zero_grad()
        latent_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_extractor.parameters(), 1.0)
        self.latent_optim.step()
        
        # 目标网络软更新
        self.rl_module.update_target_network(tau=self.args.tau)
        
        losses = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'dynamics_loss': dynamics_loss.item(),  # 新增
            'latent_loss': latent_loss.item(),
            'total_loss': (critic_loss + actor_loss + dynamics_loss + latent_loss).item()
        }
        
        return losses
    
    def train(self, train_dataset, num_epochs=10):
        """完整训练循环 - 阶段 2"""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.training_args.dataloader_num_workers
        )
        
        total_steps = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage 2 - Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            epoch_losses = {k: 0.0 for k in ['critic_loss', 'actor_loss', 'dynamics_loss', 'latent_loss', 'total_loss']}
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    losses = self.training_step(batch)
                    
                    for k, v in losses.items():
                        epoch_losses[k] += v
                    
                    total_steps += 1
                    
                    if (batch_idx + 1) % self.training_args.logging_steps == 0:
                        avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                        log_msg = f"Step {total_steps} [{batch_idx + 1}/{len(train_loader)}]\n"
                        for k, v in avg_losses.items():
                            log_msg += f"  {k}: {v:.4f}\n"
                        logger.info(log_msg)
                    
                    if total_steps % self.training_args.save_steps == 0:
                        self.save_checkpoint(epoch, total_steps, stage=2)
                    
                    if total_steps >= self.training_args.max_steps:
                        logger.info(f"✓ Reached max steps ({self.training_args.max_steps})")
                        return
                
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    raise
    
    def save_checkpoint(self, epoch, step, stage=2):
        """保存检查点"""
        checkpoint_dir = Path(self.training_args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.latent_extractor.state_dict(), checkpoint_dir / "latent_extractor.pt")
        torch.save(self.rl_module.actor.state_dict(), checkpoint_dir / "actor.pt")
        torch.save(self.rl_module.critic.state_dict(), checkpoint_dir / "critic.pt")
        torch.save(self.dynamics.state_dict(), checkpoint_dir / "dynamics.pt")  # 新增
        
        logger.info(f"✓ Stage {stage} Checkpoint saved to {checkpoint_dir}")


def main():
    from transformers import HfArgumentParser, TrainingArguments
    from dataclasses import dataclass, field
    import logging
    
    @dataclass
    class RLTrainingArguments:
        actor_lr: float = field(default=3e-4)
        critic_lr: float = field(default=3e-4)
        latent_lr: float = field(default=1e-4)
        dynamics_lr: float = field(default=1e-4)  # 新增
        latent_dim: int = field(default=256)
        hidden_dim: int = field(default=1024)
        action_dim: int = field(default=3)
        gamma: float = field(default=0.99)
        tau: float = field(default=0.005)
    
    parser = HfArgumentParser((TrainingArguments, RLTrainingArguments))
    training_args, rl_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(level=logging.INFO)
    logger_main = logging.getLogger(__name__)
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = RLTrainerStage2(rl_args, training_args)
    
    logger_main.info(f"Loading dataset from {training_args.data_path}")
    from utils.datasets import Emu3DrivingVAVA_AR_Dataset
    train_dataset = Emu3DrivingVAVA_AR_Dataset(training_args, tokenizer=trainer.vla_model)
    
    trainer.train(train_dataset, num_epochs=10)
    
    final_dir = Path(training_args.output_dir) / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(trainer.latent_extractor.state_dict(), final_dir / "latent_extractor.pt")
    torch.save(trainer.rl_module.actor.state_dict(), final_dir / "actor.pt")
    torch.save(trainer.dynamics.state_dict(), final_dir / "dynamics.pt")
    logger_main.info(f"✓ Final model saved to {final_dir}")


if __name__ == "__main__":
    main()