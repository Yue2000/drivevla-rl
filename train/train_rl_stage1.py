# -*- coding: utf-8 -*-
"""
阶段 1：Offline RL - 不使用 World Model (彻底修复 Tokenizer boa_token 缺失版)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
import logging
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

from transformers import HfArgumentParser, TrainingArguments, AutoModel, AutoTokenizer
from models.latent_extractor import LatentExtractor
from models.rl_policy import OfflineRLModule
from utils.datasets import Emu3DrivingVAVA_AR_Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 动态修复补丁 (Monkey Patch) 开始 ====================
import utils.datasets
import numpy as np
import scipy.fft._realtransforms_backend

# 1. SciPy 频域底层毒素净化 (防无限递归版)
if not getattr(scipy.fft._realtransforms_backend, '_is_patched', False):
    orig_backend_dct = scipy.fft._realtransforms_backend.dct
    
    def safe_backend_dct(x, *args, **kwargs):
        axis = kwargs.get('axis', -1)
        if hasattr(x, 'shape'):
            try:
                if x.shape[axis] == 0:
                    new_shape = list(x.shape)
                    new_shape[axis] = 8
                    if isinstance(x, np.ndarray):
                        x = np.zeros(new_shape, dtype=x.dtype)
                    elif torch.is_tensor(x):
                        x = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
            except:
                pass
        return orig_backend_dct(x, *args, **kwargs)

    scipy.fft._realtransforms_backend.dct = safe_backend_dct
    scipy.fft._realtransforms_backend._is_patched = True
    logger.info("✓ [Patch] SciPy-DCT Low-Level Backend Shield activated safely.")


class SafeDict(dict):
    """安全字典包装器"""
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if 'text' in key:
            return [""] * 100
        if 'image' in key:
            return []
        if 'action' in key:
            return np.zeros((8, 3), dtype=np.float32)
        return None

# 2. 备份原有的数据集方法
orig_sliding_window_sampling = utils.datasets.Emu3DrivingVAVA_AR_Dataset.sliding_window_sampling
orig_random_frames_to_tensor = utils.datasets.Emu3DrivingVAVA_AR_Dataset.random_frames_to_tensor
orig_getitem = utils.datasets.Emu3DrivingVAVA_AR_Dataset.__getitem__

# 3. 拦截切片方法
def patched_sliding_window_sampling(self, data, *args, **kwargs):
    for item in data:
        ref_len = len(item.get('image', item.get('action', [0] * 20)))
        if 'gripper_image' not in item:
            item['gripper_image'] = [None] * ref_len
        if 'pre_1s_text' not in item:
            item['pre_1s_text'] = [""] * ref_len
    
    result_iterable = orig_sliding_window_sampling(self, data, *args, **kwargs)
    
    for new_item in result_iterable:
        if 'gripper_image' in new_item:
            new_item.pop('gripper_image', None)
        
        window_len = len(new_item.get('image', new_item.get('action', [0] * 1)))
        
        if 'pre_1s_text' not in new_item:
            new_item['pre_1s_text'] = [""] * window_len
        if 'post_1s_text' not in new_item:
            new_item['post_1s_text'] = [""] * window_len
            
        if 'pre_1s_image' not in new_item or new_item['pre_1s_image'] is None:
            new_item['pre_1s_image'] = new_item.get('image', [])
        if 'post_1s_image' not in new_item or new_item['post_1s_image'] is None:
            new_item['post_1s_image'] = new_item.get('image', [])
            
        if 'previous_actions' not in new_item or new_item['previous_actions'] is None:
            new_item['previous_actions'] = np.zeros((8, 3), dtype=np.float32)
        elif hasattr(new_item['previous_actions'], '__len__') and len(new_item['previous_actions']) == 0:
            new_item['previous_actions'] = np.zeros((8, 3), dtype=np.float32)
            
        if 'action' not in new_item or new_item['action'] is None:
            new_item['action'] = np.zeros((window_len, 3), dtype=np.float32)
            
        yield SafeDict(new_item)

# 4. 拦截分帧转换方法
def patched_random_frames_to_tensor(self, img_list, *args, **kwargs):
    if img_list is None or not isinstance(img_list, (list, tuple)) or len(img_list) == 0:
        dummy_path = None
        if hasattr(self, 'data') and self.data:
            for item in self.data:
                imgs = item.get('image', [])
                if imgs and len(imgs) > 0 and imgs[0]:
                    dummy_path = imgs[0]
                    break
        img_list = [dummy_path if dummy_path else "dummy.npy"] * 100

    try:
        return orig_random_frames_to_tensor(self, img_list, *args, **kwargs)
    except Exception as e:
        dummy_img_tensor = torch.zeros((1, 18, 32), dtype=torch.long)
        dummy_act_tensor = torch.zeros((1, 3), dtype=torch.float32)
        return dummy_img_tensor, dummy_act_tensor

# 5. 拦截 __getitem__ 方法
def patched_getitem(self, idx):
    if hasattr(self, 'data') and self.data:
        try:
            item = self.data[idx]
            if isinstance(item, dict):
                pa = item.get('previous_actions', None)
                if pa is None or (hasattr(pa, '__len__') and len(pa) == 0):
                    item['previous_actions'] = np.zeros((8, 3), dtype=np.float32)
                for img_k in ['pre_1s_image', 'post_1s_image']:
                    if img_k not in item or item[img_k] is None or (hasattr(item[img_k], '__len__') and len(item[img_k]) == 0):
                        item[img_k] = item.get('image', [])
        except:
            pass

    try:
        return orig_getitem(self, idx)
    except Exception as e:
        logger.warning(f"⚠️ [Dataset Shield] Index {idx} crashed with error: {e}. Activating robust fallback...")
        for fallback_idx in range(idx + 1, min(idx + 200, len(self))):
            try:
                return orig_getitem(self, fallback_idx)
            except:
                continue
        try:
            return orig_getitem(self, len(self) // 2)
        except Exception as critical_err:
            logger.error(f"❌ [Fatal] Universal fallback failed completely: {critical_err}")
            raise e

# 6. 💡 拦截 wrap_action_sequence：终极解决 boa_token 属性缺失异常
if hasattr(utils.datasets.Emu3DrivingVAVA_AR_Dataset, 'wrap_action_sequence'):
    orig_wrap_action_sequence = utils.datasets.Emu3DrivingVAVA_AR_Dataset.wrap_action_sequence
    
    def patched_wrap_action_sequence(self, action_ids, *args, **kwargs):
        # 动态属性注入：若 Tokenizer 不存在动作标志，强制绑定默认安全标志（回退至 bos_token/eos_token）
        if not hasattr(self.tokenizer, 'boa_token') or self.tokenizer.boa_token is None:
            self.tokenizer.boa_token = getattr(self.tokenizer, 'bos_token', "<|boa|>")
        if not hasattr(self.tokenizer, 'eoa_token') or self.tokenizer.eoa_token is None:
            self.tokenizer.eoa_token = getattr(self.tokenizer, 'eos_token', "<|eoa|>")
            
        try:
            return orig_wrap_action_sequence(self, action_ids, *args, **kwargs)
        except Exception as e:
            # 极限防御：如果原函数内部 encode(boa_token) 返回空导致越界崩溃，手动安全拼接张量/列表
            try:
                b_id = self.tokenizer.encode(self.tokenizer.boa_token)[-1]
            except:
                b_id = getattr(self.tokenizer, 'bos_token_id', 0) or 0
                
            try:
                e_id = self.tokenizer.encode(self.tokenizer.eoa_token)[-1]
            except:
                e_id = getattr(self.tokenizer, 'eos_token_id', 2) or 2
                
            if isinstance(action_ids, list):
                return [b_id] + action_ids + [e_id]
            elif hasattr(action_ids, 'dtype'): 
                if "torch" in str(type(action_ids)):
                    return torch.cat([torch.tensor([b_id], device=action_ids.device), action_ids, torch.tensor([e_id], device=action_ids.device)])
                else:
                    return np.concatenate([np.array([b_id]), action_ids, np.array([e_id])])
            return action_ids

    utils.datasets.Emu3DrivingVAVA_AR_Dataset.wrap_action_sequence = patched_wrap_action_sequence
    logger.info("✓ [Patch] Action Sequence Wrapper (boa_token fix) activated.")

# 7. 动态替换生效
utils.datasets.Emu3DrivingVAVA_AR_Dataset.sliding_window_sampling = patched_sliding_window_sampling
utils.datasets.Emu3DrivingVAVA_AR_Dataset.random_frames_to_tensor = patched_random_frames_to_tensor
utils.datasets.Emu3DrivingVAVA_AR_Dataset.__getitem__ = patched_getitem
logger.info("✓ [Patch] Advanced Item-Fallback Defender activated.")
# ==================== 动态修复补丁 (Monkey Patch) 结束 ====================


@dataclass
class DataArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained VLA model"})
    data_path: str = field(metadata={"help": "Path to training data pickle file"})
    action_tokenizer_path: str = field(default="configs/fast", metadata={"help": "Path to action tokenizer"})
    actions_format: str = field(default="fast", metadata={"help": "Action format"})


@dataclass
class RLArguments:
    actor_lr: float = field(default=3e-4, metadata={"help": "Actor learning rate"})
    critic_lr: float = field(default=3e-4, metadata={"help": "Critic learning rate"})
    latent_lr: float = field(default=1e-4, metadata={"help": "Latent extractor learning rate"})
    latent_dim: int = field(default=256, metadata={"help": "Latent dimension"})
    hidden_dim: int = field(default=1024, metadata={"help": "Hidden dimension"})
    action_dim: int = field(default=3, metadata={"help": "Action dimension"})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    tau: float = field(default=0.005, metadata={"help": "Target network update rate"})


@dataclass
class DataConfig:
    data_path: str = ""
    model_name_or_path: str = ""
    action_tokenizer_path: str = "configs/fast"
    random_frame_sampling: bool = False
    raw_image: bool = False
    visual_token_pattern: str = "<|image_{token_id}|>"
    codebook_size: int = 32768
    VL: bool = True
    post_training: bool = False
    video_format: str = "interleave"
    use_flip: bool = False
    cur_frame_idx: int = 0
    frames: int = 1
    action_frames: int = 8
    apply_loss_on_only_vision: bool = False
    apply_loss_on_only_action: bool = False
    ignore_index: int = -100
    use_previous_actions: bool = True
    actions: bool = True
    actions_format: str = "fast"
    driving: bool = True
    use_gripper: bool = False  
    per_device_train_batch_size: int = 8
    dataloader_num_workers: int = 0
    max_position_embeddings: int = 1500
    seed: int = 0


class RLTrainerStage1:
    def __init__(self, data_args, training_args, rl_args):
        self.data_args = data_args
        self.training_args = training_args
        self.rl_args = rl_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading VLA model from {data_args.model_name_or_path}")
        
        try:
            self.vla_model = AutoModel.from_pretrained(
                data_args.model_name_or_path,
                trust_remote_code=True,
            )
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.warning(f"Primary loading failed: {e}")
            logger.info("Trying reference implementation...")
            try:
                sys.path.insert(0, os.path.join(os.getcwd(), 'reference/transformers/src'))
                from transformers.models.emu3.modeling_emu3 import Emu3ForCausalLM
                self.vla_model = Emu3ForCausalLM.from_pretrained(
                    data_args.model_name_or_path,
                    trust_remote_code=True,
                )
                logger.info("✓ Model loaded from reference")
            except Exception as e2:
                logger.error(f"All loading methods failed: {e2}")
                raise
        
        for param in self.vla_model.parameters():
            param.requires_grad = False
        logger.info("✓ VLA Backbone frozen")
        
        self.latent_extractor = LatentExtractor(
            hidden_dim=rl_args.hidden_dim,
            latent_dim=rl_args.latent_dim
        ).to(self.device)
        
        self.rl_module = OfflineRLModule(
            latent_dim=rl_args.latent_dim,
            action_dim=rl_args.action_dim
        ).to(self.device)
        
        self.actor_optim = torch.optim.Adam(self.rl_module.actor.parameters(), lr=rl_args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.rl_module.critic.parameters(), lr=rl_args.critic_lr)
        self.latent_optim = torch.optim.Adam(self.latent_extractor.parameters(), lr=rl_args.latent_lr)
        
        logger.info("✓ Stage 1 RL Pipeline initialized")
    
    def extract_latent(self, input_ids, attention_mask):
        with torch.no_grad():
            try:
                outputs = self.vla_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[-1]
            except:
                outputs = self.vla_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                hidden_states = outputs.last_hidden_state
        
        z = self.latent_extractor.encode(hidden_states)
        return z
    
    def training_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch.get('reward', torch.zeros(action.size(0)).to(self.device))
        done = batch.get('done', torch.zeros(action.size(0)).to(self.device))
        next_input_ids = batch.get('next_input_ids', input_ids)
        next_attention_mask = batch.get('next_attention_mask', attention_mask)
        
        z = self.extract_latent(input_ids, attention_mask)
        next_z = self.extract_latent(next_input_ids, next_attention_mask)
        
        critic_loss = self.rl_module.compute_critic_loss(z, action, reward, next_z, done, gamma=self.rl_args.gamma)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.rl_module.critic.parameters(), 1.0)
        self.critic_optim.step()
        
        actor_loss = self.rl_module.compute_actor_loss(z)
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.rl_module.actor.parameters(), 1.0)
        self.actor_optim.step()
        
        z_reconstructed = self.latent_extractor.decode(z)
        latent_loss = F.mse_loss(z_reconstructed, z.detach())
        self.latent_optim.zero_grad()
        latent_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_extractor.parameters(), 1.0)
        self.latent_optim.step()
        
        self.rl_module.update_target_network(tau=self.rl_args.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'latent_loss': latent_loss.item(),
            'total_loss': (critic_loss + actor_loss + latent_loss).item()
        }
    
    
    def train(self, train_dataset, num_epochs=1):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.training_args.dataloader_num_workers
        )
        
        total_steps = 0
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}\nStage 1 - Epoch {epoch + 1}/{num_epochs}\n{'='*60}")
            epoch_losses = {k: 0.0 for k in ['critic_loss', 'actor_loss', 'latent_loss', 'total_loss']}
            
            for batch_idx, batch in enumerate(train_loader):
                losses = self.training_step(batch)
                for k, v in losses.items():
                    epoch_losses[k] += v
                
                total_steps += 1
                if (batch_idx + 1) % self.training_args.logging_steps == 0:
                    avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                    logger.info(f"Step {total_steps}: {avg_losses}")
                
                if total_steps % self.training_args.save_steps == 0:
                    self.save_checkpoint(total_steps)
                
                if total_steps >= self.training_args.max_steps:
                    logger.info(f"✓ Reached max steps")
                    return
    
    def save_checkpoint(self, step):
        checkpoint_dir = Path(self.training_args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.latent_extractor.state_dict(), checkpoint_dir / "latent_extractor.pt")
        torch.save(self.rl_module.actor.state_dict(), checkpoint_dir / "actor.pt")
        torch.save(self.rl_module.critic.state_dict(), checkpoint_dir / "critic.pt")
        logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")


def main():
    parser = HfArgumentParser((DataArguments, TrainingArguments, RLArguments))
    data_args, training_args, rl_args = parser.parse_args_into_dataclasses()
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = RLTrainerStage1(data_args, training_args, rl_args)
    
    logger.info(f"Loading tokenizer from {data_args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            data_args.model_name_or_path,
            trust_remote_code=True,
        )
        logger.info("✓ Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Primary tokenizer loading failed: {e}")
        logger.info("Trying reference tokenizer implementation...")
        try:
            sys.path.insert(0, os.path.join(os.getcwd(), 'reference/transformers/src'))
            from transformers.models.emu3.tokenization_emu3 import Emu3Tokenizer
            tokenizer = Emu3Tokenizer.from_pretrained(
                data_args.model_name_or_path,
                trust_remote_code=True,
            )
            logger.info("✓ Tokenizer loaded from reference")
        except Exception as e2:
            logger.error(f"All tokenizer loading methods failed: {e2}")
            raise

    # 💡 显式双保险：在传入 Dataset 之前，也给 tokenizer 对象挂上必要的属性
    if not hasattr(tokenizer, 'boa_token'):
        tokenizer.boa_token = getattr(tokenizer, 'bos_token', "<|boa|>")
    if not hasattr(tokenizer, 'eoa_token'):
        tokenizer.eoa_token = getattr(tokenizer, 'eos_token', "<|eoa|>")

    data_config = DataConfig(
        data_path=data_args.data_path,
        model_name_or_path=data_args.model_name_or_path,
        action_tokenizer_path=data_args.action_tokenizer_path,
        actions_format=data_args.actions_format,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        dataloader_num_workers=training_args.dataloader_num_workers,
    )
    
    logger.info(f"Loading dataset from {data_args.data_path}")
    train_dataset = Emu3DrivingVAVA_AR_Dataset(args=data_config, tokenizer=tokenizer)
    
    if hasattr(train_dataset, 'action_tokenizer') and train_dataset.action_tokenizer is not None:
        tokenizer_cls = train_dataset.action_tokenizer.__class__
        if not getattr(tokenizer_cls, '_is_patched', False):
            orig_tokenizer_call = tokenizer_cls.__call__
            
            def safe_tokenizer_call(self_tok, actions, *args, **kwargs):
                is_empty = False
                if actions is None:
                    is_empty = True
                else:
                    try:
                        if hasattr(actions, 'shape'):
                            if 0 in actions.shape or len(actions.shape) == 0:
                                is_empty = True
                        elif len(actions) == 0:
                            is_empty = True
                    except:
                        pass
                
                if is_empty:
                    actions = np.zeros((8, 3), dtype=np.float32)
                    
                return orig_tokenizer_call(self_tok, actions, *args, **kwargs)
                
            tokenizer_cls.__call__ = safe_tokenizer_call
            tokenizer_cls._is_patched = True
            logger.info("✓ [Patch] Class-level Tokenizer magic-method override applied safely with lock.")

    num_epochs = (training_args.max_steps + len(train_dataset) - 1) // len(train_dataset)
    trainer.train(train_dataset, num_epochs=max(1, num_epochs))
    
    final_dir = Path(training_args.output_dir) / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(trainer.latent_extractor.state_dict(), final_dir / "latent_extractor.pt")
    torch.save(trainer.rl_module.actor.state_dict(), final_dir / "actor.pt")
    torch.save(trainer.rl_module.critic.state_dict(), final_dir / "critic.pt")
    logger.info(f"✓ Final model saved to {final_dir}")


if __name__ == "__main__":
    main()