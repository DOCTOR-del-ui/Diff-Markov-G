import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class StepDiscretizer(nn.Module):
    def __init__(self, num_bins=128):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, x):
        # 将输入缩放到 [0, num_bins-1]
        scaled = x * (self.num_bins - 1)
        scaled = torch.clamp(scaled, 0, self.num_bins - 1)  # ✅ 防止越界
        hard = torch.floor(scaled)
        output = (hard - scaled).detach() + scaled
        return output

class MarkovTransition(nn.Module):
    def __init__(self, num_states=128, temp=1e-4):
        super().__init__()
        self.num_states = num_states
        self.temp = temp
        self.discretizer = StepDiscretizer(self.num_states)

    def soft_one_hot(self, index):
        B, F_DIM, L = index.shape
        states = torch.arange(self.num_states, device=index.device).float().view(1, 1, 1, self.num_states)
        index_exp = index.unsqueeze(-1)
        logits = -((index_exp - states) ** 2) / self.temp
        return F.softmax(logits, dim=-1)

    def one_hot_ste(self, index):
        # ✅ clamp 防止 scatter 越界
        index = torch.clamp(index, 0, self.num_states - 1)
        soft = self.soft_one_hot(index)
        hard = torch.zeros_like(soft).scatter_(-1, index.long().unsqueeze(-1), 1.0)
        return (hard - soft).detach() + soft

    def forward(self, x):
        
        x_min = x.min(dim=1, keepdim=True).values   # (B, 1, F)
        x_max = x.max(dim=1, keepdim=True).values   # (B, 1, F)

        # 防止除零
        denom = (x_max - x_min).clamp_min(1e-8)

        # 归一化到 0~1
        x = (x - x_min) / denom                     # (B, L, F)
        
        # 连续 → 离散 → one-hot
        soft_index = self.discretizer(x)
        soft_index = soft_index.transpose(1, 2)  # (B, F, L)
        S = self.one_hot_ste(soft_index)

        # 相邻状态
        S1 = S[:, :, :-1, :]
        S2 = S[:, :, 1:, :]
        C = torch.matmul(S1.transpose(-2, -1), S2)  # (B, F, num_states, num_states)

        # ✅ 安全归一化
        denom = C.sum(dim=-1, keepdim=True)
        denom = denom.clamp_min(1e-8)  # 防止除零
        P = C / denom
        return P
    
    
class MarkovAwareEmbedding(nn.Module):
    def __init__(self, num_features, d_model, scale=1, use_maremb_weight=False):
        super().__init__()
        
        self.use_adaptive = use_maremb_weight
        
        if self.use_adaptive:
            # 初始化可学习通道级 scale 向量（d_model）
            #print("使用了嵌入层自适应")
            self.scale = nn.Parameter(torch.ones(d_model) * scale)
        else:
            self.scale = scale
        
        # Δ + prev + sign = 3 * F  →  d_model
        self.proj = nn.Linear(num_features * 3, d_model)

    def forward(self, x):
        """
        x: (B, L, F)
        """
        B, L, F = x.shape

        # ----- 1. 计算 Δ -----
        dx = x[:, 1:, :] - x[:, :-1, :]        # (B, L-1, F)
        zero = torch.zeros(B, 1, F, device=x.device)
        dx = torch.cat([zero, dx], dim=1)      # (B, L, F)

        # ----- 2. 计算 prev -----
        prev = torch.cat([zero, x[:, :-1, :]], dim=1)  # (B, L, F)

        # ----- 3. 计算 sign -----
        sign = torch.sign(dx)                  # (B, L, F)

        # ----- 4. 拼接 -----
        feats = torch.cat([dx, prev, sign], dim=-1)  # (B, L, 3F)

        # ----- 5. 映射到 d_model -----
        out = self.proj(feats)   # (B, L, d_model)
        
        #print(f"MarkovAwareEmbedding scale: {self.scale}")

        return out * self.scale



class MarkovResidualBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=None, scale=0.1, use_resid_pdrop=False):
        """
        scale >= 0：固定缩放因子（原行为）
        scale < 0 ：启用通道级自适应 scale（可学习参数）
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2

        # 判断是否使用自适应 scale
        self.use_adaptive = use_resid_pdrop

        if self.use_adaptive:
            # 初始化可学习通道级 scale 向量（d_model）
            #print("使用了残差层自适应")
            self.scale = nn.Parameter(torch.ones(d_model) * scale)
        else:
            # 原版标量 scale
            self.scale = scale

        # 把三通道 (3*d_model) → d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, h):
        """
        h: (B, L, d_model)
        """
        B, L, D = h.shape

        # Δh
        dh = h[:, 1:, :] - h[:, :-1, :]
        dh = torch.cat([torch.zeros(B, 1, D, device=h.device), dh], dim=1)

        # prev h
        prev = torch.cat([torch.zeros(B, 1, D, device=h.device), h[:, :-1, :]], dim=1)

        # sign
        sign = torch.sign(dh)

        # concat → (B, L, 3*D)
        feats = torch.cat([dh, prev, sign], dim=-1)

        out = self.proj(feats)   # (B, L, d_model)
        
        #print(f"MarkovResidualBlock scale: {self.scale}")

        return out * self.scale

    
    
class MarkovHead(nn.Module):
    def __init__(self, d_model=64, num_features=14, num_states=128, hidden=256, seq_length=96):
        super().__init__()

        self.num_features = num_features
        self.num_states = num_states

        # 把 embedding 还原到 feature 维
        self.to_feat = nn.Linear(d_model, num_features)

        # 为每个 feature 预测独立的 128×128 矩阵
        self.fc = nn.Sequential(
            nn.Linear(seq_length, hidden),        # 时间维汇聚
            nn.GELU(),
            nn.Linear(hidden, num_states * num_states)
        )

    def forward(self, x):
        # x: (B, 96, 64)

        feat = self.to_feat(x)              # (B, 96, 14)
        feat = feat.permute(0, 2, 1)        # (B, 14, 96)

        B = feat.size(0)

        # 对每个 feature (14个) 处理
        out = self.fc(feat)                 # (B, 14, 16384)
        out = out.view(B, self.num_features, self.num_states, self.num_states)
        return out
    
""" class MarkovHead(nn.Module):
    def __init__(self, d_model=64, num_features=14, num_states=128, hidden=256):
        super().__init__()

        self.num_features = num_features
        self.num_states = num_states

        # 把 embedding 还原成 feature 维度
        self.to_feat = nn.Linear(d_model, num_features)

        # 用 1D 卷积代替 MLP，从 96 时间维提取信息
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 卷积后 feature 的输出维度仍为 96，所以输出仍是 (B, 14, 64, 96)
        # 展开后用线性层映射到 128×128
        self.fc = nn.Linear(64 * 96, num_states * num_states)

    def forward(self, x):
        # x: (B, 96, 64)
        feat = self.to_feat(x)        # (B, 96, 14)
        feat = feat.permute(0, 2, 1)  # (B, 14, 96)

        B = feat.size(0)

        # 卷积要求 channel 维，因此把每个 feature 单独卷积
        # (B,14,96) → (B*14,1,96)
        feat = feat.reshape(B * self.num_features, 1, 96)

        # 卷积处理
        conv_out = self.conv(feat)   # (B*14, 64, 96)

        # 展开 & 映射到 128×128
        conv_out = conv_out.reshape(B, self.num_features, -1)   # (B,14, 64*96)
        out = self.fc(conv_out)                                # (B,14,16384)

        return out.reshape(B, self.num_features, self.num_states, self.num_states) """

""" def kl_markov_loss(P_real, P_pred, eps=1e-12):
    # 1) clamp 防止 log(0)
    P_real = P_real.clamp(min=eps)
    P_pred = P_pred.clamp(min=eps)
    
    # 2) KL(P_real || P_pred) = sum( P_real * (log(P_real) - log(P_pred)) )
    kl = (P_real * (P_real.log() - P_pred.log())).sum(dim=-1).sum(dim=-1)  # 对最后两维(128,128)求和
    
    # 3) 对 batch 和特征做平均
    return kl.mean() """

def generate_markov_sequence_torch(transition_matrix, sequence_length, initial_state=None):
    """
    transition_matrix: (batch, num_states, num_states)
    sequence_length: int
    initial_state: (batch,) or None
    return: (batch, sequence_length)  (state values between 0~1)
    """

    batch, num_states, _ = transition_matrix.shape
    device = transition_matrix.device

    # state_values: e.g. [1/num_states, 2/num_states, ..., 1]
    state_values = torch.linspace(0, 1, num_states + 1, device=device)[1:]

    # find absorbing states (row sum = 0)
    row_sums = transition_matrix.sum(dim=-1)  # (batch, num_states)
    absorbing_mask = (row_sums == 0)

    # possible initial states mask
    possible_initial_mask = (~absorbing_mask).float()  # (batch, num_states)

    # sample initial states if not provided
    if initial_state is None:
        probs = possible_initial_mask / possible_initial_mask.sum(dim=1, keepdim=True)
        initial_state = torch.multinomial(probs, 1).squeeze(1)
    else:
        initial_state = initial_state.to(device)

    # output sequence
    sequences = torch.empty(batch, sequence_length, device=device)

    current_state = initial_state  # (batch,)

    uniform_probs = torch.full((batch, num_states), 1.0 / num_states, device=device)

    for t in range(sequence_length):

        # write state value
        sequences[:, t] = state_values[current_state]

        # transition row for each sample: (batch, num_states)
        probs = transition_matrix[torch.arange(batch), current_state]

        # if a row is absorbing (all zero), fallback to uniform distribution
        zero_mask = (probs.sum(dim=1) == 0)  # (batch,)
        if zero_mask.any():
            probs = probs.clone()
            probs[zero_mask] = uniform_probs[zero_mask]

        # normalize row just in case
        probs = probs / probs.sum(dim=1, keepdim=True)

        # sample next state
        current_state = torch.multinomial(probs, 1).squeeze(1)

    return sequences
