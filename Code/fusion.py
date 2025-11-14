import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class SummationFusion(nn.Module):
    """
    Summation-based Fusion Strategy
    Formula: h_fused = MLP(v_[CLS] + t_[CLS])

    Takes CLS tokens from both modalities, sums them, and passes through MLP
    """
    def __init__(self, hidden_size=768):
        super(SummationFusion, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, text_cls, image_cls, text_all_tokens=None, image_all_tokens=None):
        """
        Args:
            text_cls: [batch_size, hidden_size]
            image_cls: [batch_size, hidden_size]
            text_all_tokens: Not used in summation fusion
            image_all_tokens: Not used in summation fusion

        Returns:
            fused: [batch_size, hidden_size]
        """
        # Sum the CLS tokens
        summed = text_cls + image_cls

        # Pass through MLP
        fused = self.mlp(summed)

        return fused


class ConcatenationFusion(nn.Module):
    """
    Concatenation-based Fusion Strategy with Self-Attention
    Formula: H_fused = MLP(Self-Attention([H_V; H_T]))
             h_fused = Mean-Pooling(H_fused)

    Concatenates all tokens from both modalities, applies self-attention,
    then MLP, and finally mean-pools to get single representation
    """
    def __init__(self, hidden_size=768, num_heads=8):
        super(ConcatenationFusion, self).__init__()

        self.hidden_size = hidden_size

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=config.DROPOUT,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, text_cls, image_cls, text_all_tokens, image_all_tokens):
        """
        Args:
            text_cls: [batch_size, hidden_size] 
            image_cls: [batch_size, hidden_size] 
            text_all_tokens: [batch_size, text_seq_len, hidden_size]
            image_all_tokens: [batch_size, image_seq_len, hidden_size]

        Returns:
            fused: [batch_size, hidden_size]
        """

        concat_tokens = torch.cat([image_all_tokens, text_all_tokens], dim=1)

        # Apply self-attention
        attn_output, _ = self.self_attention(
            concat_tokens, concat_tokens, concat_tokens
        )

        # Residual connection + layer norm
        attn_output = self.layer_norm(concat_tokens + attn_output)

        # Apply MLP
        mlp_output = self.mlp(attn_output)

        # Mean pooling across sequence dimension
        fused = mlp_output.mean(dim=1)  # [batch_size, hidden_size]

        return fused


class CoAttentionFusion(nn.Module):
    """
    Co-Attention based Fusion Strategy
    Cross-modal attention where each modality attends to the other

    Formula:
        H_fused_V = MLP_V(Attention(Q_V, K_T, V_T))  # Vision attends to Text
        H_fused_T = MLP_T(Attention(Q_T, K_V, V_V))  # Text attends to Vision
        h_fused_V = Mean-Pool(H_fused_V)
        h_fused_T = Mean-Pool(H_fused_T)
        h_fused = MLP_final([h_fused_V; h_fused_T])
    """
    def __init__(self, hidden_size=768, num_heads=8):
        super(CoAttentionFusion, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scale = hidden_size ** -0.5

        # Query, Key, Value projections
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)

        # MLPs for each modality
        self.mlp_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        self.mlp_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # Final fusion MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, text_cls, image_cls, text_all_tokens, image_all_tokens):
        """
        Args:
            text_cls: [batch_size, hidden_size] 
            image_cls: [batch_size, hidden_size] 
            text_all_tokens: [batch_size, text_seq_len, hidden_size]
            image_all_tokens: [batch_size, image_seq_len, hidden_size]

        Returns:
            fused: [batch_size, hidden_size]
        """
        # Vision attended by Text (Vision queries attend to Text keys/values)
        Q_v = self.W_Q(image_all_tokens)  # [batch_size, image_seq_len, hidden_size]
        K_t = self.W_K(text_all_tokens)   # [batch_size, text_seq_len, hidden_size]
        V_t = self.W_V(text_all_tokens)   # [batch_size, text_seq_len, hidden_size]

        # Compute attention scores: Q_v @ K_t^T
        attn_scores_vt = torch.matmul(Q_v, K_t.transpose(-2, -1)) * self.scale
        attn_weights_vt = F.softmax(attn_scores_vt, dim=-1)

        # Apply attention to values
        H_fused_V = torch.matmul(attn_weights_vt, V_t)  # [batch_size, image_seq_len, hidden_size]
        H_fused_V = self.mlp_v(H_fused_V)

        # Text attended by Vision (Text queries attend to Vision keys/values)
        Q_t = self.W_Q(text_all_tokens)   # [batch_size, text_seq_len, hidden_size]
        K_v = self.W_K(image_all_tokens)  # [batch_size, image_seq_len, hidden_size]
        V_v = self.W_V(image_all_tokens)  # [batch_size, image_seq_len, hidden_size]

        # Compute attention scores: Q_t @ K_v^T
        attn_scores_tv = torch.matmul(Q_t, K_v.transpose(-2, -1)) * self.scale
        attn_weights_tv = F.softmax(attn_scores_tv, dim=-1)

        # Apply attention to values
        H_fused_T = torch.matmul(attn_weights_tv, V_v)  # [batch_size, text_seq_len, hidden_size]
        H_fused_T = self.mlp_t(H_fused_T)

        # Mean pool both fused representations
        h_fused_V = H_fused_V.mean(dim=1)  # [batch_size, hidden_size]
        h_fused_T = H_fused_T.mean(dim=1)  # [batch_size, hidden_size]

        # Concatenate and pass through final MLP
        fused = torch.cat([h_fused_V, h_fused_T], dim=1)  # [batch_size, hidden_size * 2]
        fused = self.final_mlp(fused)  # [batch_size, hidden_size]

        return fused


def create_fusion_module(fusion_type="concatenation", hidden_size=768):
    """
    Factory function to create fusion module

    Args:
        fusion_type: "summation", "concatenation", or "coattention"
        hidden_size: Hidden dimension size

    Returns:
        Fusion module instance
    """
    fusion_type = fusion_type.lower()

    if fusion_type == "summation":
        return SummationFusion(hidden_size)
    elif fusion_type == "concatenation":
        return ConcatenationFusion(hidden_size, config.NUM_ATTENTION_HEADS)
    elif fusion_type == "coattention":
        return CoAttentionFusion(hidden_size, config.NUM_ATTENTION_HEADS)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
