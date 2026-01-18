import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 512, output_dim: int = 4096, agent_num: int = 2):
        super().__init__()
        if input_dim % agent_num != 0:
            raise ValueError("input_dim must be divisible by agent_num")
        if output_dim % agent_num != 0:
            raise ValueError("output_dim must be divisible by agent_num")
        self.agent_num = agent_num
        self.per_agent_dim = input_dim // agent_num
        self.per_agent_out_dim = output_dim // agent_num
        self.net = nn.Sequential(
            nn.Linear(self.per_agent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.per_agent_out_dim),
            nn.LayerNorm(self.per_agent_out_dim)
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: Tensor of shape [B, F, D]
        Returns:
            action_embeds: Tensor of shape [B, F, output_dim]
        """
        if actions.shape[-1] != self.per_agent_dim * self.agent_num:
            raise ValueError(f"Expected action dim {self.per_agent_dim * self.agent_num}, got {actions.shape[-1]}")

        outputs = []
        for agent_idx in range(self.agent_num):
            start = agent_idx * self.per_agent_dim
            end = (agent_idx + 1) * self.per_agent_dim
            action_slice = actions[:, :, start:end]
            outputs.append(self.net(action_slice))
        return torch.cat(outputs, dim=-1)
