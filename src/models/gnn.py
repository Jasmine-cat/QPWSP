"""正式实验使用的基础 GCN 与物理边门控 PIGNN。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCN(nn.Module):
    """不使用管段物理特征时的拓扑基线网络。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")
        self.layers = nn.ModuleList()
        prev_dim = int(in_dim)
        for _ in range(int(num_layers) - 1):
            self.layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            prev_dim = int(hidden_dim)
        self.out_layer = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.dropout(F.relu(layer(A_norm @ h)))
        return torch.sigmoid(self.out_layer(A_norm @ h).squeeze(-1))


class LightPhysicsGNN(nn.Module):
    """具有可学习物理边门控的节点-边消息传递网络。"""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_physics_gate: bool = True,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_physics_gate = bool(use_physics_gate)
        self.node_encoder = nn.Linear(int(in_dim), self.hidden_dim)
        self.edge_encoder = nn.Linear(int(edge_dim), self.hidden_dim)
        self.topo_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.edge_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 3 + int(edge_dim), self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.fuse_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 4, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.out_layer = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout))

        if self.use_physics_gate:
            self.edge_gate = nn.Sequential(
                nn.Linear(int(edge_dim), self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
            # 初始门控接近 1，避免训练初期过度削弱管段消息。
            nn.init.constant_(self.edge_gate[-2].bias, 2.0)
        else:
            self.edge_gate = None

    def compute_edge_gate(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """返回每条管段的消息通过强度。"""
        if self.edge_gate is None:
            return torch.ones(edge_attr.shape[0], dtype=edge_attr.dtype, device=edge_attr.device)
        return self.edge_gate(edge_attr).squeeze(-1)

    def forward(
        self,
        x_node: torch.Tensor,
        A_norm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        B_incidence: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.node_encoder(x_node))
        g = F.relu(self.edge_encoder(edge_attr))
        b_pos = torch.clamp(B_incidence, min=0.0)
        b_neg = torch.clamp(-B_incidence, min=0.0)
        deg_pos = torch.clamp(b_pos.sum(dim=1, keepdim=True), min=1.0)
        deg_neg = torch.clamp(b_neg.sum(dim=1, keepdim=True), min=1.0)
        src, dst = edge_index[0].long(), edge_index[1].long()

        for layer_idx in range(self.num_layers):
            h_top = F.relu(self.topo_layers[layer_idx](A_norm @ h))
            if g.shape[0] > 0:
                edge_input = torch.cat([h[src], h[dst], g, edge_attr], dim=1)
                g = F.relu(self.edge_layers[layer_idx](edge_input))
                g_msg = g * self.compute_edge_gate(edge_attr).unsqueeze(1)
                agg_out = (b_pos @ g_msg) / deg_pos
                agg_in = (b_neg @ g_msg) / deg_neg
            else:
                agg_out = torch.zeros_like(h_top)
                agg_in = torch.zeros_like(h_top)
            h = self.dropout(F.relu(self.fuse_layers[layer_idx](torch.cat([h, h_top, agg_out, agg_in], dim=1))))

        return torch.sigmoid(self.out_layer(h).squeeze(-1))
