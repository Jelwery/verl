import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NormalizeType = Union[bool, Tuple[bool, bool]]


class CausalGraphWeighterCausal:
    """Build per-token causal weights from token-token attention graphs."""

    def __init__(
        self,
        attention_threshold: float = 0.1,
        max_path_length: int = 10,
        temperature: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 2.0,
        aggregation_method: str = "mean",
        use_gradient_attribution: bool = False,
        normalize_weights: NormalizeType = True,
        gradient_target: str = "attention",
        centrality_measure: str = "causal_effect",
        pagerank_damping: float = 0.85,
    ):
        self.threshold = attention_threshold
        self.max_path_length = max_path_length
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.aggregation = aggregation_method
        self.use_gradient_attribution = use_gradient_attribution
        self.normalize_weights = normalize_weights
        self.gradient_target = gradient_target
        self.centrality_measure = centrality_measure
        self.pagerank_damping = pagerank_damping

        if gradient_target not in {"attention", "hidden"}:
            raise ValueError("gradient_target must be 'attention' or 'hidden'")
        if centrality_measure not in {"causal_effect", "pagerank", "betweenness", "fusion_pr_bc"}:
            raise ValueError(
                "centrality_measure must be one of "
                "{'causal_effect', 'pagerank', 'betweenness', 'fusion_pr_bc'}"
            )

    def compute_weights(
        self,
        attention_mask: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gradient_target == "attention":
            if attention is None:
                raise ValueError("attention is required when gradient_target='attention'")
            adjacency = self._compute_adjacency_matrix(attention, gradients, attention_mask)

            if self.centrality_measure == "causal_effect":
                scores = self._compute_causal_effect_centrality(adjacency, attention_mask)
            elif self.centrality_measure == "pagerank":
                scores = self._compute_pagerank_centrality(adjacency, attention_mask)
            elif self.centrality_measure == "betweenness":
                scores = self._compute_betweenness_centrality(adjacency, attention_mask)
            elif self.centrality_measure == "fusion_pr_bc":
                scores_pr = self._compute_pagerank_centrality(adjacency, attention_mask)
                scores_bc = self._compute_betweenness_centrality(adjacency, attention_mask)
                scores = torch.stack([scores_pr, scores_bc], dim=1)
            else:
                raise ValueError(f"Unknown centrality_measure={self.centrality_measure}")
        else:
            if hidden_states is None:
                raise ValueError("hidden_states is required when gradient_target='hidden'")
            scores = self._compute_hidden_attribution(hidden_states, gradients, attention_mask)
            scores = torch.abs(scores)

        if self.gradient_target == "attention" and self.centrality_measure == "fusion_pr_bc":
            if isinstance(self.normalize_weights, (tuple, list)) and len(self.normalize_weights) == 2:
                norm_pr, norm_bc = self.normalize_weights
            else:
                norm_pr = norm_bc = self.normalize_weights
            weights_pr = self._apply_post_processing(scores[:, 0, :], norm_pr, attention_mask)
            weights_bc = self._apply_post_processing(scores[:, 1, :], norm_bc, attention_mask)
            return torch.stack([weights_pr, weights_bc], dim=1)

        norm_flag = self.normalize_weights[0] if isinstance(self.normalize_weights, (tuple, list)) else self.normalize_weights
        return self._apply_post_processing(scores, norm_flag, attention_mask)

    def _apply_post_processing(
        self, scores: torch.Tensor, normalize: bool, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if normalize:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            seq_len = weights.shape[-1]
            weights = weights * seq_len
            weights = torch.clamp(weights, self.min_weight, self.max_weight)
        else:
            weights = scores

        if mask is not None:
            valid_seq_len = mask.sum(dim=-1, keepdim=True).clamp(min=1.0).to(weights.dtype)
            weights = weights * valid_seq_len
        else:
            seq_len = weights.shape[-1]
            weights = weights * seq_len

        return weights

    def _compute_hidden_attribution(self, hidden_states, gradients, mask):
        if gradients is None or not self.use_gradient_attribution:
            token_scores = torch.norm(hidden_states, dim=-1)
        else:
            token_scores = torch.sum(hidden_states * gradients, dim=-1)

        if self.aggregation == "mean":
            final_scores = token_scores.mean(dim=0)
        else:
            final_scores = token_scores.max(dim=0)[0]

        if mask is not None:
            final_scores = final_scores * mask.float()
        return final_scores

    def _compute_adjacency_matrix(self, attention, gradients, mask):
        if self.aggregation == "mean":
            avg_attn = attention.mean(dim=0)
            avg_grad = gradients.mean(dim=0) if gradients is not None else None
        else:
            avg_attn = attention.max(dim=0)[0]
            avg_grad = gradients.max(dim=0)[0] if gradients is not None else None

        avg_attn = avg_attn.mean(dim=1)
        if avg_grad is not None:
            avg_grad = avg_grad.mean(dim=1)

        enable_grad = self.use_gradient_attribution and (avg_grad is not None)
        if enable_grad:
            metric_matrix = torch.abs(avg_attn * avg_grad)
            metric_matrix = torch.nan_to_num(metric_matrix, nan=0.0)
        else:
            metric_matrix = avg_attn

        adjacency = torch.where(metric_matrix > self.threshold, avg_attn, torch.zeros_like(avg_attn))
        adjacency = adjacency.transpose(-2, -1)
        adjacency.diagonal(dim1=-2, dim2=-1).fill_(0.0)

        if mask is not None:
            first_valid_indices = mask.float().argmax(dim=-1)
            batch_indices = torch.arange(adjacency.shape[0], device=adjacency.device)
            adjacency[batch_indices, first_valid_indices, :] = 0.0
            bool_mask = mask.bool()
            adj_mask = bool_mask.unsqueeze(2) & bool_mask.unsqueeze(1)
            adjacency = adjacency * adj_mask.float()
        else:
            adjacency[:, 0, :] = 0.0

        return adjacency

    def _compute_causal_effect_centrality(self, adjacency, mask):
        solved = self._solve_matrix_inverse(adjacency)
        row_sums = solved.sum(dim=-1)
        nonzero_count = (solved > 1e-6).float().sum(dim=-1)
        scores = row_sums / (nonzero_count + 1e-8)
        if mask is not None:
            scores = scores * mask.float()
        return scores

    def _compute_pagerank_centrality(self, adjacency, mask):
        batch_size, seq_len, _ = adjacency.shape
        damping = self.pagerank_damping
        device = adjacency.device
        dtype = adjacency.dtype

        if mask is not None:
            valid_mask = mask.float()
            valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            teleport_prob = valid_mask / valid_count
        else:
            teleport_prob = torch.ones(batch_size, seq_len, device=device, dtype=dtype) / seq_len
        teleport_prob = teleport_prob.unsqueeze(-1)

        degree = adjacency.sum(dim=-1, keepdim=True)
        is_dangling = (degree < 1e-6).float()
        adjacency_norm = torch.where(degree > 1e-6, adjacency / (degree + 1e-8), torch.zeros_like(adjacency))
        _ = adjacency_norm + is_dangling * teleport_prob.transpose(-2, -1)
        rhs = teleport_prob * (1 - damping)

        try:
            eye = torch.eye(seq_len, device=device, dtype=dtype).unsqueeze(0)
            system = eye - damping * adjacency_norm.transpose(-2, -1)
            scores = torch.linalg.solve(system, rhs).squeeze(-1)
        except RuntimeError:
            walk = torch.ones(batch_size, seq_len, 1, device=device, dtype=dtype) / seq_len
            transition = adjacency_norm.transpose(-2, -1)
            for _ in range(20):
                walk = damping * torch.bmm(transition, walk) + rhs
            scores = walk.squeeze(-1)

        if mask is not None:
            scores = scores * mask.float()
        return scores

    def _compute_betweenness_centrality(self, adjacency, mask):
        solved = self._solve_matrix_inverse(adjacency)
        in_flow = solved.sum(dim=-2)
        out_flow = solved.sum(dim=-1)
        scores = torch.sqrt(in_flow * out_flow + 1e-8)
        if mask is not None:
            scores = scores * mask.float()
        return scores

    def _solve_matrix_inverse(self, adjacency):
        batch_size, seq_len, _ = adjacency.shape
        eye = torch.eye(seq_len, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        system = eye - adjacency
        try:
            return torch.linalg.solve(system, eye)
        except RuntimeError:
            solved = eye.clone()
            current = adjacency
            for _ in range(min(self.max_path_length, seq_len)):
                current = torch.bmm(current, adjacency)
                solved += current
            return solved


def _cfg_get(config, key: str, default):
    if hasattr(config, "get"):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return value.data if hasattr(value, "data") else value


def _parse_normalize_arg_causal(value) -> NormalizeType:
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
        if len(parts) == 2:
            return parts[0] == "true", parts[1] == "true"
        return parts[0] == "true"
    return value


def create_causal_graph_weighter_causal(config) -> CausalGraphWeighterCausal:
    normalize_value = _parse_normalize_arg_causal(_cfg_get(config, "causal_normalize", True))
    return CausalGraphWeighterCausal(
        attention_threshold=float(_cfg_get(config, "causal_attention_threshold", 0.1)),
        max_path_length=int(_cfg_get(config, "causal_max_path_length", 10)),
        temperature=float(_cfg_get(config, "causal_temperature", 1.0)),
        min_weight=float(_cfg_get(config, "causal_min_weight", 0.0)),
        max_weight=float(_cfg_get(config, "causal_max_weight", 2.0)),
        aggregation_method=str(_cfg_get(config, "causal_aggregation", "mean")),
        normalize_weights=normalize_value,
        use_gradient_attribution=bool(_cfg_get(config, "causal_use_gradient", False)),
        gradient_target=str(_cfg_get(config, "causal_gradient_target", "attention")),
        centrality_measure=str(_cfg_get(config, "causal_centrality_measure", "causal_effect")),
        pagerank_damping=float(_cfg_get(config, "causal_pagerank_damping", 0.85)),
    )


__all__ = ["CausalGraphWeighterCausal", "create_causal_graph_weighter_causal"]
