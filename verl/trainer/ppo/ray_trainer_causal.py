import logging
import math

import torch
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

logger = logging.getLogger(__name__)


class RayPPOTrainerCausal(RayPPOTrainer):
    """RayPPOTrainer variant with causal advantage reweighting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._causal_enabled_causal():
            logger.info(
                "Causal advantage reweighting enabled: centrality=%s, ablation=%s",
                self.config.algorithm.get("causal_centrality_measure", "causal_effect"),
                self.config.algorithm.get("causal_ablation_mode", "fusion"),
            )

    def _causal_enabled_causal(self) -> bool:
        enabled = self.config.algorithm.get("enable_causal", None)
        if enabled is None:
            enabled = self.config.algorithm.get("causal_advantage_reweight", False)
        return bool(enabled)

    def _causal_reweight_enabled_causal(self) -> bool:
        return bool(self.config.algorithm.get("causal_advantage_reweight", self._causal_enabled_causal()))

    def _compute_old_log_prob(self, batch: DataProto):
        old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        if self._causal_enabled_causal() and "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch["causal_token_entropy_causal"] = old_log_prob.batch["entropys"].clone()
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        causal_metrics = {}
        if self._causal_reweight_enabled_causal():
            causal_metrics = self._apply_causal_advantage_reweight_causal(batch)

        if "causal_token_entropy_causal" in batch.batch.keys():
            batch.batch.pop("causal_token_entropy_causal")

        actor_output = super()._update_actor(batch)
        if causal_metrics:
            actor_output.meta_info["metrics"].update(causal_metrics)
        return actor_output

    def _compute_causal_weights_causal(self, batch: DataProto) -> torch.Tensor:
        non_tensor_keys = []
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_keys.append("multi_modal_inputs")

        causal_batch = batch.select(
            batch_keys=["input_ids", "attention_mask", "response_mask", "position_ids"],
            non_tensor_batch_keys=non_tensor_keys,
            meta_info_keys=["temperature"],
        )
        batch_td = causal_batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            causal_attention_threshold=float(self.config.algorithm.get("causal_attention_threshold", 0.1)),
            causal_max_path_length=int(self.config.algorithm.get("causal_max_path_length", 10)),
            causal_temperature=float(self.config.algorithm.get("causal_temperature", 1.0)),
            causal_min_weight=float(self.config.algorithm.get("causal_min_weight", 0.0)),
            causal_max_weight=float(self.config.algorithm.get("causal_max_weight", 2.0)),
            causal_aggregation=str(self.config.algorithm.get("causal_aggregation", "mean")),
            causal_normalize=self.config.algorithm.get("causal_normalize", True),
            causal_use_gradient=bool(self.config.algorithm.get("causal_use_gradient", False)),
            causal_gradient_target=str(self.config.algorithm.get("causal_gradient_target", "attention")),
            causal_centrality_measure=str(self.config.algorithm.get("causal_centrality_measure", "causal_effect")),
            causal_pagerank_damping=float(self.config.algorithm.get("causal_pagerank_damping", 0.85)),
        )

        output = self.actor_rollout_wg.compute_attention_attribution_causal(batch_td)
        weights = tu.get(output, "causal_weights")
        if weights is None:
            raise RuntimeError("Actor worker did not return causal_weights.")
        return weights.float()

    def _align_causal_weights_to_response_causal(self, batch: DataProto, causal_weights: torch.Tensor) -> torch.Tensor:
        response_mask = batch.batch["response_mask"].bool()
        attention_mask = batch.batch["attention_mask"].bool()
        advantages = batch.batch["advantages"]
        response_lens = response_mask.sum(dim=-1)
        prompt_lens = attention_mask.sum(dim=-1) - response_lens

        if causal_weights.dim() == 2:
            aligned = torch.zeros_like(advantages, dtype=causal_weights.dtype, device=causal_weights.device)
            for batch_idx in range(advantages.shape[0]):
                response_len = int(response_lens[batch_idx].item())
                if response_len <= 0:
                    continue
                start = int(prompt_lens[batch_idx].item())
                end = min(start + response_len, causal_weights.shape[-1])
                valid = max(end - start, 0)
                if valid > 0:
                    aligned[batch_idx, :valid] = causal_weights[batch_idx, start:end]
            return aligned

        if causal_weights.dim() == 3:
            aligned = torch.zeros(
                causal_weights.shape[0],
                causal_weights.shape[1],
                advantages.shape[1],
                dtype=causal_weights.dtype,
                device=causal_weights.device,
            )
            for batch_idx in range(advantages.shape[0]):
                response_len = int(response_lens[batch_idx].item())
                if response_len <= 0:
                    continue
                start = int(prompt_lens[batch_idx].item())
                end = min(start + response_len, causal_weights.shape[-1])
                valid = max(end - start, 0)
                if valid > 0:
                    aligned[batch_idx, :, :valid] = causal_weights[batch_idx, :, start:end]
            return aligned

        raise ValueError(f"Unsupported causal_weights shape {tuple(causal_weights.shape)}")

    def _get_alpha_causal(self, global_step: int) -> float:
        if self.config.algorithm.get("dynamic_alpha_enabled", False):
            total_steps = max(int(self.config.algorithm.get("dynamic_alpha_total_steps", 1000)), 1)
            initial_alpha = float(self.config.algorithm.get("dynamic_alpha_initial", 0.5))
            min_alpha = float(self.config.algorithm.get("dynamic_alpha_min", -0.2))
            progress = min(global_step / total_steps, 1.0)
            cosine_value = (1 + math.cos(math.pi * progress)) / 2
            return initial_alpha * cosine_value + min_alpha * (1 - cosine_value)
        return float(self.config.algorithm.get("causal_alpha", 0.5))

    @staticmethod
    def _sample_random_support_causal(valid_idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        perm = torch.randperm(valid_idx.numel(), generator=generator).to(valid_idx.device)
        return valid_idx[perm[:k]]

    def _build_budget_matched_ablation_modulator_causal(
        self, batch: DataProto, fused_modulator: torch.Tensor, global_step: int
    ) -> torch.Tensor:
        mode = str(self.config.algorithm.get("causal_ablation_mode", "fusion"))
        if mode == "fusion":
            return fused_modulator

        if mode not in {"random", "entropy"}:
            logger.warning("Unknown causal_ablation_mode=%s. Falling back to fusion.", mode)
            return fused_modulator

        response_mask = batch.batch["response_mask"].bool().to(fused_modulator.device)
        raw_gain = torch.relu(fused_modulator - 1.0)
        ablated = torch.ones_like(fused_modulator)

        token_entropy = None
        if mode == "entropy":
            if "causal_token_entropy_causal" not in batch.batch.keys():
                logger.warning("Entropy ablation requested but causal_token_entropy_causal is missing. Falling back.")
                return fused_modulator
            token_entropy = batch.batch["causal_token_entropy_causal"].to(fused_modulator.device)

        base_seed = int(
            self.config.trainer.get(
                "seed",
                self.config.actor_rollout_ref.actor.get("data_loader_seed", 42),
            )
            or 42
        )

        for batch_idx in range(fused_modulator.shape[0]):
            valid_idx = torch.nonzero(response_mask[batch_idx], as_tuple=False).squeeze(-1)
            active_idx = torch.nonzero(
                (raw_gain[batch_idx] > 1e-6) & response_mask[batch_idx], as_tuple=False
            ).squeeze(-1)

            if valid_idx.numel() == 0 or active_idx.numel() == 0:
                continue

            gains = raw_gain[batch_idx, active_idx]
            k = active_idx.numel()

            if mode == "random":
                target_idx = self._sample_random_support_causal(
                    valid_idx=valid_idx,
                    k=k,
                    seed=base_seed + global_step * 100003 + batch_idx,
                )
                gain_perm = self._sample_random_support_causal(
                    valid_idx=torch.arange(k, device=valid_idx.device),
                    k=k,
                    seed=base_seed + global_step * 100019 + batch_idx,
                )
                gains = gains[gain_perm]
            else:
                entropy_values = token_entropy[batch_idx, valid_idx]
                topk = torch.topk(entropy_values, k=k, largest=True)
                target_idx = valid_idx[topk.indices]
                gains = torch.sort(gains, descending=True).values

            ablated[batch_idx, target_idx] = 1.0 + gains

        return ablated

    def _build_default_modulation_map_causal(
        self, response_weights: torch.Tensor, response_mask: torch.Tensor, modulation_mode: str, global_step: int
    ) -> torch.Tensor:
        lookback = int(self.config.algorithm.get("causal_lookback", 50))
        end_penalty = float(self.config.algorithm.get("causal_end_penalty", 0.1))
        k_sigma = float(self.config.algorithm.get("causal_k_sigma", 1.08))
        tanh_scale = float(self.config.algorithm.get("causal_tanh_scale", 3.0))
        alpha = self._get_alpha_causal(global_step)

        modulation_map = torch.ones_like(response_weights)
        for batch_idx in range(response_weights.shape[0]):
            valid_len = int(response_mask[batch_idx].sum().item())
            if valid_len < 2:
                continue

            weight_seq = response_weights[batch_idx, :valid_len]
            deltas = torch.diff(weight_seq, prepend=weight_seq[:1])
            stats_deltas = deltas[:-lookback] if valid_len > lookback + 1 else deltas

            mean_bg = stats_deltas.mean()
            std_bg = stats_deltas.std(unbiased=False) + 1e-8
            z_scores = (deltas - mean_bg) / std_bg
            zeros = torch.zeros_like(z_scores)

            if modulation_mode == "signed":
                signal = torch.where(torch.abs(z_scores) > k_sigma, z_scores, zeros)
            elif modulation_mode == "absolute":
                signal = torch.where(torch.abs(z_scores) > k_sigma, torch.abs(z_scores), zeros)
            elif modulation_mode == "positive":
                signal = torch.where(z_scores > k_sigma, z_scores, zeros)
            else:
                raise ValueError(f"Unsupported modulation_mode={modulation_mode}")

            if valid_len > lookback + 1:
                decay = torch.linspace(1.0, end_penalty, lookback, device=signal.device, dtype=signal.dtype)
                signal = signal.clone()
                signal[-lookback:] = signal[-lookback:] * decay

            final_signal = torch.tanh(signal / tanh_scale)
            multiplier = 1.0 + alpha * final_signal
            modulation_map[batch_idx, :valid_len] = multiplier

        return modulation_map

    def _build_betweenness_modulation_map_causal(
        self, response_weights: torch.Tensor, response_mask: torch.Tensor, global_step: int
    ) -> torch.Tensor:
        alpha = self._get_alpha_causal(global_step)
        smooth_window = int(self.config.algorithm.get("causal_bc_smooth_window", 3))
        trend_window = int(self.config.algorithm.get("causal_bc_trend_window", 21))
        z_threshold = float(self.config.algorithm.get("causal_bc_z_threshold", 1.0))
        max_clip = float(self.config.algorithm.get("causal_bc_max_clip", 2.0))
        min_clip = float(self.config.algorithm.get("causal_bc_min_clip", 0.2))

        modulation_map = torch.ones_like(response_weights)
        for batch_idx in range(response_weights.shape[0]):
            valid_len = int(response_mask[batch_idx].sum().item())
            if valid_len < 2:
                continue

            weight_seq = response_weights[batch_idx, :valid_len]
            device = weight_seq.device

            if valid_len >= smooth_window:
                pad_s = smooth_window // 2
                padded = F.pad(weight_seq.view(1, 1, -1), (pad_s, pad_s), mode="replicate")
                kernel = torch.ones(1, 1, smooth_window, device=device, dtype=weight_seq.dtype) / smooth_window
                smooth = F.conv1d(padded, kernel).view(-1)[:valid_len]
            else:
                smooth = weight_seq

            if valid_len >= trend_window:
                pad_t = trend_window // 2
                padded_trend = F.pad(smooth.view(1, 1, -1), (pad_t, pad_t), mode="replicate")
                kernel_trend = torch.ones(1, 1, trend_window, device=device, dtype=weight_seq.dtype) / trend_window
                trend = F.conv1d(padded_trend, kernel_trend).view(-1)[:valid_len]
            else:
                trend = torch.full_like(smooth, smooth.mean())

            detrended = smooth - trend
            mean = detrended.mean()
            std = detrended.std(unbiased=False) + 1e-8
            z_scores = (detrended - mean) / std

            sparse = torch.sign(z_scores) * F.relu(torch.abs(z_scores) - z_threshold)
            signal = torch.abs(sparse)
            modulator = torch.clamp(1.0 + alpha * signal, min=min_clip, max=max_clip)
            modulation_map[batch_idx, :valid_len] = modulator

        return modulation_map

    def _build_fusion_modulation_map_causal(
        self, batch: DataProto, response_weights: torch.Tensor, response_mask: torch.Tensor, global_step: int
    ) -> torch.Tensor:
        if response_weights.dim() != 3 or response_weights.shape[1] != 2:
            raise ValueError(
                "fusion_pr_bc expects causal weights aligned to response with shape [batch, 2, response_len]."
            )

        mod_pr = self._build_default_modulation_map_causal(
            response_weights=response_weights[:, 0, :],
            response_mask=response_mask,
            modulation_mode="absolute",
            global_step=global_step,
        )
        mod_bc = self._build_betweenness_modulation_map_causal(
            response_weights=response_weights[:, 1, :],
            response_mask=response_mask,
            global_step=global_step,
        )

        raw_gain_pr = torch.relu(mod_pr - 1.0)
        raw_gain_bc = torch.relu(mod_bc - 1.0)
        scale_pr = raw_gain_pr.max() + 1e-5
        scale_bc = raw_gain_bc.max() + 1e-5

        norm_gain_pr = raw_gain_pr / scale_pr
        norm_gain_bc = raw_gain_bc / scale_bc
        fusion_gain = torch.sqrt(norm_gain_pr**2 + norm_gain_bc**2)
        restore_scale = (scale_pr + scale_bc) / 2.0
        fused_modulator = 1.0 + fusion_gain * restore_scale

        return self._build_budget_matched_ablation_modulator_causal(
            batch=batch,
            fused_modulator=fused_modulator,
            global_step=global_step,
        )

    @staticmethod
    def _renormalize_advantage_group_causal(
        advantages: torch.Tensor, response_mask: torch.Tensor, sample_indices: list[int]
    ) -> None:
        valid_advantages = [advantages[index][response_mask[index]] for index in sample_indices if response_mask[index].any()]
        if not valid_advantages:
            return

        flat = torch.cat(valid_advantages).float()
        rms = torch.sqrt(torch.mean(flat.square()) + 1e-8)
        for index in sample_indices:
            mask = response_mask[index]
            if mask.any():
                advantages[index, mask] = advantages[index, mask] / rms.to(advantages.dtype)

    def _renormalize_advantages_causal(self, batch: DataProto) -> None:
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"].bool()

        if "uid" not in batch.non_tensor_batch:
            self._renormalize_advantage_group_causal(
                advantages=advantages,
                response_mask=response_mask,
                sample_indices=list(range(advantages.shape[0])),
            )
            return

        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(batch.non_tensor_batch["uid"]):
            groups.setdefault(str(uid), []).append(index)

        for sample_indices in groups.values():
            self._renormalize_advantage_group_causal(
                advantages=advantages,
                response_mask=response_mask,
                sample_indices=sample_indices,
            )

    @staticmethod
    def _valid_stats_causal(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
        valid_values = tensor[mask]
        if valid_values.numel() == 0:
            return 0.0, 0.0, 0.0
        return (
            float(valid_values.mean().item()),
            float(valid_values.min().item()),
            float(valid_values.max().item()),
        )

    def _apply_causal_advantage_reweight_causal(self, batch: DataProto) -> dict[str, float]:
        causal_weights = self._compute_causal_weights_causal(batch).to(batch.batch["advantages"].device)
        response_weights = self._align_causal_weights_to_response_causal(batch, causal_weights)

        response_mask = batch.batch["response_mask"].bool().to(response_weights.device)
        centrality_measure = str(self.config.algorithm.get("causal_centrality_measure", "causal_effect"))

        if centrality_measure == "fusion_pr_bc":
            modulation_map = self._build_fusion_modulation_map_causal(
                batch=batch,
                response_weights=response_weights,
                response_mask=response_mask,
                global_step=self.global_steps,
            )
        elif centrality_measure == "betweenness":
            modulation_map = self._build_betweenness_modulation_map_causal(
                response_weights=response_weights,
                response_mask=response_mask,
                global_step=self.global_steps,
            )
        elif centrality_measure in {"pagerank", "causal_effect"}:
            modulation_map = self._build_default_modulation_map_causal(
                response_weights=response_weights,
                response_mask=response_mask,
                modulation_mode="absolute",
                global_step=self.global_steps,
            )
        else:
            logger.warning("Unknown causal centrality measure %s. Skipping reweighting.", centrality_measure)
            return {}

        batch.batch["advantages"] = batch.batch["advantages"] * modulation_map

        renorm_enabled = bool(
            self.config.algorithm.get(
                "causal_renormalize",
                not self.config.algorithm.get("no_advantage_std_norm", False),
            )
        )
        if renorm_enabled:
            self._renormalize_advantages_causal(batch)

        metrics = {
            "causal/alpha": float(self._get_alpha_causal(self.global_steps)),
            "causal/renormalized": float(renorm_enabled),
        }
        mod_mean, mod_min, mod_max = self._valid_stats_causal(modulation_map, response_mask)
        metrics["causal/modulator_mean"] = mod_mean
        metrics["causal/modulator_min"] = mod_min
        metrics["causal/modulator_max"] = mod_max

        if response_weights.dim() == 2:
            weight_mean, weight_min, weight_max = self._valid_stats_causal(response_weights, response_mask)
            metrics["causal/weight_mean"] = weight_mean
            metrics["causal/weight_min"] = weight_min
            metrics["causal/weight_max"] = weight_max
        else:
            weight_mean_pr, _, _ = self._valid_stats_causal(response_weights[:, 0, :], response_mask)
            weight_mean_bc, _, _ = self._valid_stats_causal(response_weights[:, 1, :], response_mask)
            metrics["causal/weight_mean_pr"] = weight_mean_pr
            metrics["causal/weight_mean_bc"] = weight_mean_bc

        return metrics
