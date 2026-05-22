import logging
from contextlib import nullcontext

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.causal_graph_weighter_causal import create_causal_graph_weighter_causal
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id, get_device_name
from verl.utils.profiler import DistProfiler
from verl.workers.engine_workers import ActorRolloutRefWorker, _with_routing_replay_flag

logger = logging.getLogger(__name__)


class ActorRolloutRefWorkerCausal(ActorRolloutRefWorker):
    """Actor worker with an extra RPC for causal attention attribution."""

    @staticmethod
    def _get_causal_weighter_causal(data: TensorDict):
        return create_causal_graph_weighter_causal(data)

    @staticmethod
    def _aggregate_attentions_causal(attentions, aggregation: str) -> torch.Tensor:
        attention_list = list(attentions)
        if len(attention_list) == 0:
            raise RuntimeError("The model returned an empty attention list.")

        if aggregation == "mean":
            attention_tensor = attention_list[0].clone()
            for current in attention_list[1:]:
                attention_tensor += current
            attention_tensor /= len(attention_list)
            return attention_tensor.unsqueeze(0)

        if aggregation == "max":
            attention_tensor = attention_list[0].clone()
            for current in attention_list[1:]:
                attention_tensor = torch.maximum(attention_tensor, current)
            return attention_tensor.unsqueeze(0)

        return torch.stack(attention_list, dim=0)

    def _prepare_attention_forward_inputs_causal(self, data: TensorDict) -> TensorDict:
        if "temperature" not in data.keys():
            tu.assign_non_tensor(data, temperature=float(self.config.rollout.get("temperature", 1.0)))

        pad_token_id = getattr(getattr(self.actor.model_config, "hf_config", None), "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        defaults = {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "pad_mode": DatasetPadMode.NO_PADDING,
            "pad_token_id": pad_token_id,
            "compute_loss": False,
            "calculate_entropy": False,
        }
        for key, value in defaults.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: value})
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="actor_compute_attention_causal")
    @_with_routing_replay_flag(enabled=True)
    def compute_attention_attribution_causal(self, data: TensorDict) -> TensorDict:
        if self.actor is None:
            raise RuntimeError("compute_attention_attribution_causal requires an actor worker.")

        backend = self.config.actor.strategy
        if backend not in {"fsdp", "automodel"}:
            raise NotImplementedError(
                f"Causal attention attribution is implemented for fsdp/automodel actors only, got {backend}."
            )

        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        data = self._prepare_attention_forward_inputs_causal(data)

        engine = self.actor.engine
        device_name = get_device_name()
        autocast_dtype = getattr(engine, "_autocast_dtype", torch.bfloat16)
        autocast_ctx = (
            nullcontext()
            if autocast_dtype == torch.float32
            else torch.autocast(device_type=device_name, dtype=autocast_dtype)
        )

        with engine.eval_mode(disable_auto_offload=disable_auto_offload):
            adapter_ctx = (
                engine.disable_adapter()
                if no_lora_adapter and hasattr(engine, "disable_adapter")
                else nullcontext()
            )
            with adapter_ctx:
                device_batch = data.to(get_device_id())
                model_inputs, _ = engine.prepare_model_inputs(micro_batch=device_batch)
                model_inputs["output_attentions"] = True
                model_inputs["output_hidden_states"] = False
                model_inputs["return_dict"] = True
                model_inputs["use_cache"] = False

                with torch.no_grad():
                    with autocast_ctx:
                        raw_output = engine.module(**model_inputs)

                attentions = getattr(raw_output, "attentions", None)
                if attentions is None:
                    module_config = getattr(getattr(engine.module, "module", engine.module), "config", None)
                    attn_impl = getattr(module_config, "_attn_implementation", None)
                    raise RuntimeError(
                        "Causal attention forward returned no attention tensors. "
                        "If the actor uses flash attention, rerun with "
                        "actor_rollout_ref.model.override_config.attn_implementation=eager or sdpa. "
                        f"Detected attn_implementation={attn_impl}."
                    )

                weighter = self._get_causal_weighter_causal(data)
                attention_tensor = self._aggregate_attentions_causal(attentions, weighter.aggregation)
                weights = weighter.compute_weights(
                    attention_mask=model_inputs["attention_mask"],
                    attention=attention_tensor,
                )
                if weights.dim() == 1:
                    weights = weights.unsqueeze(0)

                weights = weights.float()
                invalid_mask = torch.isnan(weights) | torch.isinf(weights)
                if invalid_mask.any():
                    logger.warning("Found NaN/Inf in causal weights; invalid entries were reset to 1.")
                    weights = torch.where(invalid_mask, torch.ones_like(weights), weights)

        return tu.get_tensordict({"causal_weights": weights.detach().cpu()})
