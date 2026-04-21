import time
import logging

import torch

logger = logging.getLogger(__name__)


class MimoLayerProfiler:
    """
    Register forward/backward timing hook for each layer in MimoModel
    (modality submodules + language model).
    """

    def __init__(self, timers, enabled=False):
        self.timers = timers
        self.enabled = enabled
        self._hooks = []
        self._registered_keys = set()  # Maintain all registered keys
        self._records = {}  # Store elapsed time histories for min/max

    # ================================================================
    # Main entry: register on MimoModel
    # ================================================================
    def register_on_mimo_model(self, model):
        """
        Traverse all key layers of MimoModel and register hooks.

        For MimoModel, the structure is:
            model.modality_submodules[name]   -> modality submodules
            model.language_model.embedding
            model.language_model.decoder.layers[i]
            model.language_model.decoder.final_layernorm
            model.language_model.output_layer
        """

        # ----------------------------------------------------------
        # A. Modality Submodules
        # ----------------------------------------------------------
        for modality_name, submodule in model.modality_submodules.items():
            self._register(submodule, f"modality/{modality_name}")
            self._register_children_recursive(
                submodule, f"modality/{modality_name}"
            )

        # ----------------------------------------------------------
        # B. Language Model
        # ----------------------------------------------------------
        lm = model.language_model

        # B1. Embedding
        if hasattr(lm, 'embedding') and lm.embedding is not None:
            self._register(lm.embedding, "lm/embedding")

        # B2. Decoder layers
        if hasattr(lm, 'decoder') and hasattr(lm.decoder, 'layers'):
            for layer in lm.decoder.layers:
                layer_num = getattr(layer, 'layer_number', '?')
                layer_key = f"lm/decoder/layer_{layer_num}"
                self._register(layer, layer_key)
                self._register_transformer_layer_internals(layer, layer_key)

        # B3. Final LayerNorm
        if hasattr(lm, 'decoder') and hasattr(lm.decoder, 'final_layernorm'):
            if lm.decoder.final_layernorm is not None:
                self._register(lm.decoder.final_layernorm, "lm/decoder/final_layernorm")

        # B4. Output layer
        if hasattr(lm, 'output_layer') and lm.output_layer is not None:
            self._register(lm.output_layer, "lm/output_layer")

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(
            f"[Rank {rank}] MimoLayerProfiler: registered {len(self._registered_keys)} keys, "
            f"{len(self._hooks)} hooks total"
        )

    # ================================================================
    # Register internal subcomponents of TransformerLayer
    # ================================================================
    def _register_transformer_layer_internals(self, layer, parent_key):
        """Register attention / mlp / layernorm inside a single TransformerLayer"""
        internals = [
            'input_layernorm',
            'self_attention',
            'pre_cross_attn_layernorm',
            'cross_attention',
            'pre_mlp_layernorm',
            'mlp',
        ]
        for attr_name in internals:
            child = getattr(layer, attr_name, None)
            if child is not None:
                self._register(child, f"{parent_key}/{attr_name}")

    # ================================================================
    # Recursively find layers inside modality submodule
    # ================================================================
    def _register_children_recursive(self, module, parent_key):
        """
        Recursively traverse inside modality submodule, register hooks for all submodules containing 'layer_number' attribute
        or class name containing 'Layer'/'Block'.
        """
        for name, child in module.named_children():
            child_key = f"{parent_key}/{name}"

            if hasattr(child, 'layers') and isinstance(child.layers, torch.nn.ModuleList):
                self._register(child, child_key)
                for sub_layer in child.layers:
                    ln = getattr(sub_layer, 'layer_number', None)
                    sub_key = f"{child_key}/layer_{ln}" if ln is not None else f"{child_key}/{type(sub_layer).__name__}"
                    self._register(sub_layer, sub_key)
                    self._register_transformer_layer_internals(sub_layer, sub_key)
            elif hasattr(child, 'layer_number'):
                ln = child.layer_number
                self._register(child, f"{parent_key}/layer_{ln}")
                self._register_transformer_layer_internals(child, f"{parent_key}/layer_{ln}")
            else:
                if sum(1 for _ in child.parameters()) > 0:
                    self._register(child, child_key)
                self._register_children_recursive(child, child_key)

    # ================================================================
    # Core hook registration
    # ================================================================
    def _register(self, module, key):
        fwd_key = f"profile/{key}/fwd"
        bwd_key = f"profile/{key}/bwd"

        h1 = module.register_forward_pre_hook(self._make_fwd_pre(fwd_key))
        h2 = module.register_forward_hook(self._make_fwd_post(fwd_key))

        if hasattr(module, 'register_full_backward_pre_hook'):
            h3 = module.register_full_backward_pre_hook(self._make_bwd_pre(bwd_key))
            h4 = module.register_full_backward_hook(self._make_bwd_post(bwd_key))
            self._hooks.extend([h1, h2, h3, h4])
        else:
            self._hooks.extend([h1, h2])

        # Maintain all registered keys using a set
        self._registered_keys.add(fwd_key)
        self._registered_keys.add(bwd_key)

    def _make_fwd_pre(self, key):
        profiler = self
        def hook(module, args):
            if profiler.enabled:
                torch.cuda.synchronize()
                profiler.timers(key, log_level=2).start()
        return hook

    def _make_fwd_post(self, key):
        profiler = self
        def hook(module, args, output):
            if profiler.enabled:
                torch.cuda.synchronize()
                timer = profiler.timers(key, log_level=2)
                prev = getattr(timer, '_elapsed', getattr(timer, 'elapsed_', 0.0))
                timer.stop()
                curr = getattr(timer, '_elapsed', getattr(timer, 'elapsed_', 0.0))
                profiler._records.setdefault(key, []).append(curr - prev)
        return hook

    def _make_bwd_pre(self, key):
        profiler = self
        def hook(module, grad_output):
            if profiler.enabled:
                torch.cuda.synchronize()
                profiler.timers(key, log_level=2).start()
        return hook

    def _make_bwd_post(self, key):
        profiler = self
        def hook(module, grad_input, grad_output):
            if profiler.enabled:
                torch.cuda.synchronize()
                timer = profiler.timers(key, log_level=2)
                prev = getattr(timer, '_elapsed', getattr(timer, 'elapsed_', 0.0))
                timer.stop()
                curr = getattr(timer, '_elapsed', getattr(timer, 'elapsed_', 0.0))
                profiler._records.setdefault(key, []).append(curr - prev)
        return hook

    # ================================================================
    # Control interface
    # ================================================================
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_timers(self):
        """Reset timer status for all registered keys"""
        for key in self._registered_keys:
            if key in self.timers._timers:
                self.timers._timers[key].elapsed_ = 0.0
                self.timers._timers[key].started_ = False
                if hasattr(self.timers._timers[key], '_elapsed'):
                    self.timers._timers[key]._elapsed = 0.0
                if hasattr(self.timers._timers[key], '_started'):
                    self.timers._timers[key]._started = False
        self._records.clear()

    def log_results(self, normalizer=1):
        """Use the built-in log mechanism of megatron timers to output"""
        active_keys = [k for k in self._registered_keys if k in self.timers._timers]
        if active_keys:
            self.timers.log(active_keys, normalizer=normalizer, barrier=False)

    def get_sorted_keys(self):
        """Return a list of keys sorted by name for ordered printing"""
        return sorted(self._registered_keys)

    def print_results(self, num_iters=1):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return

        print(f"\n{'='*80}")
        print(f"  [Rank {rank}] MimoModel Layer Profiling  ({num_iters} iterations)")
        print(f"{'='*80}")

        current_section = ""
        for key in self.get_sorted_keys():
            if key not in self.timers._timers:
                continue
            timer = self.timers._timers[key]
            elapsed = getattr(timer, 'elapsed_', 0.0) or getattr(timer, '_elapsed', 0.0)
            if elapsed <= 0:
                continue

            avg_ms = (elapsed / num_iters) * 1000.0

            records = self._records.get(key, [])
            if records:
                min_ms = min(records) * 1000.0
                max_ms = max(records) * 1000.0
            else:
                min_ms = avg_ms
                max_ms = avg_ms

            # Parse key: "profile/lm/decoder/layer_1/self_attention/fwd"
            parts = key.split('/')
            direction = parts[-1].upper()
            module_path = '/'.join(parts[1:-1])

            section = parts[1] if len(parts) > 1 else ""
            if section != current_section:
                current_section = section
                print(f"\n  ── [Rank {rank}] {section} ──")

            short = '/'.join(parts[2:-1]) if len(parts) > 3 else module_path
            print(f"    [Rank {rank}] {short:50s} [{direction}] Avg: {avg_ms:10.3f} ms | Min: {min_ms:10.3f} ms | Max: {max_ms:10.3f} ms")

        print(f"\n{'='*80}\n")