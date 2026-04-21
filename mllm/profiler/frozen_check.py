import torch
from collections import defaultdict


class ParamFreezeChecker:
    """Check the parameter freezing status in MimoModel and verify if gradients actually flow after backward."""

    def __init__(self, model, check_grad_after_backward=True, verbose=False):
        self.model = model
        self.verbose = verbose
        self.check_grad_after_backward = check_grad_after_backward
        self._hooks = []
        self._grad_records = {}  # name -> {has_grad, grad_norm}

    def _analyze_requires_grad(self):
        """Group and statistics requires_grad status by module."""
        trainable = defaultdict(lambda: {"count": 0, "numel": 0, "names": []})
        frozen = defaultdict(lambda: {"count": 0, "numel": 0, "names": []})

        for name, param in self.model.named_parameters():
            # Extract module prefix: remove trailing .weight / .bias
            parts = name.rsplit(".", 1)
            prefix = parts[0] if len(parts) > 1 else name

            bucket = trainable if param.requires_grad else frozen
            bucket[prefix]["count"] += 1
            bucket[prefix]["numel"] += param.numel()
            bucket[prefix]["names"].append(name)

        return trainable, frozen

    def _register_grad_hooks(self):
        """Register hooks for each parameter with requires_grad=True to record gradient status after backward."""
        self._grad_records.clear()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                def _make_hook(param_name, p):
                    def hook(grad):
                        self._grad_records[param_name] = {
                            "has_grad": grad is not None,
                            "grad_norm": grad.norm().item() if grad is not None else 0.0,
                            "grad_has_nan": torch.isnan(grad).any().item() if grad is not None else False,
                            "grad_has_inf": torch.isinf(grad).any().item() if grad is not None else False,
                            "grad_all_zero": (grad == 0).all().item() if grad is not None else True,
                        }
                    return hook
                handle = param.register_hook(_make_hook(name, param))
                self._hooks.append(handle)

    def _print_separator(self, title):
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")

    def report_requires_grad(self):
        """Print requires_grad status report."""
        trainable, frozen = self._analyze_requires_grad()

        self._print_separator("Parameter Freeze Status: requires_grad")

        total_trainable = sum(v["numel"] for v in trainable.values())
        total_frozen = sum(v["numel"] for v in frozen.values())
        total = total_trainable + total_frozen

        print(f"  Total params:     {total:>14,}")
        print(f"  Trainable params: {total_trainable:>14,} ({100 * total_trainable / max(total, 1):.1f}%)")
        print(f"  Frozen params:    {total_frozen:>14,} ({100 * total_frozen / max(total, 1):.1f}%)")

        if frozen:
            print(f"\n  ── FROZEN (requires_grad=False) ──")
            for prefix in sorted(frozen.keys()):
                info = frozen[prefix]
                print(f"    {prefix:<70s} {info['numel']:>12,} params ({info['count']} tensors)")
                if self.verbose:
                    for n in info["names"]:
                        print(f"      - {n}")

        # Highlight: find inconsistencies in CLIP related modules
        print(f"\n  ── TRAINABLE (requires_grad=True) ──")
        for prefix in sorted(trainable.keys()):
            info = trainable[prefix]
            print(f"    {prefix:<70s} {info['numel']:>12,} params ({info['count']} tensors)")
            if self.verbose:
                for n in info["names"]:
                    print(f"      - {n}")

        # Check CLIP internally for freeze inconsistencies
        clip_trainable = {k for k in trainable if "clip" in k.lower() or "encoder" in k.lower()}
        clip_frozen = {k for k in frozen if "clip" in k.lower() or "encoder" in k.lower()}
        if clip_trainable and clip_frozen:
            print(f"\n  ⚠️  WARNING: Inconsistent freezing found in CLIP/Encoder modules!")
            print(f"      Trainable modules: {sorted(clip_trainable)}")
            print(f"      Frozen modules:    {sorted(clip_frozen)}")

    def report_grad_status(self):
        """Call after backward, print actual gradient status report."""
        self._print_separator("Gradient Status After Backward")

        if not self._grad_records:
            print("  No gradient records. Make sure backward() is called and hooks are registered.")
            return

        # Group by module
        module_grads = defaultdict(list)
        for name, info in self._grad_records.items():
            parts = name.rsplit(".", 1)
            prefix = parts[0] if len(parts) > 1 else name
            module_grads[prefix].append((name, info))

        anomalies = []
        for prefix in sorted(module_grads.keys()):
            entries = module_grads[prefix]
            all_zero = all(e[1]["grad_all_zero"] for e in entries)
            any_nan = any(e[1]["grad_has_nan"] for e in entries)
            any_inf = any(e[1]["grad_has_inf"] for e in entries)

            status = "OK"
            if all_zero:
                status = "⚠️  ALL_ZERO"
                anomalies.append((prefix, "all gradients are zero"))
            if any_nan:
                status = "❌ HAS_NAN"
                anomalies.append((prefix, "has NaN gradients"))
            if any_inf:
                status = "❌ HAS_INF"
                anomalies.append((prefix, "has Inf gradients"))

            avg_norm = sum(e[1]["grad_norm"] for e in entries) / len(entries)
            print(f"  {status:<14s} {prefix:<60s} avg_grad_norm={avg_norm:.6f}")

            if self.verbose:
                for name, info in entries:
                    print(f"               - {name}: norm={info['grad_norm']:.6f} "
                          f"zero={info['grad_all_zero']} nan={info['grad_has_nan']}")

        # Check for parameters with requires_grad=True but no gradient received
        params_with_grad_expected = {
            n for n, p in self.model.named_parameters() if p.requires_grad
        }
        params_with_grad_received = set(self._grad_records.keys())
        missing = params_with_grad_expected - params_with_grad_received
        if missing:
            print(f"\n  ⚠️  WARNING: {len(missing)} parameters have requires_grad=True but received no gradient:")
            for n in sorted(missing):
                print(f"      - {n}")
                anomalies.append((n, "requires_grad=True but no gradient received"))

        if anomalies:
            print(f"\n  {'=' * 60}")
            print(f"  Total {len(anomalies)} anomalies found:")
            for name, reason in anomalies:
                print(f"    - {name}: {reason}")
        else:
            print(f"\n  ✅ All trainable parameters have normal gradients.")

    def cleanup(self):
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._grad_records.clear()


def frozen_check_on_mimo_model(model, check_grad_after_backward=True, verbose=False):
    """
    Register parameter freeze checking on MimoModel.

    Usage:
        checker = register_on_mimo_model(model)
        # Output requires_grad status immediately
        # After executing forward + backward once:
        checker.report_grad_status()
        # Cleanup after use
        checker.cleanup()
    """
    checker = ParamFreezeChecker(
        model,
        check_grad_after_backward=check_grad_after_backward,
        verbose=verbose,
    )

    # Print requires_grad status immediately
    checker.report_requires_grad()

    # Register backward hooks for subsequent checks
    if check_grad_after_backward:
        checker._register_grad_hooks()
        print("\n  ℹ️  Backward hooks registered. Call checker.report_grad_status() "
              "after executing backward() to check gradient status.")

    return checker