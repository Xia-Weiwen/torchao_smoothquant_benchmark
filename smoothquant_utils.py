"""
Shared utilities for SmoothQuant benchmarks.

Provides the core quantization, compilation, and inference functions
used by both performance and accuracy benchmark scripts.
The API follows the patterns in smoothquant_example.py.
"""

import os
import torch
import torch.nn as nn

import torchao
# Must import to register fusion passes for x86 inductor quantizer
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer  # noqa: F401
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.granularity import PerTensor, PerRow
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
)
import torch._inductor.config as inductor_config

# ── Inductor defaults ─────────────────────────────────────────────────────────
inductor_config.freezing = True
inductor_config.max_autotune = True
inductor_config.cpp_wrapper = True


# ── Autocast helper ───────────────────────────────────────────────────────────

def optional_autocast(func, use_autocast):
    """Wrap *func* with torch.no_grad and optionally torch.autocast("cpu").

    Follows the decorator pattern from smoothquant_example.py but accepts
    *use_autocast* explicitly so it works as a reusable utility.
    """
    def wrapper(*args, **kw):
        if use_autocast:
            with torch.no_grad(), torch.autocast("cpu"):
                return func(*args, **kw)
        else:
            with torch.no_grad():
                return func(*args, **kw)
    return wrapper


# ── Quantization ──────────────────────────────────────────────────────────────

def _skip_small_linears(mod, fqn):
    """filter_fn: skip Linear layers whose output dimension is < 32.

    Tiny head layers (e.g. qa_outputs [2, 768]) trigger an inductor
    LoweringException with dynamic smooth-quant + torch.compile.
    """
    return isinstance(mod, nn.Linear) and min(mod.weight.shape) >= 32


def apply_smoothquant(model, quant_mode, alpha, calib_inputs, use_autocast=True,
                      filter_fn=_skip_small_linears):
    """PREPARE → calibrate → CONVERT.

    Parameters
    ----------
    model : nn.Module
    quant_mode : str  ("smooth-dynamic" or "smooth-static")
    alpha : float
    calib_inputs : tuple or list
        Either a single (input_ids, attention_mask) tuple, or a list of such
        tuples for multi-sample calibration.  Each tuple is unpacked as
        positional args: ``model(*inp)``.
    use_autocast : bool
    filter_fn : callable or None
        Passed to ``torchao.quantization.quantize_``.  The default skips tiny
        classifier / QA-head layers that cause inductor issues.
    """
    if quant_mode == "smooth-dynamic":
        base_config = Int8DynamicActivationInt8WeightConfig(
            version=2, granularity=[PerRow(), PerRow()],
        )
    elif quant_mode == "smooth-static":
        base_config = Int8StaticActivationInt8WeightConfig(
            granularity=[PerTensor(), PerRow()],
        )
    else:
        return model

    # Normalize calib_inputs: single tuple → list of one tuple
    if isinstance(calib_inputs, tuple) and len(calib_inputs) == 2 and isinstance(calib_inputs[0], torch.Tensor):
        calib_inputs = [calib_inputs]

    qcfg = SmoothQuantConfig(
        base_config=base_config, step=QuantizationStep.PREPARE, alpha=alpha,
    )
    torchao.quantization.quantize_(model, qcfg, filter_fn=filter_fn)

    _infer = optional_autocast(lambda m, inp: m(*inp), use_autocast)
    for inp in calib_inputs:
        _infer(model, inp)

    qcfg.step = QuantizationStep.CONVERT
    torchao.quantization.quantize_(model, qcfg, filter_fn=filter_fn)
    return model


# ── Compilation ───────────────────────────────────────────────────────────────

def do_compile(model, use_autocast):
    """torch.compile the model (matching smoothquant_example.py)."""
    def _compile(m):
        options = {"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe}
        return torch.compile(m, options=options, fullgraph=True)
    fn = optional_autocast(_compile, use_autocast)
    return fn(model)


def do_aoti_compile(model, model_inputs, use_autocast, save_path):
    """Export → AOTI compile → package → load (matching smoothquant_example.py)."""
    inductor_config.cpp.enable_concat_linear = False

    import torch._export.utils as eu

    def _export_and_compile(m):
        with eu._disable_aten_to_metadata_assertions():
            exported = torch.export.export(m, args=model_inputs)
        torch._inductor.aoti_compile_and_package(exported, package_path=save_path)
        return torch._inductor.aoti_load_package(save_path)

    fn = optional_autocast(_export_and_compile, use_autocast)
    return fn(model)


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model, model_inputs, use_autocast):
    """Run one forward pass with positional args: model(*model_inputs)."""
    fn = optional_autocast(lambda m, inp: m(*inp), use_autocast)
    return fn(model, model_inputs)
