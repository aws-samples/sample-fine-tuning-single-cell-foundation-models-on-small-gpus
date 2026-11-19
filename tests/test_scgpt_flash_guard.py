# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The scGPT FlashAttention guard (no GPU or flash-attn needed: both are faked)."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from scfm_ft.backends.scgpt import _train
from scfm_ft.backends.scgpt._train import _amp, _resolve_use_flash_attn


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _install_flash(monkeypatch, *, loadable: bool):
    """Fake the flash_attn package; the submodule import fails like a broken CUDA build."""
    pkg = types.ModuleType("flash_attn")
    pkg.__path__ = []  # a package, so submodule imports are attempted
    monkeypatch.setitem(sys.modules, "flash_attn", pkg)
    if loadable:
        sub = types.ModuleType("flash_attn.flash_attention")
        sub.FlashMHA = object
        monkeypatch.setitem(sys.modules, "flash_attn.flash_attention", sub)
    else:
        # Top-level `import flash_attn` succeeds, but loading the submodule raises the
        # kind of error a CUDA extension built against a different torch produces.
        monkeypatch.delitem(sys.modules, "flash_attn.flash_attention", raising=False)

        class _BrokenExtension:
            def find_spec(self, name, path=None, target=None):  # other names: defer
                if name == "flash_attn.flash_attention":
                    raise ImportError(BROKEN_MSG)

        monkeypatch.setattr(sys, "meta_path", [_BrokenExtension(), *sys.meta_path])


BROKEN_MSG = "flash_attn_cuda.so: undefined symbol: _ZN3c104cuda9SetDeviceEi"


def _install_scgpt(monkeypatch, *, available: bool):
    scgpt = types.ModuleType("scgpt")
    model_pkg = types.ModuleType("scgpt.model")
    model_mod = types.ModuleType("scgpt.model.model")
    model_mod.flash_attn_available = available
    model_pkg.model = model_mod
    monkeypatch.setitem(sys.modules, "scgpt", scgpt)
    monkeypatch.setitem(sys.modules, "scgpt.model", model_pkg)
    monkeypatch.setitem(sys.modules, "scgpt.model.model", model_mod)


def test_not_requested_is_off():
    assert _resolve_use_flash_attn(False) is False


def test_no_cuda_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_use_flash_attn(True) is False
    assert "no CUDA device" in capsys.readouterr().out


def test_top_level_import_ok_but_extension_broken_falls_back(cuda, monkeypatch, capsys):
    """The bug this guard exists for: `import flash_attn` works, the submodule does not."""
    _install_flash(monkeypatch, loadable=False)
    _install_scgpt(monkeypatch, available=True)
    assert _resolve_use_flash_attn(True) is False
    out = capsys.readouterr().out
    assert "flash_attn.flash_attention is not loadable" in out
    assert f"ImportError: {BROKEN_MSG}" in out  # the real exception is logged, not hidden


def test_scgpt_disagreeing_falls_back(cuda, monkeypatch, capsys):
    _install_flash(monkeypatch, loadable=True)
    _install_scgpt(monkeypatch, available=False)
    assert _resolve_use_flash_attn(True) is False
    assert "scGPT could not import it" in capsys.readouterr().out


def test_all_good_enables(cuda, monkeypatch, capsys):
    _install_flash(monkeypatch, loadable=True)
    _install_scgpt(monkeypatch, available=True)
    assert _resolve_use_flash_attn(True) is True
    assert "FlashAttention enabled" in capsys.readouterr().out


def test_amp_is_noop_when_disabled():
    with _amp(False):
        x = torch.ones(2) * 2.0
    assert x.dtype == torch.float32


def test_amp_uses_fp16_cuda_autocast_when_enabled():
    ctx = _amp(True)
    assert isinstance(ctx, torch.autocast)
    assert ctx.device == "cuda"
    assert ctx.fast_dtype == torch.float16


def test_module_has_no_stale_top_level_flash_import():
    """Guard must not regress to only `import flash_attn` (which hides broken builds)."""
    import inspect

    src = inspect.getsource(_train._resolve_use_flash_attn)
    assert "from flash_attn.flash_attention import FlashMHA" in src
