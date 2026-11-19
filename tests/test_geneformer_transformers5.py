# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The build-time patch that runs the pinned Geneformer on transformers 5.

The fixtures below are the exact upstream lines at the pinned commit
(1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5), so a test failure here means the patch no
longer matches what it targets.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "sagemaker" / "patches"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHES / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch = _load("geneformer_transformers5")
compat = _load("geneformer_transformers5_compat")

COLLATOR = '''import numpy as np
import torch
from transformers import (
    BatchEncoding,
    DataCollatorForTokenClassification,
    SpecialTokensMixin,
)
from transformers.utils import is_tf_available, is_torch_available, logging, to_py_obj
from transformers.utils.generic import _is_tensorflow, _is_torch

class PrecollatorForGeneAndCellClassification(SpecialTokensMixin):
    pass
'''

PRETRAINER = '''from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    SpecialTokensMixin,
    Trainer,
)
from transformers.file_utils import is_datasets_available, is_sagemaker_dp_enabled
from transformers.utils import is_tf_available, is_torch_available, logging, to_py_obj
from transformers.utils.generic import _is_tensorflow, _is_torch

        # Build the sampler.
        if self.args.group_by_length:
            pass
'''

CLASSIFIER_UTILS = '''    training_args = {
        "save_strategy": "epoch",
        "group_by_length": False,
        "length_column_name": "length",
    }
'''

FIXTURES = {
    "collator_for_classification.py": COLLATOR,
    "pretrainer.py": PRETRAINER,
    "classifier_utils.py": CLASSIFIER_UTILS,
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_patch_removes_every_transformers4_only_name(name):
    out = patch.patch_source(FIXTURES[name], name)
    assert patch.MARKER in out
    # Nothing transformers 5 removed is still imported from transformers.
    for removed in ("SpecialTokensMixin", "is_tf_available", "_is_tensorflow", "_is_torch"):
        assert not re.search(rf"from transformers[\w.]* import[^\n]*\b{removed}\b", out)
    from_tf_block = re.search(r"from transformers import \(([^)]*)\)", out)
    if from_tf_block:
        assert "SpecialTokensMixin" not in from_tf_block.group(1)
    assert "group_by_length" not in out.replace('"group_by_length"', "")  # only as a value


def test_patched_imports_come_from_the_compat_module():
    out = patch.patch_source(COLLATOR, "collator_for_classification.py")
    assert "from ._scfm_transformers5_compat import (" in out
    assert "from transformers.utils import is_torch_available, logging, to_py_obj\n" in out
    # Other transformers imports are untouched.
    assert "    DataCollatorForTokenClassification,\n" in out


def test_group_by_length_maps_to_the_equivalent_sampling_strategy():
    out = patch.patch_source(CLASSIFIER_UTILS, "classifier_utils.py")
    assert '"train_sampling_strategy": "random",' in out  # transformers 5 default == old False
    out = patch.patch_source(PRETRAINER, "pretrainer.py")
    assert 'if self.args.train_sampling_strategy == "group_by_length":' in out


@pytest.mark.parametrize("name", list(FIXTURES))
def test_patch_is_idempotent(name):
    once = patch.patch_source(FIXTURES[name], name)
    assert patch.patch_source(once, name) == once


def test_patch_fails_loudly_when_upstream_changes():
    changed = COLLATOR.replace("    SpecialTokensMixin,\n", "")
    with pytest.raises(ValueError, match="Upstream Geneformer changed"):
        patch.patch_source(changed, "collator_for_classification.py")


def test_cli_patches_a_package_and_leaves_nothing_half_done(tmp_path, monkeypatch, capsys):
    def run(pkg_dir):
        monkeypatch.setattr(sys, "argv", ["geneformer_transformers5.py", str(pkg_dir)])
        return patch.main()

    pkg = tmp_path / "geneformer"
    pkg.mkdir()
    for name, src in FIXTURES.items():
        (pkg / name).write_text(src)
    assert run(pkg) == 0
    assert (pkg / "_scfm_transformers5_compat.py").read_text() == (
        PATCHES / "geneformer_transformers5_compat.py"
    ).read_text()
    assert all(patch.MARKER in (pkg / n).read_text() for n in FIXTURES)
    assert run(pkg) == 0  # rerun is a no-op

    # A missing anchor in one file must leave every file untouched.
    broken = tmp_path / "broken" / "geneformer"
    broken.mkdir(parents=True)
    for name, src in FIXTURES.items():
        (broken / name).write_text(src)
    (broken / "pretrainer.py").write_text("nothing to patch\n")
    capsys.readouterr()
    assert run(broken) == 1
    assert "ERROR" in capsys.readouterr().err
    assert (broken / "collator_for_classification.py").read_text() == COLLATOR
    assert not (broken / "_scfm_transformers5_compat.py").exists()


# --- the compat module reproduces transformers 4.46.3 SpecialTokensMixin behaviour -------
class _PreCollator(compat.SpecialTokensMixin):
    """Shaped like Geneformer's pre-collators: tokens in a dict, own convert_*."""

    def __init__(self, token_dictionary):
        super().__init__(mask_token="<mask>", pad_token="<pad>")  # nosec B106 -- vocabulary tokens, not credentials
        self.token_dictionary = token_dictionary

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.token_dictionary.get(tokens)
        return [self.token_dictionary.get(t) for t in tokens]

    def convert_ids_to_tokens(self, value):
        return {v: k for k, v in self.token_dictionary.items()}.get(value)


def test_compat_mixin_stores_and_resolves_special_tokens():
    pc = _PreCollator({"<pad>": 0, "<mask>": 1, "ENSG01": 2})
    assert (pc.pad_token, pc.mask_token, pc.cls_token) == ("<pad>", "<mask>", None)
    assert (pc.pad_token_id, pc.mask_token_id, pc.cls_token_id) == (0, 1, None)
    assert pc.pad_token_type_id == 0
    assert pc.special_tokens_map == {"pad_token": "<pad>", "mask_token": "<mask>"}
    assert pc.all_special_tokens == ["<pad>", "<mask>"]
    assert pc.all_special_ids == [0, 1]
    assert pc.additional_special_tokens == [] and pc.verbose is False
    pc.pad_token_id = 2  # id setters go through convert_ids_to_tokens, as in 4.46.3
    assert pc.pad_token == pc.convert_ids_to_tokens(2)
    assert pc.pad_token_id == 2


def test_compat_mixin_rejects_non_string_tokens():
    with pytest.raises(TypeError):
        compat.SpecialTokensMixin(pad_token=5)
    m = compat.SpecialTokensMixin()
    with pytest.raises(ValueError):
        m.mask_token = 3


def test_compat_tf_helpers_match_a_torch_only_transformers():
    assert compat.is_tf_available() is False
    assert compat._is_tensorflow(torch.zeros(1)) is False
    assert compat._is_torch(torch.zeros(1)) is True
    assert compat._is_torch([0]) is False


# --- wiring -------------------------------------------------------------------------------
def test_image_and_local_setup_apply_the_patch():
    docker = (ROOT / "sagemaker" / "Dockerfile").read_text()
    assert "sagemaker/patches/geneformer_transformers5.py" in docker
    assert "sagemaker/patches/geneformer_transformers5_compat.py" in docker
    assert "RUN python3 /tmp/geneformer_transformers5.py /Geneformer/geneformer" in docker
    # The patch must run before Geneformer is installed into the venv.
    assert docker.index("geneformer_transformers5.py /Geneformer") < docker.index(
        "uv pip install --no-deps /Geneformer"
    )
    setup = (ROOT / "scripts" / "setup.sh").read_text()
    assert setup.index("geneformer_transformers5.py") < setup.index('uv pip install --no-deps "./')


def test_pyproject_requires_transformers_5():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"transformers>=5.10,<6"' in pyproject
    assert '"datasets>=5.0.1,<6"' in pyproject
    assert '"fsspec>=2026.6.0"' in pyproject
    assert "4.47" not in pyproject.split("[project.optional-dependencies]")[0]
