# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""transformers 4.x names that Geneformer imports and transformers 5 removed.

Installed into the Geneformer package as ``geneformer/_scfm_transformers5_compat.py`` by
``geneformer_transformers5.py`` at image build time. Geneformer's two pre-collators
(``collator_for_classification.PrecollatorForGeneAndCellClassification`` and
``pretrainer.GeneformerPreCollator``) subclass ``SpecialTokensMixin`` only to carry their
``<mask>``/``<pad>`` tokens; transformers 5 folded that mixin into
``PreTrainedTokenizerBase``, whose constructor expects a real tokenizer. The same two
modules also import TensorFlow helpers that transformers 5 dropped along with TF support.

This module reproduces only the behaviour those classes rely on, ported from
transformers 4.46.3 (``tokenization_utils_base.SpecialTokensMixin`` and
``utils.generic._is_torch``): special-token storage and the ``*_token``,
``*_token_id``, ``pad_token_type_id``, ``special_tokens_map``, ``all_special_tokens`` and
``all_special_ids`` accessors. Token-to-id conversion is delegated to the subclass's own
``convert_tokens_to_ids``, exactly as the 4.46.3 mixin did; both Geneformer pre-collators
define it. The tokenizer-editing methods (``add_tokens``, ``add_special_tokens``,
``sanitize_special_tokens``) are deliberately absent: they need a real vocabulary, and
Geneformer never calls them on a pre-collator.
"""

from __future__ import annotations

from typing import ClassVar

import torch

try:  # tokenizers is a hard dependency of transformers; keep AddedToken accepted.
    from tokenizers import AddedToken
except ImportError:  # pragma: no cover
    AddedToken = str  # type: ignore[assignment,misc]

__all__ = ["SpecialTokensMixin", "_is_tensorflow", "_is_torch", "is_tf_available"]


def is_tf_available() -> bool:
    """transformers 5 has no TensorFlow backend."""
    return False


def _is_tensorflow(x) -> bool:
    return False


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


_TOKEN_ATTRS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def _token_property(name: str) -> property:
    private = f"_{name}"

    def fget(self):
        value = getattr(self, private)
        return None if value is None else str(value)

    def fset(self, value):
        if value is not None and not isinstance(value, (str, AddedToken)):
            raise ValueError(f"Cannot set a non-string value as the {name}")
        setattr(self, private, value)

    return property(fget, fset)


def _token_id_property(name: str) -> property:
    private = f"_{name}"

    def fget(self):
        if getattr(self, private) is None:
            return None
        return self.convert_tokens_to_ids(getattr(self, name))

    def fset(self, value):
        setattr(self, private, self.convert_ids_to_tokens(value) if value is not None else None)

    return property(fget, fset)


class SpecialTokensMixin:
    """Special-token holder with transformers 4.46.3 ``SpecialTokensMixin`` semantics."""

    SPECIAL_TOKENS_ATTRIBUTES: ClassVar[list[str]] = [*_TOKEN_ATTRS, "additional_special_tokens"]

    def __init__(self, verbose: bool = False, **kwargs) -> None:
        for attr in _TOKEN_ATTRS:
            setattr(self, f"_{attr}", None)
        self._pad_token_type_id = 0
        self._additional_special_tokens: list = []
        self.verbose = verbose

        for key, value in kwargs.items():
            if value is None:
                continue
            if key not in self.SPECIAL_TOKENS_ATTRIBUTES:
                continue
            if key == "additional_special_tokens":
                if not isinstance(value, (list, tuple)) or not all(
                    isinstance(t, (str, AddedToken)) for t in value
                ):
                    raise TypeError(f"additional_special_tokens must be a list of str, got {value!r}")
                self._additional_special_tokens = list(value)
            elif isinstance(value, (str, AddedToken)):
                setattr(self, key, value)
            else:
                raise TypeError(
                    f"Special token {key} has to be either str or AddedToken but got: {type(value)}"
                )

    bos_token = _token_property("bos_token")
    eos_token = _token_property("eos_token")
    unk_token = _token_property("unk_token")
    sep_token = _token_property("sep_token")
    pad_token = _token_property("pad_token")
    cls_token = _token_property("cls_token")
    mask_token = _token_property("mask_token")

    bos_token_id = _token_id_property("bos_token")
    eos_token_id = _token_id_property("eos_token")
    unk_token_id = _token_id_property("unk_token")
    sep_token_id = _token_id_property("sep_token")
    pad_token_id = _token_id_property("pad_token")
    cls_token_id = _token_id_property("cls_token")
    mask_token_id = _token_id_property("mask_token")

    @property
    def additional_special_tokens(self) -> list[str]:
        return [str(t) for t in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value) -> None:
        self._additional_special_tokens = list(value) if value is not None else []

    @property
    def additional_special_tokens_ids(self) -> list[int]:
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @property
    def pad_token_type_id(self) -> int:
        return self._pad_token_type_id

    @property
    def special_tokens_map(self) -> dict:
        out = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            value = getattr(self, attr)
            if value:
                out[attr] = value
        return out

    @property
    def all_special_tokens(self) -> list[str]:
        tokens: list[str] = []
        for value in self.special_tokens_map.values():
            for token in value if isinstance(value, (list, tuple)) else [value]:
                if token not in tokens:
                    tokens.append(token)
        return tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self.convert_tokens_to_ids(self.all_special_tokens)
