# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scGPT cell-type classifier fine-tuning loop.

Loads the pretrained scGPT ``TransformerModel`` from a checkpoint triple
(``model.pt`` / ``args.json`` / ``vocab.json``), attaches/uses its cell-type
classification head, and fine-tunes on the value-binned tokens produced by
``backend.tokenize``. The train/valid/test split follows the SHARED
``cfg.training.split_sizes`` so Geneformer-vs-scGPT numbers are comparable, and the
returned metrics use the standardized keys from ``backends.base``.

Everything here is GPU + scGPT-stack dependent; it is imported lazily by the backend
and exercised by the on-SageMaker smoke run, not the offline unit tests. The loop is
deliberately framed in scGPT's own idioms (its ``TransformerModel`` forward returns a
``cls_output`` for classification when ``CLS=True``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scfm_ft.backends.base import METRIC_ACCURACY, METRIC_CONF_MATRIX, METRIC_MACRO_F1

if TYPE_CHECKING:
    from scfm_ft.config import Config


def _stratified_split(labels, split_sizes: dict, seed: int):
    """Index arrays for train/valid/test with class-stratified sampling."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    test_frac = split_sizes.get("test", 0.0)
    valid_frac = split_sizes.get("valid", 0.0)

    if test_frac > 0:
        rest, test = train_test_split(
            idx, test_size=test_frac, random_state=seed, stratify=labels
        )
    else:
        rest, test = idx, np.array([], dtype=int)

    if valid_frac > 0:
        # valid fraction is relative to the whole dataset; rescale to the rest.
        rel = valid_frac / (1.0 - test_frac) if test_frac < 1.0 else 0.0
        train, valid = train_test_split(
            rest, test_size=rel, random_state=seed, stratify=labels[rest]
        )
    else:
        train, valid = rest, np.array([], dtype=int)
    return train, valid, test


def _fold_indices(labels, num_crossval_splits: int, split_sizes: dict, seed: int, groups=None):
    """List of ``(train_idx, test_idx)`` pairs, one per fold.

    - ``1``: the single stratified train/valid/test split (unchanged, so earlier
      single-split results stay reproducible). The valid slice is unused by the loop.
    - ``5``: stratified 5-fold CV over ALL cells. Every cell is held out exactly once,
      and each fold trains on the other four fifths.
    - ``0``: train on everything, no held-out evaluation.

    With ``groups`` (one id per cell, usually the donor), splits and folds keep every
    group on one side: ``1`` holds out whole test groups (scfm_ft.splits.group_split),
    and ``5`` uses stratified group k-fold.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    idx = np.arange(len(labels))
    if num_crossval_splits == 0:
        return [(idx, np.array([], dtype=int))]
    if groups is not None:
        from scfm_ft.splits import group_folds, group_split, indices_for

        if num_crossval_splits == 1:
            split = group_split(groups, split_sizes, seed)
            # Same as the cell-level path: the loop evaluates on test; valid is unused.
            return [(indices_for(groups, split.train + split.valid), indices_for(groups, split.test))]
        return group_folds(groups, labels, num_crossval_splits, seed)
    if num_crossval_splits == 1:
        train, _valid, test = _stratified_split(labels, split_sizes, seed)
        return [(train, test)]
    skf = StratifiedKFold(n_splits=num_crossval_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(idx, labels)]


def _write_privacy_audit(parts: list, cfg: Config, output_dir: Path) -> dict:
    """Write ``{name}_privacy_audit.json`` (one audit per fold) and return summary metrics."""
    import json

    import numpy as np

    from scfm_ft.privacy import audit, print_release_check, summary_lines, write_report

    seed = int(cfg.training.seed)
    reports = [audit(p["member"], p["nonmember"], seed=seed + k) for k, p in enumerate(parts)]
    path = Path(output_dir) / f"{cfg.name}_privacy_audit.json"
    if len(reports) == 1:
        write_report(reports[0], path)
    else:
        for k, rep in enumerate(reports):
            for line in summary_lines(rep):
                print(f"[privacy-audit] fold {k + 1}/{len(reports)} {line}", flush=True)
        path.write_text(json.dumps({"folds": reports}, indent=2))
        print_release_check({"folds": reports})
    aucs = [r["cells"]["auc"] for r in reports]
    return {"privacy_audit_auc": float(np.mean(aucs))}


def _record_group_folds(groups, labels, folds, cfg: Config, output_dir: Path) -> None:
    """Write ``{name}_group_split.json``: which groups each fold trained and tested on."""
    import json

    import numpy as np

    from scfm_ft.splits import split_record, write_split_record

    groups = np.asarray(groups)
    column = cfg.classifier.group_column
    records = []
    for train_idx, test_idx in folds:
        assignment = {
            "train": sorted(set(groups[train_idx].tolist())),
            "test": sorted(set(groups[test_idx].tolist())),
        }
        records.append(split_record(groups, labels, assignment, column))
    path = Path(output_dir) / f"{cfg.name}_group_split.json"
    if len(records) == 1:
        write_split_record(records[0], path)
    else:
        path.write_text(json.dumps({"group_column": column, "folds": records}, indent=2))
        for rec in records:
            for name, missing in (rec.get("labels_missing_from_train") or {}).items():
                print(f"[split] WARNING: a fold's {name} split has labels never in its train: "
                      f"{', '.join(missing[:10])}", flush=True)


def _resolve_use_flash_attn(requested: bool) -> bool:
    """Effective FlashAttention flag.

    scGPT's fast-transformer path needs a CUDA device AND a loadable
    ``flash_attn.flash_attention.FlashMHA`` (the exact import scGPT's ``model.py``
    makes; importing just the top-level ``flash_attn`` package succeeds even when the
    compiled CUDA extension is broken). If either is missing, fall back to vanilla
    attention and log the real reason, since scGPT itself reports any failure as
    "flash_attn is not installed". Returns the flag actually passed to
    ``TransformerModel(use_fast_transformer=...)``.
    """
    if not requested:
        return False
    import torch

    if not torch.cuda.is_available():
        print("[scgpt] use_flash_attn requested but no CUDA device — using vanilla attention.")
        return False
    try:
        from flash_attn.flash_attention import FlashMHA  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — a broken CUDA build raises more than ImportError
        print(
            "[scgpt] use_flash_attn requested but flash_attn.flash_attention is not "
            f"loadable ({type(exc).__name__}: {exc}) — using vanilla attention."
        )
        return False
    # scGPT decides at its own import time; never report success if it disagrees.
    try:
        from scgpt.model import model as scgpt_model
    except ImportError:
        scgpt_model = None
    if scgpt_model is not None and not getattr(scgpt_model, "flash_attn_available", True):
        print("[scgpt] flash-attn loads here but scGPT could not import it — "
              "using vanilla attention.")
        return False
    print("[scgpt] FlashAttention enabled (fp16 mixed precision on this path).")
    return True


def _load_pretrained_weights(model, state: dict, use_fast: bool) -> dict:
    """Load a pretrained scGPT ``state`` into ``model``; fail if the encoder is incomplete.

    The published scGPT checkpoints were trained with FlashAttention, so each encoder
    layer stores its attention input projection as ``self_attn.Wqkv.*``. Standard
    PyTorch attention names the same tensor ``self_attn.in_proj_*``. Without a rename,
    a ``strict=False`` load silently skips those weights and every layer's attention
    starts from random values. This applies the same rename as scGPT's own
    ``scgpt.utils.load_pretrained``.

    Only tensors whose name AND shape match the model are loaded; the fresh
    classification head is expected to be absent. Raises ``RuntimeError`` if any
    ``transformer_encoder`` parameter was not loaded. Returns a summary dict.
    """
    if not use_fast:
        state = {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}

    model_state = model.state_dict()
    matched = {
        k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(matched, strict=False)

    missing = sorted(k for k in model_state if k not in matched)
    encoder_missing = [k for k in missing if k.startswith("transformer_encoder.")]
    if encoder_missing:
        shown = ", ".join(encoder_missing[:6])
        raise RuntimeError(
            f"{len(encoder_missing)} pretrained transformer_encoder parameters did not "
            f"load (e.g. {shown}). The checkpoint does not match this model's attention "
            f"implementation (use_fast_transformer={use_fast})."
        )
    unused = sorted(k for k in state if k not in matched)
    print(
        f"[scgpt] loaded {len(matched)}/{len(model_state)} pretrained tensors; "
        f"{len(missing)} left at init ({_summarize_prefixes(missing)}); "
        f"{len(unused)} checkpoint tensors unused."
    )
    return {"loaded": len(matched), "missing": missing, "unused": unused}


def _summarize_prefixes(keys: list[str]) -> str:
    """Compact 'cls_decoder x6, mvc_decoder x4' style summary of parameter names."""
    from collections import Counter

    if not keys:
        return "none"
    counts = Counter(k.split(".")[0] for k in keys)
    return ", ".join(f"{p} x{n}" for p, n in sorted(counts.items()))


def _class_weights(y_train, n_classes: int, mode: str):
    """Per-class loss weights from the TRAINING labels, or ``None`` for ``mode="none"``.

    ``balanced`` uses scikit-learn's convention, ``n / (k * count_c)``, so that
    ``sum_c weight_c * count_c == n`` (the loss keeps the same overall scale as an
    unweighted one). ``sqrt`` takes the square root of those weights and rescales to the
    same total. A class with no training cells gets weight 0 (it cannot be learned from
    this fold anyway, and stratified splits avoid it).
    """
    import numpy as np

    if mode == "none":
        return None
    if mode not in ("balanced", "sqrt"):
        raise ValueError(f"unknown class_weighting {mode!r}")
    counts = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    present = counts > 0
    n, k = counts.sum(), int(present.sum())
    weights = np.zeros(n_classes, dtype=np.float64)
    weights[present] = n / (k * counts[present])
    if mode == "sqrt":
        weights[present] = np.sqrt(weights[present])
        weights *= n / float((weights * counts).sum())
    return weights


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) so a run's training is repeatable.

    Covers the fresh classification head's initialisation, dropout and batch order.
    GPU kernels (FlashAttention, cuBLAS) can still differ slightly between runs, so
    results are close to, not bit-for-bit, identical.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp(enabled: bool):
    """fp16 CUDA autocast when ``enabled`` (the FlashAttention path), else a no-op."""
    import contextlib

    import torch

    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _enable_grad_checkpointing(model) -> bool:
    """Turn on activation (gradient) checkpointing on the scGPT transformer if asked.

    scGPT's ``TransformerModel`` wraps a ``nn.TransformerEncoder`` whose layers can be
    recomputed in the backward pass. Prefer a native ``gradient_checkpointing_enable``
    if the model exposes one; otherwise flip the ``TransformerEncoder``'s per-layer
    checkpoint flag where available. Returns True if checkpointing was engaged.
    """
    # 1) HF-style hook, if scGPT ever grows one.
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(fn):
        fn()
        return True
    # 2) torch's TransformerEncoder supports per-call use_reentrant checkpointing via
    #    its `enable_nested_tensor`/layer flags only indirectly; the portable lever is
    #    to wrap each encoder layer's forward in torch.utils.checkpoint. scGPT exposes
    #    the stack as `model.transformer_encoder` (a nn.TransformerEncoder).
    import torch.utils.checkpoint as ckpt
    from torch import nn

    encoder = getattr(model, "transformer_encoder", None)
    if isinstance(encoder, nn.TransformerEncoder):
        for layer in encoder.layers:
            _wrap_layer_with_checkpoint(layer, ckpt)
        return True
    return False


def _wrap_layer_with_checkpoint(layer, ckpt) -> None:
    """Replace a module's forward with a checkpointed version (non-reentrant)."""
    if getattr(layer, "_ckpt_wrapped", False):
        return
    orig_forward = layer.forward

    def _checkpointed_forward(*args, **kwargs):
        # use_reentrant=False is the modern, safer checkpoint path (works with
        # kwargs and no-grad inputs); recomputes activations in the backward pass.
        return ckpt.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

    layer.forward = _checkpointed_forward
    layer._ckpt_wrapped = True


def train_scgpt_classifier(
    checkpoint_dir: Path,
    gene_ids,
    values,
    labels,
    pad_token_id: int,
    cfg: Config,
    output_dir: Path,
    groups=None,
):
    """Fine-tune scGPT's cell-type classifier.

    Returns ``(metrics, conf_matrix, label_map)`` where ``metrics`` carries the
    standardized ``accuracy`` + ``macro_f1`` keys, ``conf_matrix`` is a DataFrame (or
    ``None`` when there is no held-out test split), and ``label_map`` maps integer id
    → class name.
    """
    import json

    import numpy as np
    import torch
    from scgpt.model import TransformerModel
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode string labels -> contiguous ids.
    classes = sorted(set(labels.tolist()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    label_map = {i: c for c, i in class_to_id.items()}
    y = np.array([class_to_id[lbl] for lbl in labels], dtype=np.int64)
    n_classes = len(classes)

    # Model hyperparameters come from the pretrained args.json.
    args_file = checkpoint_dir / "args.json"
    model_args = json.loads(args_file.read_text()) if args_file.exists() else {}
    embsize = model_args.get("embsize", 512)
    nhead = model_args.get("nheads", 8)
    d_hid = model_args.get("d_hid", 512)
    nlayers = model_args.get("nlayers", 12)
    n_input_bins = cfg.scgpt.n_input_bins

    # FlashAttention: resolve to the flag actually usable here (needs CUDA + flash-attn),
    # falling back to vanilla attention with a log line when unavailable.
    use_fast = _resolve_use_flash_attn(cfg.scgpt.use_flash_attn)

    # scGPT's TransformerModel derives the gene-token padding_idx from ``vocab[pad_token]``
    # (GeneEncoder, model.py:86), so it needs the actual GeneVocab object — passing
    # vocab=None crashes with 'NoneType is not subscriptable'. Load the same vocab the
    # tokenize stage used and set its default index to <pad> so OOV lookups are safe.
    from scfm_ft.backends.scgpt.backend import _load_gene_vocab

    vocab = _load_gene_vocab(checkpoint_dir)
    vocab.set_default_index(vocab["<pad>"])
    # ntoken must match the pretrained embedding table; use the loaded vocab length
    # (which equals vocab.json's for scGPT-human, whose specials are already present).
    vocab_size = len(vocab)

    import pandas as pd

    def _build_model():
        """Fresh TransformerModel with the pretrained weights loaded (one per fold)."""
        model = TransformerModel(
            ntoken=vocab_size,
            d_model=embsize,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=3,
            n_cls=n_classes,
            vocab=vocab,
            dropout=0.2,
            pad_token="<pad>",  # nosec B106 -- tokenizer padding token, not a credential
            pad_value=pad_token_id,
            do_mvc=cfg.scgpt.do_mvc,
            do_dab=False,
            use_batch_labels=False,
            input_emb_style="continuous",
            n_input_bins=n_input_bins,
            use_fast_transformer=use_fast,
        )
        # Load pretrained weights. A missing model.pt must not silently mean "train
        # from scratch", and a name mismatch must not silently skip encoder weights.
        ckpt_pt = checkpoint_dir / "model.pt"
        if not ckpt_pt.exists():
            hint = (
                " (found best_model.pt: copy or rename it to model.pt)"
                if (checkpoint_dir / "best_model.pt").exists()
                else ""
            )
            raise FileNotFoundError(f"scGPT weights not found at {ckpt_pt}{hint}.")
        # weights_only: load tensors only, never arbitrary pickled objects, so a
        # tampered checkpoint cannot run code. scGPT's model.pt is a plain state dict.
        state = torch.load(ckpt_pt, map_location=device, weights_only=True)
        _load_pretrained_weights(model, state, use_fast)
        model.to(device)

        if cfg.scgpt.freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("cls_decoder"):
                    param.requires_grad = False

        # Activation (gradient) checkpointing: recompute block activations in the
        # backward pass to cut activation memory, at some extra compute.
        if cfg.scgpt.grad_checkpointing:
            engaged = _enable_grad_checkpointing(model)
            if engaged:
                print("[scgpt] gradient checkpointing enabled.")
            else:
                print("[scgpt] gradient checkpointing requested but no checkpointable "
                      "encoder found — continuing without it.")
        return model

    def _run_fold(train_idx, test_idx):
        """Train a fresh model on ``train_idx``; return (model, preds, trues)."""
        model = _build_model()

        # Cap the gene-token sequence length to bound attention cost. scGPT keeps
        # every in-vocab gene; with vanilla attention (O(seq^2)) an uncapped sequence
        # OOMs a 16GB GPU. Select the top-variance ("highly variable") genes, computed
        # on THIS fold's TRAIN cells only so held-out cells never inform selection.
        fold_values, fold_gene_ids = values, gene_ids
        max_seq_len = int(getattr(cfg.scgpt, "max_seq_len", 0) or 0)
        if max_seq_len > 0 and values.shape[1] > max_seq_len:
            train_var = values[train_idx].var(axis=0)
            keep = np.sort(np.argsort(train_var)[::-1][:max_seq_len])  # keep gene order
            print(
                f"[scgpt] capping sequence length: {values.shape[1]} genes -> "
                f"{max_seq_len} highest-variance (train-split HVG)."
            )
            fold_values = values[:, keep]
            fold_gene_ids = gene_ids[keep]

        gene_ids_t = torch.as_tensor(fold_gene_ids, dtype=torch.long)
        n_genes = gene_ids_t.shape[0]

        def _loader(sel, shuffle):
            vals = torch.as_tensor(fold_values[sel], dtype=torch.float32)
            ys = torch.as_tensor(y[sel], dtype=torch.long)
            return DataLoader(
                TensorDataset(vals, ys),
                batch_size=cfg.training.batch_size,
                shuffle=shuffle,
            )

        train_loader = _loader(train_idx, True)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.training.learning_rate,
        )
        # Optional class weighting, from THIS fold's training labels only.
        mode = getattr(cfg.scgpt, "class_weighting", "none")
        weights = _class_weights(y[train_idx], n_classes, mode)
        if weights is None:
            criterion = nn.CrossEntropyLoss()
        else:
            shown = ", ".join(f"{label_map[i]}={w:.2f}" for i, w in enumerate(weights))
            print(f"[scgpt] class weighting ({mode}): {shown}")
            criterion = nn.CrossEntropyLoss(
                weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
            )
        # FlashAttention only accepts fp16/bf16 inputs, and scGPT's flash layer does not
        # cast, so the fast path runs under fp16 autocast with a gradient scaler (bf16
        # would need Ampere; fp16 also covers the T4). The standard-attention path is
        # unchanged and stays fp32.
        scaler = torch.cuda.amp.GradScaler(enabled=use_fast)

        for _epoch in range(cfg.training.epochs):
            model.train()
            for vals, ys in train_loader:
                vals, ys = vals.to(device), ys.to(device)
                gids = gene_ids_t.unsqueeze(0).expand(vals.shape[0], n_genes).to(device)
                optimizer.zero_grad()
                with _amp(use_fast):
                    out = model(
                        gids, vals, src_key_padding_mask=gids.eq(pad_token_id), CLS=True
                    )
                    loss = criterion(out["cls_output"], ys)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # clip the real (unscaled) gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        preds, trues = np.array([], dtype=int), np.array([], dtype=int)
        if len(test_idx) > 0:
            model.eval()
            p_list, t_list = [], []
            with torch.no_grad():
                for vals, ys in _loader(test_idx, False):
                    vals = vals.to(device)
                    gids = gene_ids_t.unsqueeze(0).expand(vals.shape[0], n_genes).to(device)
                    with _amp(use_fast):
                        out = model(
                            gids, vals, src_key_padding_mask=gids.eq(pad_token_id), CLS=True
                        )
                    p_list.append(out["cls_output"].argmax(-1).cpu().numpy())
                    t_list.append(ys.numpy())
            preds, trues = np.concatenate(p_list), np.concatenate(t_list)
        audit_part = None
        do_audit = getattr(cfg.training, "privacy_audit", False)
        if do_audit and len(test_idx) == 0:
            print("[privacy-audit] skipped: no held-out cells (num_crossval_splits 0)", flush=True)
        elif do_audit:
            # Per-cell loss for members (training cells) and non-members (held-out cells).
            from scfm_ft.privacy import cap_indices, per_cell_loss

            cap = getattr(cfg.training, "privacy_audit_max_cells", 20000)
            seed = int(cfg.training.seed)
            model.eval()
            audit_part = {}
            for side, sel in (
                ("member", cap_indices(train_idx, cap, seed)),
                ("nonmember", cap_indices(test_idx, cap, seed + 1)),
            ):
                logit_list = []
                with torch.no_grad():
                    for vals, _ys in _loader(sel, False):
                        vals = vals.to(device)
                        gids = gene_ids_t.unsqueeze(0).expand(vals.shape[0], n_genes).to(device)
                        with _amp(use_fast):
                            out = model(
                                gids, vals, src_key_padding_mask=gids.eq(pad_token_id), CLS=True
                            )
                        logit_list.append(out["cls_output"].float().cpu().numpy())
                loss, prob, correct = per_cell_loss(np.concatenate(logit_list), y[sel])
                audit_part[side] = {"loss": loss, "prob": prob, "correct": correct}
                if groups is not None:
                    audit_part[side]["groups"] = np.asarray(groups)[sel]
        return model, preds, trues, audit_part

    folds = _fold_indices(
        y, cfg.training.num_crossval_splits, cfg.training.split_sizes, cfg.training.seed, groups
    )
    if groups is not None:
        _record_group_folds(groups, labels, folds, cfg, output_dir)
    n_folds = len(folds)
    fold_acc, fold_f1 = [], []
    all_preds, all_trues = [], []
    audit_parts: list = []
    model = None
    for k, (train_idx, test_idx) in enumerate(folds):
        if n_folds > 1:
            print(f"[scgpt] fold {k + 1}/{n_folds}: train={len(train_idx)} test={len(test_idx)}")
        if model is not None:
            # Free the previous fold's model before building the next one.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # training.seed sets both the fold assignment (above) and the training
        # randomness. Seeding per fold makes each fold repeatable on its own.
        _seed_everything(int(cfg.training.seed) + k)
        model, preds, trues, audit_part = _run_fold(train_idx, test_idx)
        if audit_part is not None:
            audit_parts.append(audit_part)
        if len(trues) > 0:
            acc = float(accuracy_score(trues, preds))
            f1 = float(f1_score(trues, preds, average="macro"))
            fold_acc.append(acc)
            fold_f1.append(f1)
            all_preds.append(preds)
            all_trues.append(trues)
            if n_folds > 1:
                print(f"[scgpt] fold {k + 1}/{n_folds}: accuracy={acc:.4f} macro_f1={f1:.4f}")

    # Held-out evaluation (parity with Geneformer's validate()). For CV, accuracy and
    # macro_f1 are the MEAN over folds; per-fold values and the std are reported too,
    # and the confusion matrix pools every fold's held-out predictions.
    metrics: dict = {}
    conf_df = None
    if fold_acc:
        metrics[METRIC_ACCURACY] = float(np.mean(fold_acc))
        metrics[METRIC_MACRO_F1] = float(np.mean(fold_f1))
        if n_folds > 1:
            metrics["n_folds"] = n_folds
            metrics["accuracy_std"] = float(np.std(fold_acc))
            metrics["macro_f1_std"] = float(np.std(fold_f1))
            metrics["fold_accuracy"] = fold_acc
            metrics["fold_macro_f1"] = fold_f1
        pooled_p, pooled_t = np.concatenate(all_preds), np.concatenate(all_trues)
        cm = confusion_matrix(pooled_t, pooled_p, labels=list(range(n_classes)))
        conf_df = pd.DataFrame(cm, index=classes, columns=classes)
        metrics[METRIC_CONF_MATRIX] = conf_df

    if audit_parts:
        metrics.update(_write_privacy_audit(audit_parts, cfg, output_dir))

    # Persist the fine-tuned weights (the last fold's model under CV).
    torch.save(model.state_dict(), output_dir / f"{cfg.name}_scgpt_finetuned.pt")

    return metrics, conf_df, label_map
