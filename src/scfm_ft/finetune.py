# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 2: fine-tune Geneformer with the maintained ``geneformer.Classifier`` API.

``Classifier`` handles label preparation, train/valid/test splitting (or k-fold
cross-validation), and the HuggingFace ``Trainer`` loop. We drive it from the typed
``Config`` so every run is reproducible from a single YAML file.

Workflow:
  1. ``Classifier.prepare_data`` reads the tokenized ``.dataset``, builds the label
     id<->class mapping, filters/caps as configured, and writes a ``*_labeled.dataset``.
  2. ``Classifier.train_all_data`` (splits==0) or ``Classifier.validate`` (splits>=1)
     fine-tunes the base model and writes the fine-tuned model directory.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from scfm_ft.config import Config
from scfm_ft.tokenize import tokenize


@contextlib.contextmanager
def _kbit_prep_get_peft_model():
    """Make Geneformer's ``get_peft_model`` run ``prepare_model_for_kbit_training``
    first (Fix 1 of the NaN investigation -- the PRIMARY cause).

    Geneformer's ``perturber_utils.load_model`` applies LoRA to the 4-bit model with
    a bare ``get_peft_model(model, peft_config)`` and SKIPS peft's mandatory
    ``prepare_model_for_kbit_training`` step. That helper upcasts LayerNorm and the
    output head to fp32 (so normalization stats do not overflow under bf16 compute),
    re-enables gradient checkpointing correctly, and enables input grads. Without the
    fp32 upcast, a 4-bit BERT's LayerNorm produces inf/NaN that propagate forward --
    the observed ``loss 0.0, grad_norm nan`` from step 1.

    Geneformer imports the symbol as ``from peft import get_peft_model`` and calls it
    as module-global ``get_peft_model`` inside ``load_model``, so we patch the name in
    the ``geneformer.perturber_utils`` namespace (where the call resolves) to a wrapper
    that preps the model first, then delegates to the real ``get_peft_model``. Restored
    on exit. Only used on the QLoRA path.
    """
    import geneformer.perturber_utils as pu
    from peft import prepare_model_for_kbit_training

    original = pu.get_peft_model

    def _wrapped(model, peft_config, *args, **kwargs):  # type: ignore[no-untyped-def]
        # use_gradient_checkpointing=False: Geneformer manages checkpointing via its
        # own TrainingArguments; enabling it here can double-wrap and is unnecessary
        # for a small classifier head. The key effect we need is the fp32 upcast of
        # LayerNorm + embeddings that stops the NaN.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
        peft_model = original(model, peft_config, *args, **kwargs)
        # NaN diagnostic (opt-in via GENEFORMER_FT_NAN_TRACE=1): report post-prep
        # norm dtypes and arm forward hooks that pinpoint the first non-finite
        # module output. No effect on a normal run.
        if os.environ.get("GENEFORMER_FT_NAN_TRACE"):
            _arm_nan_trace(peft_model)
        return peft_model

    pu.get_peft_model = _wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        pu.get_peft_model = original  # type: ignore[assignment]


def _arm_nan_trace(model) -> None:  # type: ignore[no-untyped-def]
    """Diagnostic (opt-in, GENEFORMER_FT_NAN_TRACE=1): find WHERE the QLoRA NaN
    originates instead of guessing.

    Two reports, then it detaches itself:

    1. **Post-prep dtype audit** -- print the dtype of every LayerNorm and the
       classifier head AFTER prepare_model_for_kbit_training ran. The theory that
       the fp32 upcast "should" protect the norms is only true if the norms are
       actually fp32 here; this prints the ground truth.

    2. **First non-finite module** -- register a forward hook on every leaf module.
       On the first forward that produces a NaN/inf anywhere, print the EARLIEST
       module (by execution order) whose output is non-finite, with the finite
       status of its inputs -- so we see e.g. "embeddings fine -> LayerNorm_0 inf"
       (overflow in norm) vs "everything finite -> classifier logits inf" (head
       init) vs a masked-softmax -inf. Hooks remove themselves after the first
       bad forward so training/logs are not flooded.
    """
    import torch

    # --- Report 1: dtype audit after kbit prep ---
    print("[NAN-TRACE] post-prep dtype audit:", flush=True)
    norm_dtypes: dict[str, int] = {}
    head_lines: list[str] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if "LayerNorm" in cls or "RMSNorm" in cls:
            w = getattr(module, "weight", None)
            if w is not None:
                norm_dtypes[str(w.dtype)] = norm_dtypes.get(str(w.dtype), 0) + 1
        if "classifier" in name.split(".")[-1] or name.endswith("pooler.dense"):
            w = getattr(module, "weight", None)
            if w is not None:
                head_lines.append(f"    {name}: {w.dtype} (requires_grad={w.requires_grad})")
    print(f"[NAN-TRACE]   LayerNorm weight dtypes: {norm_dtypes}", flush=True)
    for ln in head_lines:
        print(f"[NAN-TRACE] {ln}", flush=True)

    # --- Report 2: first non-finite module output ---
    state = {"fired": False}
    handles: list = []

    def _finite(t) -> bool:  # type: ignore[no-untyped-def]
        return bool(torch.isfinite(t).all()) if torch.is_tensor(t) else True

    def _any_nonfinite(obj) -> bool:  # type: ignore[no-untyped-def]
        if torch.is_tensor(obj):
            return obj.is_floating_point() and not _finite(obj)
        if isinstance(obj, (tuple, list)):
            return any(_any_nonfinite(x) for x in obj)
        return False

    def _make_hook(mod_name: str):  # type: ignore[no-untyped-def]
        def _hook(module, inputs, output):  # type: ignore[no-untyped-def]
            if state["fired"]:
                return
            if _any_nonfinite(output):
                in_finite = not _any_nonfinite(inputs)
                state["fired"] = True
                print(
                    f"[NAN-TRACE] FIRST non-finite output at module: {mod_name} "
                    f"({module.__class__.__name__}); inputs_all_finite={in_finite}",
                    flush=True,
                )
                # One extra line: what kind (nan vs +inf vs -inf) for the first bad tensor.
                def _describe(o):  # type: ignore[no-untyped-def]
                    if torch.is_tensor(o) and o.is_floating_point():
                        return (
                            f"nan={int(torch.isnan(o).sum())} "
                            f"posinf={int(torch.isposinf(o).sum())} "
                            f"neginf={int(torch.isneginf(o).sum())} "
                            f"shape={tuple(o.shape)} dtype={o.dtype}"
                        )
                    return "n/a"
                first = output[0] if isinstance(output, (tuple, list)) else output
                print(f"[NAN-TRACE]   output stats: {_describe(first)}", flush=True)
                for h in handles:
                    h.remove()
        return _hook

    n = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            handles.append(module.register_forward_hook(_make_hook(name)))
            n += 1
    print(f"[NAN-TRACE] armed forward hooks on {n} leaf modules", flush=True)


@dataclass
class FinetuneResult:
    model_dir: Path
    labeled_dataset: Path
    id_class_dict: Path
    metrics: dict | None
    test_dataset: Path | None = None


def crossval_coverage(split_sizes: dict, num_crossval_splits: int) -> float:
    """Fraction of all cells Geneformer's ``Classifier.validate`` evaluates on.

    ``prepare_data`` first sets aside ``split_sizes["test"]`` as a held-out set that
    ``validate`` never uses. For 5-fold, each fold then evaluates
    ``min(valid / (train + valid), 0.2)`` of the remainder. Only
    ``train 0.8 / valid 0.2 / test 0`` makes the five folds cover every cell once,
    which is what the scGPT backend's 5-fold CV does.
    """
    train, valid, test = (split_sizes.get(k, 0.0) for k in ("train", "valid", "test"))
    eval_size = valid / (train + valid) if (train + valid) > 0 else 0.0
    if num_crossval_splits == 5:
        return (1.0 - test) * 5 * min(eval_size, 0.2)
    if num_crossval_splits == 1:
        return (1.0 - test) * eval_size
    return 0.0


def _build_training_args(cfg: Config) -> dict:
    """Map our TrainingConfig onto HuggingFace TrainingArguments keys."""
    t = cfg.training
    args = {
        "num_train_epochs": t.epochs,
        "learning_rate": t.learning_rate,
        "per_device_train_batch_size": t.batch_size,
        "per_device_eval_batch_size": t.batch_size,
        "lr_scheduler_type": t.lr_scheduler_type,
        "warmup_steps": t.warmup_steps,
        "weight_decay": t.weight_decay,
        "seed": t.seed,
        # Keep the final-epoch model. Geneformer's cell-classifier defaults set
        # load_best_model_at_end=True with no metric_for_best_model, so HF restores the
        # epoch with the lowest eval loss. Eval loss bottoms out early while macro-F1
        # keeps improving (V2-104M PBMC3k 5-fold: epoch-1 models with macro-F1 0.78-0.83
        # vs 0.94-0.95 at epoch 10), and selecting on the fold that is then scored
        # leaks the eval set into model choice. This also matches the scGPT backend,
        # which reports the final epoch.
        "load_best_model_at_end": False,
    }
    # QLoRA-path numerical-stability args.
    # Full precision is left exactly as before.
    if t.quantize:
        # bf16 compute (NOT fp32): fp32 compute kept activations full-size and OOM'd
        # the first forward at batch 32 on a 24 GB A10G -- the 4-bit weight saving is
        # undone when the dequant matmul runs in fp32, because activation memory
        # scales with the compute dtype, not the weight storage. bf16 halves the
        # activation footprint vs fp32, so a large batch (e.g. 32) fits. bf16=True
        # here aligns the HF Trainer compute dtype with bitsandbytes'
        # bnb_4bit_compute_dtype=bfloat16 (set in _build_quantize_config), avoiding
        # the fp32-Trainer/bf16-bnb dtype mismatch.
        args["bf16"] = True
        args["fp16"] = False
        # max_grad_norm=1.0 -- gradient clipping is the NaN guard in this
        # configuration: bf16 compute previously drove the LoRA deltas to inf within
        # the first optimizer steps (`loss 0.0, grad_norm nan`). Clipping the grad
        # norm to 1.0 before the optimizer step bounds the per-step update so the
        # bf16 forward does not run away, letting bf16's smaller activations coexist
        # with a stable loss. (If clipping alone is insufficient, the next lever is a
        # lower learning rate via --learning-rate.)
        args["max_grad_norm"] = 1.0
    return args


def _build_quantize_config(cfg: Config) -> dict:
    """Build Geneformer's QLoRA ``quantize`` dict: a 4-bit bitsandbytes base
    model plus a LoRA adapter. Geneformer's ``load_model`` accepts ``quantize``
    as a dict of ``{"bnb_config": BitsAndBytesConfig, "peft_config": LoraConfig}``
    and applies both for the fine-tune (see perturber_utils.load_model).

    Imports are local so the full-precision path never needs peft/bitsandbytes
    installed; QLoRA runs require those extras (and a CUDA GPU).
    """
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # Compute the dequantized 4-bit matmuls in bf16. bf16 halves the activation
        # footprint vs fp32 (which OOM'd batch 32 on a 24 GB A10G), and is paired
        # with bf16=True in the Trainer args so the HF autocast path matches this
        # dtype. bf16 is NOT the source of the historic NaN -- see below.
        bnb_4bit_compute_dtype=torch.bfloat16,
        # ROOT CAUSE of the `loss 0.0, grad_norm nan`-from-step-1 divergence
        # (found by the GENEFORMER_FT_NAN_TRACE forward-hook diagnostic): the first
        # non-finite tensor was bert.pooler.dense (a Linear4bit) emitting ALL-NaN
        # from finite inputs. Geneformer-V2 is a masked-LM checkpoint with NO
        # pooler; BertForSequenceClassification ADDS a fresh bert.pooler.dense (and
        # the classifier head). bitsandbytes quantizes every nn.Linear it sees at
        # load time -- including that freshly-initialized pooler -- and NF4-packing
        # an uninitialized weight tensor dequantizes to NaN, which then poisons the
        # head on the very first forward (at any LR/batch, hence immune to grad-clip
        # or a lower LR). Fix: keep the fresh task-head layers (pooler + classifier)
        # OUT of 4-bit so they load/train in full precision; only the pretrained
        # encoder Linears (~85M) are NF4-quantized, which is the real QLoRA win.
        llm_int8_skip_modules=["pooler", "classifier"],
    )
    # Fix 2 of the NaN investigation: the PEFT task_type MUST match the model head.
    # A Geneformer "cell" classifier is BertForSequenceClassification, whose head is
    # SEQ_CLS -- the previous TOKEN_CLS wrapped the wrong module list, so the
    # sequence-classification head was never a trainable LoRA target and eval sat at
    # chance. "gene" is BertForTokenClassification -> TOKEN_CLS. We also pin the
    # classification head into modules_to_save so its fresh (non-quantized) weights
    # are fully trained alongside the LoRA deltas, not frozen.
    if cfg.classifier.kind == "cell":
        task_type = "SEQ_CLS"
        modules_to_save = ["classifier"]
    else:
        task_type = "TOKEN_CLS"
        modules_to_save = ["classifier"]
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=task_type,
        modules_to_save=modules_to_save,
    )
    return {"bnb_config": bnb_config, "peft_config": peft_config}


def _dataset_column(dataset_path: Path, column: str) -> list:
    """One column of the tokenized dataset (a Hugging Face ``datasets`` directory)."""
    from datasets import load_from_disk

    data = load_from_disk(str(dataset_path))
    if column not in data.column_names:
        raise KeyError(
            f"Column {column!r} is not in the tokenized dataset {dataset_path} "
            f"(columns: {data.column_names}). Re-tokenize with classifier.group_column set."
        )
    return list(data[column])


def _group_split_kwargs(cfg: Config, dataset_path: Path, out_dir: Path, prefix: str) -> tuple:
    """``prepare_data`` and ``validate`` kwargs for a donor-aware Geneformer split, and the
    split itself (which the privacy audit needs to tell members from non-members).

    prepare_data holds out the test groups; validate splits the remaining groups into
    train and eval. Geneformer's 5-fold path for cell classifiers slices cells by
    position and cannot be grouped, so it is refused.
    """
    from scfm_ft.splits import group_split, split_record, write_split_record

    column = cfg.classifier.group_column
    t = cfg.training
    if cfg.classifier.kind != "cell":
        raise ValueError("classifier.group_column is only supported for cell classifiers")
    if t.num_crossval_splits == 5:
        raise ValueError(
            "Grouped 5-fold CV is not supported by the Geneformer backend (its 5-fold path "
            "slices cells by position). Use num_crossval_splits 1, or the scGPT backend."
        )
    if t.num_crossval_splits == 1 and t.split_sizes.get("valid", 0) <= 0:
        raise ValueError("A grouped single split needs a non-zero valid fraction in split_sizes")
    groups = _dataset_column(dataset_path, column)
    labels = _dataset_column(dataset_path, cfg.classifier.label_column)
    split = group_split(groups, t.split_sizes, t.seed)
    assignment = {"train": split.train, "valid": split.valid, "test": split.test}
    write_split_record(
        split_record(groups, labels, assignment, column), out_dir / f"{prefix}_group_split.json"
    )
    prepare_kwargs: dict = {}
    if split.test:
        prepare_kwargs["split_id_dict"] = {
            "attr_key": column,
            "train": split.train + split.valid,
            "test": split.test,
        }
    validate_kwargs: dict = {}
    if t.num_crossval_splits == 1:
        validate_kwargs["split_id_dict"] = {"attr_key": column, "train": split.train, "eval": split.valid}
    return prepare_kwargs, validate_kwargs, split


def _find_ksplit_model(out_dir: Path, prefix: str) -> Path:
    """The model ``Classifier.validate`` saved for its single split (newest if several)."""
    found = [
        p for p in out_dir.glob(f"*_geneformer_cellClassifier_{prefix}/ksplit1")
        if (p / "config.json").exists()
    ]
    if not found:
        raise FileNotFoundError(f"No fine-tuned ksplit1 model under {out_dir} for {prefix!r}")
    return max(found, key=lambda p: p.stat().st_mtime)


def _privacy_audit(cc, cfg, out_dir, prefix, labeled_dataset, test_dataset, split) -> dict:
    """Membership-inference audit of the model ``validate`` trained (scfm_ft.privacy).

    Members are the cells of the training groups; non-members are the cells of the eval
    and test groups, none of which the model trained on. Without a group split there is
    no record of which cells ``validate`` held out (it re-splits internally), so the
    audit needs classifier.group_column on this backend.
    """
    if cfg.training.num_crossval_splits != 1:
        print("[privacy-audit] skipped: the Geneformer backend audits num_crossval_splits 1 only", flush=True)
        return {}
    if split is None:
        print(
            "[privacy-audit] skipped: the Geneformer backend needs classifier.group_column "
            "(--group-column) to know which cells were held out",
            flush=True,
        )
        return {}
    import numpy as np
    from datasets import concatenate_datasets, load_from_disk
    from geneformer import evaluation_utils as eu
    from geneformer import perturber_utils as pu

    from scfm_ft.privacy import audit, cap_indices, per_cell_loss, write_report

    column = cfg.classifier.group_column
    cap, seed = cfg.training.privacy_audit_max_cells, int(cfg.training.seed)
    model_dir = _find_ksplit_model(out_dir, prefix)
    n_classes = len(json.loads((model_dir / "config.json").read_text())["id2label"])
    model = pu.load_model("CellClassifier", n_classes, str(model_dir), "eval")

    def _subset(ds, members, limit, salt):
        groups = np.asarray(ds[column])
        idx = cap_indices(np.flatnonzero(np.isin(groups, list(members))), limit, seed + salt)
        return ds.select(idx.tolist()), groups[idx]

    train_ds = load_from_disk(str(labeled_dataset))
    member_ds, member_groups = _subset(train_ds, split.train, cap, 0)
    parts = [_subset(train_ds, split.valid, cap, 1)] if split.valid else []
    if test_dataset is not None and split.test:
        parts.append(_subset(load_from_disk(str(test_dataset)), split.test, cap, 2))
    if not parts or not len(member_ds):
        print("[privacy-audit] skipped: no member or no non-member cells", flush=True)
        return {}
    nonmember_ds = concatenate_datasets([p[0] for p in parts])
    nonmember_groups = np.concatenate([p[1] for p in parts])

    sides = {}
    for side, ds, groups in (
        ("member", member_ds, member_groups),
        ("nonmember", nonmember_ds, nonmember_groups),
    ):
        _pred, y_true, logits, _meta = eu.classifier_predict(
            model, "cell", ds, cfg.runtime.forward_batch_size, cc.gene_token_dict
        )
        loss, prob, correct = per_cell_loss(np.asarray(logits), np.asarray(y_true))
        # classifier_predict can drop the last cell to keep batches >= 2; it keeps order.
        sides[side] = {"loss": loss, "prob": prob, "correct": correct, "groups": groups[: len(loss)]}
    report = audit(sides["member"], sides["nonmember"], seed=seed)
    write_report(report, out_dir / f"{prefix}_privacy_audit.json")
    return {"privacy_audit_auc": report["cells"]["auc"]}


def finetune(cfg: Config, dataset_path: Path | None = None) -> FinetuneResult:
    """Fine-tune the configured Geneformer model.

    If ``dataset_path`` is None, the tokenize stage is run first.
    """
    from geneformer import Classifier

    if dataset_path is None:
        dataset_path = tokenize(cfg)

    if not cfg.model.model_path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found at {cfg.model.model_path}. "
            f"Clone Geneformer into {cfg.model.geneformer_repo} (git-lfs) first."
        )

    out_dir = cfg.model_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.name

    cc_kwargs: dict = {
        "classifier": cfg.classifier.kind,
        "training_args": _build_training_args(cfg),
        "freeze_layers": cfg.training.freeze_layers,
        "num_crossval_splits": cfg.training.num_crossval_splits,
        "split_sizes": cfg.training.split_sizes,
        "forward_batch_size": cfg.runtime.forward_batch_size,
        "model_version": cfg.model.version,
        "nproc": cfg.runtime.nproc,
        "ngpu": cfg.runtime.ngpu,
    }

    # QLoRA: pass Geneformer a dict quantize config (4-bit bnb + LoRA). When
    # False, quantize is left unset (full-precision full fine-tune).
    if cfg.training.quantize:
        cc_kwargs["quantize"] = _build_quantize_config(cfg)

    if cfg.classifier.kind == "cell":
        states = cfg.classifier.states if cfg.classifier.states else "all"
        cc_kwargs["cell_state_dict"] = {"state_key": cfg.classifier.label_column, "states": states}
        if cfg.classifier.rare_threshold:
            cc_kwargs["rare_threshold"] = cfg.classifier.rare_threshold
        if cfg.classifier.max_ncells is not None:
            cc_kwargs["max_ncells"] = cfg.classifier.max_ncells
        if cfg.classifier.max_ncells_per_class is not None:
            cc_kwargs["max_ncells_per_class"] = cfg.classifier.max_ncells_per_class

    cc = Classifier(**cc_kwargs)

    # Donor-aware split: whole groups go to train, valid or test, passed to Geneformer
    # as explicit id lists (split_id_dict).
    prepare_kwargs: dict = {}
    validate_kwargs: dict = {}
    group_split_used = None
    group_column = cfg.classifier.group_column
    if group_column:
        prepare_kwargs, validate_kwargs, group_split_used = _group_split_kwargs(
            cfg, dataset_path, out_dir, prefix
        )

    # 1. Prepare labeled dataset
    cc.prepare_data(
        input_data_file=str(dataset_path),
        output_directory=str(out_dir),
        output_prefix=prefix,
        **prepare_kwargs,
    )

    # prepare_data's output naming depends on the held-out test fraction:
    #   split_sizes["test"] > 0  -> {prefix}_labeled_train.dataset + {prefix}_labeled_test.dataset
    #   split_sizes["test"] == 0 -> {prefix}_labeled.dataset
    # validate()/train_all_data() consume the TRAIN dataset (they do their own
    # train/valid split internally); the test set is held out for final eval.
    has_oos_test = cfg.training.split_sizes.get("test", 0) > 0
    if has_oos_test:
        labeled_dataset = out_dir / f"{prefix}_labeled_train.dataset"
        test_dataset = out_dir / f"{prefix}_labeled_test.dataset"
    else:
        labeled_dataset = out_dir / f"{prefix}_labeled.dataset"
        test_dataset = None
    id_class_dict = out_dir / f"{prefix}_id_class_dict.pkl"

    metrics: dict | None = None

    # On the QLoRA path, run prepare_model_for_kbit_training (fp32 upcast) before LoRA
    # is applied, the primary fix for the NaN divergence (full precision uses a no-op
    # and is unchanged). The transformers 4.46 head-add crash on 4-bit Byte params
    # needs no workaround on transformers 5: it initialises only keys missing from the
    # checkpoint, and its init functions skip every loaded (`_is_hf_initialized`) tensor.
    init_guard = contextlib.ExitStack()
    if cfg.training.quantize:
        init_guard.enter_context(_kbit_prep_get_peft_model())

    if cfg.training.num_crossval_splits == 5:
        cov = crossval_coverage(cfg.training.split_sizes, 5)
        if cov < 0.999:
            print(
                f"[geneformer] WARNING: 5-fold CV will evaluate on only {cov:.0%} of cells "
                f"with split_sizes={cfg.training.split_sizes}. Use train 0.8 / valid 0.2 / "
                f"test 0 (--split-sizes 0.8,0.2,0) to hold out every cell once.",
                flush=True,
            )

    if cfg.training.privacy_audit and cfg.training.num_crossval_splits == 0:
        print("[privacy-audit] skipped: no held-out cells (num_crossval_splits 0)", flush=True)

    # 2. Fine-tune
    if cfg.training.num_crossval_splits == 0:
        # Train on all data, no held-out eval.
        with init_guard:
            cc.train_all_data(
                model_directory=str(cfg.model.model_path),
                prepared_input_data_file=str(labeled_dataset),
                id_class_dict_file=str(id_class_dict),
                output_directory=str(out_dir),
                output_prefix=prefix,
            )
    else:
        # Split (or k-fold) with evaluation; returns metrics incl. confusion matrix.
        with init_guard:
            all_metrics = cc.validate(
                model_directory=str(cfg.model.model_path),
                prepared_input_data_file=str(labeled_dataset),
                id_class_dict_file=str(id_class_dict),
                output_directory=str(out_dir),
                output_prefix=prefix,
                predict_eval=True,
                **validate_kwargs,
            )
        # Confusion matrix is a DataFrame; keep metrics JSON-serializable.
        metrics = {k: v for k, v in all_metrics.items() if k != "conf_matrix"}
        if cfg.training.privacy_audit:
            metrics.update(
                _privacy_audit(
                    cc, cfg, out_dir, prefix, labeled_dataset, test_dataset, group_split_used
                )
            )
        (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        # Persist the confusion matrix (a DataFrame) so the evaluate/report stage
        # can render plots without re-running inference: validate() already
        # evaluated on the held-out test split.
        if "conf_matrix" in all_metrics:
            from scfm_ft.conf_matrix import conf_matrix_path, save_conf_matrix

            save_conf_matrix(all_metrics["conf_matrix"], conf_matrix_path(out_dir, prefix))

    return FinetuneResult(
        model_dir=out_dir,
        labeled_dataset=labeled_dataset,
        id_class_dict=id_class_dict,
        metrics=metrics,
        test_dataset=test_dataset,
    )
