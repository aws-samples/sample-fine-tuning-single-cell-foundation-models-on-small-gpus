# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 3: report evaluation results from the fine-tune stage.

Important: when ``training.num_crossval_splits >= 1``, the fine-tune stage already
evaluates the model on a held-out test split inside ``Classifier.validate`` and
returns the metrics + confusion matrix. This stage therefore does NOT re-run
inference (doing so against the full labeled dataset would leak training cells into
the "test" numbers). Instead it loads the confusion matrix persisted by the
fine-tune stage and renders the confusion-matrix figure via ``plot_conf_mat``.

For a *separate* held-out dataset (e.g. a different cohort), use
``evaluate_saved_model`` directly against the datestamped fine-tuned model directory
``{model_out}/{datestamp}_geneformer_cellClassifier_{prefix}/ksplit1`` — see
``evaluate_external`` below.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from scfm_ft.conf_matrix import conf_matrix_path, load_conf_matrix
from scfm_ft.config import Config


def _find_finetuned_model_dir(model_out: Path, prefix: str) -> Path | None:
    """Resolve the datestamped subdir where Classifier wrote the fine-tuned model."""
    hits = sorted(glob.glob(str(model_out / f"*geneformer*{prefix}" / "ksplit*")))
    if not hits:
        hits = sorted(glob.glob(str(model_out / f"*geneformer*{prefix}*")))
    return Path(hits[-1]) if hits else None


def _classifier_kwargs(cfg: Config) -> dict:
    """Arguments for a Geneformer ``Classifier`` used only to plot or score results.

    Geneformer's ``Classifier.validate_options`` requires ``cell_state_dict`` for a cell
    classifier and raises ``AttributeError`` on ``None``, so it is built here the same
    way ``finetune`` builds it.
    """
    kwargs: dict = {
        "classifier": cfg.classifier.kind,
        "model_version": cfg.model.version,
        "nproc": cfg.runtime.nproc,
        "ngpu": cfg.runtime.ngpu,
    }
    if cfg.classifier.kind == "cell":
        states = cfg.classifier.states if cfg.classifier.states else "all"
        kwargs["cell_state_dict"] = {"state_key": cfg.classifier.label_column, "states": states}
    return kwargs


def evaluate(cfg: Config) -> dict:
    """Render evaluation report (metrics JSON + confusion matrix) from fine-tune outputs.

    Returns the metrics dict and writes a confusion-matrix figure to ``cfg.eval_dir``.
    """
    from geneformer import Classifier

    model_out = cfg.model_out_dir
    prefix = cfg.name
    metrics_file = model_out / f"{prefix}_metrics.json"
    conf_mat_file = conf_matrix_path(model_out, prefix)

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"No fine-tune metrics at {metrics_file}. Run `finetune` with "
            f"num_crossval_splits >= 1 first (num_crossval_splits=0 trains without eval)."
        )

    eval_dir = cfg.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(metrics_file.read_text())
    (eval_dir / f"{prefix}_eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    if conf_mat_file.exists():
        conf_matrix = load_conf_matrix(conf_mat_file)
        cc = Classifier(**_classifier_kwargs(cfg))
        cc.plot_conf_mat(
            conf_mat_dict={"Geneformer": conf_matrix},
            output_directory=str(eval_dir),
            output_prefix=prefix,
        )

    return metrics


def evaluate_external(cfg: Config, test_data_file: str | Path) -> dict:
    """Evaluate the fine-tuned model on a SEPARATE held-out ``.dataset``.

    ``test_data_file`` must be a Geneformer ``.dataset`` (tokenized + labeled with the
    same label scheme). This resolves the datestamped fine-tuned model directory and
    runs ``Classifier.evaluate_saved_model`` against it.
    """
    from geneformer import Classifier

    model_out = cfg.model_out_dir
    prefix = cfg.name
    id_class_dict = model_out / f"{prefix}_id_class_dict.pkl"

    model_dir = _find_finetuned_model_dir(model_out, prefix)
    if model_dir is None:
        raise FileNotFoundError(
            f"Could not locate a fine-tuned model directory under {model_out}. "
            f"Run `finetune` first."
        )

    eval_dir = cfg.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    cc = Classifier(
        **_classifier_kwargs(cfg),
        forward_batch_size=cfg.runtime.forward_batch_size,
    )
    metrics = cc.evaluate_saved_model(
        model_directory=str(model_dir),
        id_class_dict_file=str(id_class_dict),
        test_data_file=str(test_data_file),
        output_directory=str(eval_dir),
        output_prefix=f"{prefix}_external",
        predict=True,
    )

    serializable = {k: v for k, v in metrics.items() if k != "conf_matrix"}
    (eval_dir / f"{prefix}_external_metrics.json").write_text(
        json.dumps(serializable, indent=2, default=str)
    )
    if "conf_matrix" in metrics:
        cc.plot_conf_mat(
            conf_mat_dict={"Geneformer": metrics["conf_matrix"]},
            output_directory=str(eval_dir),
            output_prefix=f"{prefix}_external",
        )
    return serializable
