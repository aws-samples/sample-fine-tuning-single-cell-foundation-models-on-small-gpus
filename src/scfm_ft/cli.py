# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Command-line interface for scfm-ft.

Subcommands:
  tokenize          raw .h5ad/.loom -> model-ready dataset (backend-specific)
  finetune          fine-tune the configured model (runs tokenize if needed)
  evaluate          evaluate a fine-tuned model + plots
  run               tokenize -> finetune -> evaluate end to end
  make-sample-data  write a synthetic .h5ad for smoke tests
  load-pbmc3k       download + format the real 10x PBMC3k dataset (loom)
  checksums         write or verify a SHA256SUMS manifest for data staged to S3
  validate-config   parse a config and print the resolved settings

The model is selected by ``model.backend`` (geneformer | scgpt); the CLI dispatches
through the backend registry and never imports a model library directly.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from scfm_ft.config import Config

app = typer.Typer(add_completion=False, help="Fine-tune single-cell foundation models (Geneformer, scGPT) on scRNA-seq data.")
console = Console()


def _load(config: Path) -> Config:
    cfg = Config.from_yaml(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@app.command()
def tokenize(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Tokenize the input into a model-ready dataset (via the configured backend)."""
    from scfm_ft.backends import get_backend

    cfg = _load(config)
    backend = get_backend(cfg.model.backend)
    backend.validate_config(cfg)
    with console.status(f"Tokenizing ({backend.name})..."):
        path = backend.tokenize(cfg, cfg.input.h5ad_path).dataset_path
    rprint(f"[green]Tokenized dataset:[/green] {path}")


@app.command()
def finetune(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Fine-tune the configured model (via the configured backend)."""
    from scfm_ft.backends import get_backend

    cfg = _load(config)
    backend = get_backend(cfg.model.backend)
    backend.validate_config(cfg)
    dataset = backend.tokenize(cfg, cfg.input.h5ad_path).dataset_path
    result = backend.finetune(cfg, dataset)
    rprint(f"[green]Fine-tuned model:[/green] {result.model_dir}")
    if result.metrics:
        rprint("[bold]Metrics:[/bold]")
        rprint(result.metrics)


@app.command()
def evaluate(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Evaluate a fine-tuned model and write plots (via the configured backend)."""
    from scfm_ft.backends import get_backend

    cfg = _load(config)
    backend = get_backend(cfg.model.backend)
    metrics = backend.evaluate(cfg)
    rprint(f"[green]Evaluation written to:[/green] {cfg.eval_dir}")
    rprint(metrics)


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Run the full pipeline: tokenize -> finetune -> evaluate (via the backend)."""
    from scfm_ft.backends import get_backend

    cfg = _load(config)
    backend = get_backend(cfg.model.backend)
    backend.validate_config(cfg)
    rprint(f"[bold cyan]1/3 Tokenize[/bold cyan] [dim]({backend.name})[/dim]")
    dataset = backend.tokenize(cfg, cfg.input.h5ad_path).dataset_path
    rprint("[bold cyan]2/3 Fine-tune[/bold cyan]")
    backend.finetune(cfg, dataset)
    rprint("[bold cyan]3/3 Evaluate[/bold cyan]")
    metrics = backend.evaluate(cfg)
    rprint("[green bold]Done.[/green bold]")
    rprint(metrics)


@app.command("make-sample-data")
def make_sample_data(
    out: Path = typer.Option(Path("data/sample.h5ad"), "--out", "-o"),
    n_cells: int = typer.Option(200, "--n-cells"),
    label_column: str = typer.Option("cell_type", "--label-column"),
    n_classes: int = typer.Option(3, "--n-classes"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate a small synthetic .h5ad (for smoke tests; not biologically meaningful)."""
    from scfm_ft.data import make_sample_data as _make

    path = _make(out, n_cells=n_cells, label_column=label_column, n_classes=n_classes, seed=seed)
    rprint(f"[green]Wrote synthetic dataset:[/green] {path} ({n_cells} cells, {n_classes} classes)")


@app.command("load-pbmc3k")
def load_pbmc3k(
    out: Path = typer.Option(Path("data/pbmc3k.loom"), "--out", "-o"),
    label_column: str = typer.Option("cell_type", "--label-column"),
    min_genes: int = typer.Option(200, "--min-genes"),
) -> None:
    """Download the real 10x PBMC3k dataset and write a Geneformer-ready .loom.

    ~2,700 PBMCs with standard Louvain cell-type labels (CD4/CD8 T, B, NK,
    monocyte, DC, megakaryocyte). Requires network access (scanpy fetches the data).
    """
    from scfm_ft.datasets import load_pbmc3k as _load_pbmc3k

    with console.status("Fetching + formatting PBMC3k..."):
        path = _load_pbmc3k(out, label_column=label_column, min_genes=min_genes)
    rprint(f"[green]Wrote PBMC3k dataset:[/green] {path}")
    rprint(f"[dim]Label column:[/dim] {label_column} (real cell types)")


@app.command("checksums")
def checksums(
    directory: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory to checksum."),
    verify: bool = typer.Option(False, "--verify", help="Check the directory against its SHA256SUMS instead of writing one."),
    strict: bool = typer.Option(False, "--strict", help="With --verify: also fail on files the manifest does not list (what --require-input-hashes enforces in the job)."),
) -> None:
    """Write (or verify) a SHA256SUMS manifest for data you stage to S3.

    Upload the manifest next to the files. The SageMaker job checks the channel against it
    before reading anything, and refuses to start if a file changed.
    """
    from scfm_ft.integrity import IntegrityError, verify_channel, write_sha256sums

    if not verify:
        from scfm_ft.integrity import sha256_file

        path = write_sha256sums(directory)
        rprint(f"[green]Wrote[/green] {path} ({len(path.read_text().splitlines())} file(s))")
        rprint(
            f"Manifest SHA-256: {sha256_file(path)}\n"
            "Record it outside the bucket and pass it to the launcher with --manifest-sha256 train=<that hash>."
        )
        return
    try:
        result = verify_channel(directory, require=strict)
    except IntegrityError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    if not result.manifest_present:
        raise typer.Exit(code=1)


@app.command("privacy-check")
def privacy_check(
    report: Path = typer.Argument(..., exists=True, dir_okay=False, help="A <name>_privacy_audit.json from a training job."),
) -> None:
    """Apply the release rule to a membership-inference audit; exit 1 if it fails.

    A model trained on human donor data may be shared beyond the people allowed to see the
    data only if this passes (docs/security/aws-baseline.md, "Sharing a model").
    """
    import json

    from scfm_ft.privacy import print_release_check, summary_lines

    data = json.loads(report.read_text())
    for k, rep in enumerate(data.get("folds") or [data]):
        prefix = f"fold {k + 1}: " if "folds" in data else ""
        for line in summary_lines(rep):
            rprint(f"{prefix}{line}")
    if not print_release_check(data):
        raise typer.Exit(code=1)


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Parse and validate a config, printing the resolved settings."""
    cfg = Config.from_yaml(config)
    rprint("[green]Config is valid.[/green]")
    rprint(cfg.model_dump())
    rprint(f"[dim]Backend:[/dim] {cfg.model.backend}")
    if cfg.model.backend == "geneformer":
        rprint(f"[dim]Base model path:[/dim] {cfg.model.model_path}")
        rprint(f"[dim]Model input size:[/dim] {cfg.model.input_size}")
    else:
        from scfm_ft.backends import get_backend

        rprint(f"[dim]Checkpoint:[/dim] {get_backend(cfg.model.backend).resolve_checkpoint(cfg)}")


if __name__ == "__main__":
    app()
