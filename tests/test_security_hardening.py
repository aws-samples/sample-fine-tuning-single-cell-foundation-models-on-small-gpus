# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Security hardening: JSON confusion matrices, safe checkpoint loading, pinned
Geneformer, lockfile-based image install, scoped SageMaker execution policy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scfm_ft.conf_matrix import conf_matrix_path, load_conf_matrix, save_conf_matrix

ROOT = Path(__file__).resolve().parents[1]


# --- confusion matrix as JSON, not pickle ---------------------------------------------
def test_conf_matrix_round_trips_labels_order_and_values(tmp_path):
    labels = ["T cells", "1", "B cells"]  # "1" must stay a string label
    df = pd.DataFrame([[5.0, 1.0, 0.0], [0.0, 7.0, 2.0], [1.0, 0.0, 9.0]], index=labels, columns=labels)
    path = save_conf_matrix(df, conf_matrix_path(tmp_path, "run"))
    assert path.name == "run_conf_matrix.json"
    json.loads(path.read_text())  # plain JSON
    back = load_conf_matrix(path)
    assert list(back.index) == labels
    assert list(back.columns) == labels
    assert back.to_numpy().tolist() == df.to_numpy().tolist()


def test_no_pickle_left_in_our_code():
    banned = re.compile(r"import pickle|pickle\.(load|dump)|allow_pickle=True")
    for py in (ROOT / "src" / "scfm_ft").rglob("*.py"):
        assert not banned.search(py.read_text()), f"pickle still used in {py}"


def test_scgpt_npz_round_trips_without_pickle(tmp_path):
    import numpy as np

    labels = np.array(["B cells", "NK cells"], dtype=object)  # how loaders often return them
    path = tmp_path / "d.npz"
    np.savez(path, labels=np.asarray(labels).astype(str), pad_token_id=np.int64(3))
    data = np.load(path, allow_pickle=False)
    assert data["labels"].tolist() == ["B cells", "NK cells"]


# --- torch.load(weights_only=True) ------------------------------------------------------
class _NotATensor:
    """A pickled object a tampered checkpoint could smuggle in."""


def test_scgpt_checkpoint_load_refuses_non_tensor_objects(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    import types

    from torch import nn

    class _Fake(nn.Module):
        def __init__(self, ntoken, d_model, n_cls, **_kw):
            super().__init__()
            self.cls_decoder = nn.Linear(4, n_cls)

    mod = types.ModuleType("scgpt.model")
    mod.TransformerModel = _Fake
    monkeypatch.setitem(sys.modules, "scgpt", types.ModuleType("scgpt"))
    monkeypatch.setitem(sys.modules, "scgpt.model", mod)
    from scfm_ft.backends.scgpt import backend

    class _Vocab(dict):
        def set_default_index(self, _i):
            pass

    monkeypatch.setattr(backend, "_load_gene_vocab", lambda _d: _Vocab({"g0": 0, "<pad>": 1}))
    torch.save({"payload": _NotATensor()}, tmp_path / "model.pt")

    from types import SimpleNamespace as NS

    import numpy as np

    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    cfg = NS(
        name="t",
        training=NS(num_crossval_splits=1, split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
                    seed=0, batch_size=4, learning_rate=1e-3, epochs=1),
        scgpt=NS(n_input_bins=51, use_flash_attn=False, do_mvc=False, freeze_backbone=False,
                 grad_checkpointing=False, max_seq_len=0),
    )
    labels = np.array(["A"] * 10 + ["B"] * 10)
    with pytest.raises(Exception, match="(?i)weights.only|unpickl|not allowed|unsupported"):
        train_scgpt_classifier(tmp_path, np.arange(1), np.ones((20, 1), "float32"), labels, 1, cfg, tmp_path)


# --- pinned Geneformer --------------------------------------------------------------------
def test_geneformer_revision_is_pinned_identically_everywhere():
    sha = re.compile(r"\b[0-9a-f]{40}\b")
    docker = sha.findall((ROOT / "sagemaker" / "Dockerfile").read_text())
    setup = sha.findall((ROOT / "scripts" / "setup.sh").read_text())
    fetch = sha.findall((ROOT / "scripts" / "fetch_weights.py").read_text())
    assert docker and setup and fetch
    assert set(docker) == set(setup) == set(fetch)
    fetch_src = (ROOT / "scripts" / "fetch_weights.py").read_text()
    assert fetch_src.count("hf_hub_download(") == fetch_src.count("revision=REVISION") >= 2


# --- image built from the lockfile ---------------------------------------------------------
def test_geneformer_image_installs_from_the_lockfile_with_hashes():
    docker = (ROOT / "sagemaker" / "Dockerfile").read_text()
    assert "COPY pyproject.toml README.md uv.lock" in docker
    assert "uv export --frozen" in docker
    assert "--require-hashes" in docker
    assert "uv venv --python 3.11" in docker
    assert not re.search(r"^RUN pip3 install \.", docker, re.MULTILINE)  # no unpinned resolve
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert "uv.lock" not in ignored


# --- scoped SageMaker execution policy -------------------------------------------------
def test_execution_policy_is_scoped():
    policy = json.loads((ROOT / "docs" / "iam" / "sagemaker-execution-policy.json").read_text())
    actions = []
    for st in policy["Statement"]:
        acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        actions += acts
        assert st["Effect"] == "Allow"
        # Wildcard resources only for actions that cannot be scoped.
        if st["Resource"] == "*":
            assert set(acts) <= {"ecr:GetAuthorizationToken", "cloudwatch:PutMetricData"}
    assert not any(a.endswith(":*") or a == "*" for a in actions)
    assert not any(a.startswith(("iam:", "sagemaker:")) for a in actions)
    assert "s3:DeleteObject" not in actions
    readme = (ROOT / "README.md").read_text()
    assert "docs/iam/sagemaker-execution-policy.json" in readme
    assert "rather than\n     `AmazonSageMakerFullAccess`" in readme


def test_trust_policy_limits_to_own_training_jobs():
    trust = json.loads((ROOT / "docs" / "iam" / "sagemaker-trust-policy.json").read_text())
    (st,) = trust["Statement"]
    assert st["Principal"] == {"Service": "sagemaker.amazonaws.com"}
    assert st["Condition"]["StringEquals"]["aws:SourceAccount"] == "ACCOUNT_ID"
    assert st["Condition"]["ArnLike"]["aws:SourceArn"].endswith(":training-job/*")


# --- lockfile carries only patched releases -----------------------------------------------
def _locked_versions(name):
    text = (ROOT / "uv.lock").read_text()
    return re.findall(rf'^name = "{name}"\nversion = "([^"]+)"', text, re.MULTILINE)


def _ver(v):
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


@pytest.mark.parametrize(
    ("pkg", "floor"),
    [("pillow", "13.0.0"), ("filelock", "4.2.1"), ("certifi", "2026.10.18"),
     ("setuptools", "84.1.0"),
     # AR-1, closed by upgrading: 5.10 has every published transformers fix,
     # datasets 5.0.1 fixes CVE-2026-66007, fsspec 2026.6.0 fixes CVE-2026-104851.
     ("transformers", "5.10.0"), ("datasets", "5.0.1"), ("fsspec", "2026.6.0")],
)
def test_lock_has_no_vulnerable_versions(pkg, floor):
    versions = _locked_versions(pkg)
    assert versions, f"{pkg} missing from uv.lock"
    assert all(_ver(v) >= _ver(floor) for v in versions), versions


def test_lock_has_no_python_310_resolution():
    # A 3.10 branch is how pillow 12.3.0 / filelock 4.1.0 stayed in the lock.
    assert "python_full_version < '3.11'" not in (ROOT / "uv.lock").read_text()


# --- license headers --------------------------------------------------------------------
def test_every_source_file_has_the_license_header():
    files = [
        p.relative_to(ROOT)
        for top in ("sagemaker", "scripts", "src", "tests")
        for pattern in ("*.py", "*.sh")
        for p in (ROOT / top).rglob(pattern)
        if "__pycache__" not in p.parts
    ]
    assert len(files) >= 39
    missing = [
        f for f in files
        if "SPDX-License-Identifier: MIT-0" not in "".join((ROOT / f).read_text().splitlines(True)[:4])
    ]
    assert not missing, missing


# --- Dockerfile hardening decisions (Checkov CKV_DOCKER_2 / CKV_DOCKER_3) ----------------
@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.scgpt"])
def test_dockerfiles_declare_healthcheck_and_justify_root(name):
    docker = (ROOT / "sagemaker" / name).read_text()
    assert re.search(r"^HEALTHCHECK ", docker, re.MULTILINE)
    assert "checkov:skip=CKV_DOCKER_3:" in docker
    # SageMaker writes /opt/ml as root; a USER switch would break model output.
    assert not re.search(r"^USER ", docker, re.MULTILINE)


def test_execution_policy_s3_writes_are_limited_to_job_output():
    policy = json.loads((ROOT / "docs" / "iam" / "sagemaker-execution-policy.json").read_text())
    by_sid = {st["Sid"]: st for st in policy["Statement"]}
    assert by_sid["WriteModelOutput"]["Resource"] == (
        "arn:aws:s3:::sagemaker-REGION-ACCOUNT_ID/scfm-ft/output/*"
    )
    reads = by_sid["ReadInputsAndWeights"]["Resource"]
    # The launcher uploads no code (T018), so the job has no business reading a code prefix.
    assert not any("/code" in r for r in reads), reads
    assert "arn:aws:s3:::sagemaker-REGION-ACCOUNT_ID/*" not in reads


def test_sm_entry_has_no_fixed_tmp_path():
    src = (ROOT / "sagemaker" / "sm_entry.py").read_text()
    assert '"/tmp/' not in src
    assert "tempfile.mkdtemp(" in src


# --- untrusted-input loading paths (hygiene; was AR-1's conditions) ------------------------
def test_no_untrusted_model_or_data_loading_paths_in_project_code():
    # These are the trigger paths of the transformers/datasets/fsspec RCE advisories that
    # AR-1 once accepted. Fixed now by the upgrade, but there is no reason to start
    # using them; fail loudly if code does.
    forbidden = {
        "trust_remote_code=True": "remote code from a model repo (CVE-2026-5241/4372)",
        "resume_from_checkpoint": "Trainer rng_state torch.load (CVE-2026-1839)",
        "load_dataset(": "folder-builder file_name traversal (CVE-2026-66007)",
        "reference://": "fsspec ReferenceFileSystem (CVE-2026-104851)",
    }
    hits = []
    for top in ("src", "sagemaker", "scripts"):
        for p in (ROOT / top).rglob("*.py"):
            text = p.read_text()
            hits += [f"{p.relative_to(ROOT)}: {pat} ({why})" for pat, why in forbidden.items() if pat in text]
    assert not hits, "untrusted-input loading path in project code:\n" + "\n".join(hits)


def test_accepted_risk_register_is_linked_and_ar1_is_closed():
    doc = (ROOT / "docs" / "security" / "accepted-risks.md").read_text()
    assert re.search(r"\| AR-1 \|[^\n]*\*\*Closed 2026-11-11\*\*", doc)
    assert "docs/security/accepted-risks.md" in (ROOT / "README.md").read_text()


def test_execution_policy_s3_reads_and_lists_are_prefix_scoped():
    policy = json.loads((ROOT / "docs" / "iam" / "sagemaker-execution-policy.json").read_text())
    for st in policy["Statement"]:
        acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        resources = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
        if "s3:GetObject" in acts or "s3:PutObject" in acts:
            for arn in resources:
                # arn:aws:s3:::<bucket>/<prefix>/* -- never a whole bucket (<bucket>/*).
                key = arn.split(":::", 1)[1].split("/", 1)[1]
                assert key != "*" and key.endswith("/*"), f"{st['Sid']}: whole-bucket grant {arn}"
        if "s3:ListBucket" in acts:
            assert len(resources) == 1, f"{st['Sid']}: one bucket per list statement"
            prefixes = st["Condition"]["StringLike"]["s3:prefix"]
            assert prefixes and all(p.endswith("/*") and p != "/*" for p in prefixes), st["Sid"]


# --- scanner ignore file mirrors the accepted-risk register --------------------------------
def test_osv_ignore_file_matches_accepted_risk_doc():
    import tomllib

    path = ROOT / "osv-scanner.toml"
    if not path.exists():
        # No open accepted risks (AR-1 closed). A stale ignore file would only hide a
        # regression if someone downgraded, so its absence is the expected state.
        return
    ignore = tomllib.loads(path.read_text())
    entries = ignore.get("IgnoredVulns", [])
    doc = (ROOT / "docs" / "security" / "accepted-risks.md").read_text()

    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ignore entries"
    for e in entries:
        # Every ignored CVE must be one the register accepts, name its entry, and expire.
        assert e["id"] in doc, f"{e['id']} is ignored but not in accepted-risks.md"
        assert re.match(r"AR-\d+:", e["reason"]), e["id"]
        assert "effectiveUntil" in e, f"{e['id']} has no expiry"
    # Per-ID ignores only: a package-wide override would also hide future CVEs.
    assert "PackageOverrides" not in ignore
