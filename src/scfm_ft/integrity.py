# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Content-hash verification for inputs the job did not build itself.

Two uses:

* **Channel manifests.** A SageMaker input channel (training data, scGPT weights) is
  whatever sits at an S3 prefix at job start. Put a standard ``SHA256SUMS`` file in that
  prefix (``scfm-ft checksums DIR`` writes one) and the entry point verifies every listed
  file before anything reads it. With ``require`` set, the manifest is mandatory and the
  channel may contain no file the manifest does not list, so nothing unverified can be
  picked up by a glob.
* **Pinned downloads.** :func:`download_verified` fetches a file over HTTPS and keeps it
  only if its size and SHA-256 match values pinned in the code.

SHA-256 detects accidental and malicious modification of a file after its hash was
recorded. It does not tell you the file was good when you recorded it, and the manifest
lives next to the files, so anyone who can write the bucket can rewrite both. For that,
keep the expected hashes somewhere the bucket's writers cannot reach (see
``docs/security/aws-baseline.md``).
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "SHA256SUMS"
_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
_CHUNK = 1024 * 1024


class IntegrityError(RuntimeError):
    """A file's hash, size or place in a manifest did not match what was expected."""


@dataclass(frozen=True)
class VerifyResult:
    manifest_present: bool
    files_verified: int


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` output into ``{relative path: hex digest}``.

    Rejects absolute paths, ``..`` components and duplicates, so a manifest cannot point
    outside the channel directory it sits in.
    """
    entries: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line.rstrip("\r"))
        if not m:
            raise IntegrityError(f"{MANIFEST_NAME} line {n} is not '<sha256>  <path>': {line[:60]!r}")
        digest, rel = m.group(1).lower(), m.group(2)
        parts = Path(rel).parts
        if parts[:1] == (".",):
            parts = parts[1:]
        if not parts or Path(rel).is_absolute() or ".." in parts:
            raise IntegrityError(f"{MANIFEST_NAME} line {n} has an unsafe path: {rel!r}")
        key = "/".join(parts)
        if key in entries:
            raise IntegrityError(f"{MANIFEST_NAME} lists {key!r} twice")
        entries[key] = digest
    return entries


PIN_CHANNELS = ("train", "weights")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def parse_manifest_pins(text: str) -> dict[str, str]:
    """Parse ``train=<sha256>,weights=<sha256>`` into ``{"train": ..., "weights": ...}``.

    A bare 64-character hash means the ``train`` channel. Anything else raises
    :class:`IntegrityError`, so a typo can never silently turn a pin off.
    """
    text = text.strip()
    if not text:
        return {}
    if _SHA256_HEX.match(text.lower()):
        text = f"train={text}"
    pins: dict[str, str] = {}
    for part in text.split(","):
        name, sep, value = part.strip().partition("=")
        value = value.strip().lower()
        if not sep or name not in PIN_CHANNELS or not _SHA256_HEX.match(value):
            raise IntegrityError(
                f"Bad manifest pin {part.strip()!r}: use CHANNEL=<64 hex characters> with CHANNEL one of "
                f"{', '.join(PIN_CHANNELS)}."
            )
        if name in pins:
            raise IntegrityError(f"Manifest pin for the {name} channel is given twice.")
        pins[name] = value
    return pins


def verify_channel(
    directory: str | Path, *, require: bool = False, manifest_sha256: str | None = None
) -> VerifyResult:
    """Verify ``directory`` against its ``SHA256SUMS``; raise :class:`IntegrityError` on a mismatch.

    Without ``require``, a missing manifest only prints a warning (existing workflows keep
    working) but a present one is always enforced. With ``require``, the manifest must
    exist and must list every file in the directory.

    ``manifest_sha256`` pins the manifest itself: the expected SHA-256 of the ``SHA256SUMS``
    file, recorded somewhere the bucket's writers cannot reach. A manifest that does not
    hash to it (or is missing) is rejected, so someone who can write the bucket cannot swap
    a file and its manifest together. A pin makes the manifest mandatory.
    """
    root = Path(directory)
    manifest = root / MANIFEST_NAME
    if manifest_sha256 is not None:
        if not manifest.is_file():
            raise IntegrityError(f"{MANIFEST_NAME} is pinned but missing from {root}.")
        if sha256_file(manifest) != manifest_sha256.lower():
            raise IntegrityError(
                f"{MANIFEST_NAME} in {root} does not match the pinned hash; the manifest was changed "
                f"after the pin was recorded."
            )
        require = True
    if not manifest.is_file():
        if require:
            raise IntegrityError(
                f"{MANIFEST_NAME} is required but missing from {root}. Create one with "
                f"'scfm-ft checksums <dir>' and upload it with the data."
            )
        print(f"[integrity] WARNING: no {MANIFEST_NAME} in {root}; input contents are unverified.")
        return VerifyResult(False, 0)

    entries = parse_sha256sums(manifest.read_text())
    root_resolved = root.resolve()
    problems: list[str] = []
    for rel, expected in sorted(entries.items()):
        target = root / rel
        if target.is_symlink() or not target.resolve().is_relative_to(root_resolved):
            problems.append(f"{rel}: is a symlink or points outside the channel")
        elif not target.is_file():
            problems.append(f"{rel}: listed in {MANIFEST_NAME} but missing")
        elif sha256_file(target) != expected:
            problems.append(f"{rel}: SHA-256 does not match {MANIFEST_NAME}")

    unlisted = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != manifest and p.relative_to(root).as_posix() not in entries
    )
    if unlisted:
        if require:
            problems.append(f"files not listed in {MANIFEST_NAME}: {', '.join(unlisted[:5])}")
        else:
            print(f"[integrity] WARNING: {len(unlisted)} file(s) not in {MANIFEST_NAME}: {unlisted[:5]}")
    if problems:
        raise IntegrityError(f"Input integrity check failed for {root}:\n  " + "\n  ".join(problems))
    print(f"[integrity] {root}: {len(entries)} file(s) match {MANIFEST_NAME}.")
    return VerifyResult(True, len(entries))


def write_sha256sums(directory: str | Path) -> Path:
    """Write ``SHA256SUMS`` for every file under ``directory`` (sorted, relative paths)."""
    root = Path(directory)
    manifest = root / MANIFEST_NAME
    lines = [
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}"
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != manifest
    ]
    if not lines:
        raise IntegrityError(f"No files under {root} to checksum.")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def download_verified(
    url: str,
    dest: str | Path,
    *,
    sha256: str,
    size: int,
    opener: Callable = urllib.request.urlopen,
    timeout: float = 120.0,
) -> Path:
    """Download ``url`` to ``dest`` over HTTPS and keep it only if size and SHA-256 match.

    The bytes go to a temporary ``.part`` file and are renamed into place only after the
    hash matches, so a failed or tampered download never leaves a usable file behind. The
    size is enforced while streaming, so a hostile server cannot make us read more than
    the pinned size.
    """
    if urllib.parse.urlparse(url).scheme != "https":
        raise IntegrityError(f"Refusing non-HTTPS download: {url}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    total = 0
    try:
        with opener(url, timeout=timeout) as resp, open(part, "wb") as out:  # nosec B310 - https only, checked above
            final = getattr(resp, "geturl", lambda: url)()
            if urllib.parse.urlparse(final).scheme != "https":
                raise IntegrityError(f"{url} redirected to a non-HTTPS URL: {final}")
            for block in iter(lambda: resp.read(_CHUNK), b""):
                total += len(block)
                if total > size:
                    raise IntegrityError(f"{url} is larger than the pinned size ({size} bytes).")
                h.update(block)
                out.write(block)
        if total != size or h.hexdigest() != sha256.lower():
            raise IntegrityError(
                f"{url} does not match its pinned SHA-256/size "
                f"(got {total} bytes, sha256 {h.hexdigest()[:16]}...; expected {size} bytes, "
                f"sha256 {sha256[:16]}...). The upstream file changed or was tampered with."
            )
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    return dest
