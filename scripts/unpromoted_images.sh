#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Lists images that were pushed under a staging tag (gate-b<N>) but never got their b<N> tag,
# which means their build failed the scan gate (or is still running). Read-only: it deletes
# nothing and the launcher's default image never picks them.
#
#   scripts/unpromoted_images.sh REPO REGION PROFILE
#
# Output, one image per line: DIGEST <tab> TAGS <tab> PUSHED_AT. Delete one you no longer need
# with: aws ecr batch-delete-image --repository-name REPO --image-ids imageDigest=DIGEST
# Skip any pushed in the last half hour; its build may not have finished.
#
# Do not use an ECR lifecycle rule on the `gate-` prefix for this. A promoted image carries both
# gate-b<N> and b<N>, and a rule that matches any of an image's tags would expire the good image.

unpromoted_images() {
  local repo="$1" region="$2" profile="$3" json
  json="$(aws ecr describe-images --repository-name "${repo}" --region "${region}" \
            --profile "${profile}" --output json 2>/dev/null)" || return 1
  printf '%s' "${json}" | python3 -c '
import json, re, sys
for img in json.load(sys.stdin).get("imageDetails", []):
    tags = img.get("imageTags", [])
    if any(re.fullmatch(r"gate-b[0-9]+", t) for t in tags) and not any(re.fullmatch(r"b[0-9]+", t) for t in tags):
        print(img["imageDigest"], ",".join(tags), str(img.get("imagePushedAt", "")), sep="\t")
'
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ $# -eq 3 ]] || { echo "usage: $0 REPO REGION PROFILE" >&2; exit 2; }
  set -euo pipefail
  unpromoted_images "$@" || { echo "ERROR: could not list images in $1" >&2; exit 1; }
fi
