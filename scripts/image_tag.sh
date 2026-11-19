#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Sourced by do-fine-tuning-job.sh. Once an ECR repository has immutable tags, `latest`
# stops moving, so the launcher default has to follow the build tags instead.

# newest_build_tag REPO REGION PROFILE
#   Print the highest b<N> tag in the repository (the tags sagemaker/buildspec.yml pushes,
#   N = the CodeBuild build number). Print nothing and return 1 if the lookup fails or no
#   such tag exists.
newest_build_tag() {
  local repo="$1" region="$2" profile="$3" tags tag
  tags="$(aws ecr describe-images --repository-name "${repo}" --region "${region}" \
            --profile "${profile}" --query 'imageDetails[].imageTags[]' --output text 2>/dev/null)" || return 1
  tag="$(printf '%s\n' "${tags}" | tr '\t' '\n' | grep -E '^b[0-9]+$' | sort -t b -k2 -n | tail -n 1)"
  [[ -n "${tag}" ]] || return 1
  printf '%s\n' "${tag}"
}
