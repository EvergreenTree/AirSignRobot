#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

gate="${AIRSIGN_STAGE1_GATE:-cup-preflight}"
revision="$(git rev-parse HEAD)"
output_dir="${AIRSIGN_STAGE1_OUTPUT_DIR:-${repo_root}/evidence/stage1-physical-development/${gate}-${revision:0:12}}"
image="${AIRSIGN_CAPTURE_IMAGE:-airsignrobot:task3-${revision:0:12}}"
host_python="${AIRSIGN_HOST_PYTHON:-python3}"

case "${gate}" in
  inspect|cup-preflight|cup|tray-lift|tray-transport|all)
    ;;
  *)
    echo "Unsupported Stage 1 gate: ${gate}" >&2
    exit 2
    ;;
esac
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing evidence capture from a dirty AirSign worktree." >&2
  exit 2
fi
if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "AirSign source revision is not a full Git commit." >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to reuse an existing evidence output path: ${output_dir}" >&2
  exit 2
fi
if ! command -v "${host_python}" >/dev/null 2>&1; then
  echo "Host Python is unavailable: ${host_python}" >&2
  exit 2
fi

if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo docker)
fi

"${docker_command[@]}" build \
  --pull \
  --build-arg "AIRSIGN_REVISION=${revision}" \
  --tag "${image}" \
  .

image_id="$(
  "${docker_command[@]}" image inspect "${image}" --format='{{.Id}}'
)"
if [[ ! "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Built image has no immutable sha256 image ID: ${image_id}" >&2
  exit 2
fi

output_parent="$(dirname "${output_dir}")"
output_name="$(basename "${output_dir}")"
mkdir -p "${output_parent}"
output_parent="$(cd "${output_parent}" && pwd)"
staging_dir="$(
  mktemp -d "${output_parent}/.${output_name}.tmp.XXXXXX"
)"

"${docker_command[@]}" image inspect "${image_id}" \
  >"${staging_dir}/image-inspect.json"
"${docker_command[@]}" run --rm "${image_id}" smoke \
  2>&1 | tee "${staging_dir}/smoke.log"

set +e
"${docker_command[@]}" run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=8g \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e "AIRSIGN_CAPTURE_CONTAINER_IMAGE=${image}" \
  -e "AIRSIGN_CAPTURE_CONTAINER_IMAGE_ID=${image_id}" \
  -v "${staging_dir}:/stage1-output" \
  "${image_id}" \
  stage1-table-setup \
  --gate "${gate}" \
  --output-dir /stage1-output \
  --head-placement A \
  2>&1 | tee "${staging_dir}/controller.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${status}" >"${staging_dir}/container.exit"

if [[ ! -f "${staging_dir}/metrics.json" ]] \
    || [[ ! -f "${staging_dir}/trajectory.json" ]] \
    || [[ ! -f "${staging_dir}/manifest.json" ]]; then
  echo "Controller produced no complete evidence bundle." >&2
  echo "Incomplete staging directory: ${staging_dir}" >&2
  exit 2
fi
"${host_python}" scripts/stage1_capture_bundle.py \
  --bundle "${staging_dir}" \
  --bundle-name "${output_name}" \
  --gate "${gate}" \
  --revision "${revision}" \
  --image "${image}" \
  --image-id "${image_id}"
mv "${staging_dir}" "${output_dir}"
output_dir="${output_parent}/${output_name}"

echo "AirSign revision: ${revision}"
echo "Container image: ${image}"
echo "Container image ID: ${image_id}"
echo "Stage 1 output: ${output_dir}"
echo "Review ${output_dir}/INDEX_ENTRY.md and add its bullet to the Stage 1 evidence index."
exit "${status}"
