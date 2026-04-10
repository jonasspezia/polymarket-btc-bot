#!/usr/bin/env sh
set -eu

APP_ROOT="/app"
MODEL_TARGET_PATH="${MODEL_TARGET_PATH:-${APP_ROOT}/data/models/lgbm_btc_5m.txt}"
MODEL_SOURCE_PATH="${MODEL_SOURCE_PATH:-}"
BOT_MODE="${BOT_MODE:-validation}"

load_secret() {
  var_name="$1"
  file_var="${var_name}_FILE"
  eval "file_path=\${${file_var}:-}"
  if [ -n "${file_path}" ] && [ -f "${file_path}" ]; then
    value="$(tr -d '\r\n' < "${file_path}")"
    export "${var_name}=${value}"
  fi
}

for secret_name in \
  POLYGON_PRIVATE_KEY \
  FUNDER_ADDRESS \
  POLYMARKET_API_KEY \
  POLYMARKET_API_SECRET \
  POLYMARKET_API_PASSPHRASE
do
  load_secret "${secret_name}"
done

copy_if_exists() {
  src="$1"
  dst="$2"
  if [ -f "${src}" ]; then
    cp "${src}" "${dst}"
  fi
}

mkdir -p \
  "${APP_ROOT}/data/models" \
  "${APP_ROOT}/data/raw" \
  "${APP_ROOT}/data/processed" \
  "${APP_ROOT}/data/artifacts" \
  "${APP_ROOT}/logs"

if [ -n "${MODEL_SOURCE_PATH}" ] && [ -f "${MODEL_SOURCE_PATH}" ]; then
  source_dir="$(dirname "${MODEL_SOURCE_PATH}")"
  source_base="$(basename "${MODEL_SOURCE_PATH}")"
  source_stem="${source_base%.*}"
  target_dir="$(dirname "${MODEL_TARGET_PATH}")"
  target_base="$(basename "${MODEL_TARGET_PATH}")"
  target_stem="${target_base%.*}"

  cp "${MODEL_SOURCE_PATH}" "${MODEL_TARGET_PATH}"
  copy_if_exists "${source_dir}/${source_stem}.metadata.json" "${target_dir}/${target_stem}.metadata.json"
  copy_if_exists "${source_dir}/training_metadata.json" "${target_dir}/training_metadata.json"
  copy_if_exists "${source_dir}/${source_stem}.calibrator.pkl" "${target_dir}/${target_stem}.calibrator.pkl"
  copy_if_exists "${source_dir}/calibrator.pkl" "${target_dir}/calibrator.pkl"
fi

if [ "${SKIP_MODEL_CHECK:-false}" != "true" ] && [ ! -f "${MODEL_TARGET_PATH}" ]; then
  echo "ERRO: modelo não encontrado em ${MODEL_TARGET_PATH}" >&2
  echo "Defina MODEL_SOURCE_PATH ou monte data/models/lgbm_btc_5m.txt no container." >&2
  exit 64
fi

case "${BOT_MODE}" in
  validation)
    exec python -m src.execution.engine --validation-only
    ;;
  dry-run|dry_run)
    exec python -m src.execution.engine --dry-run
    ;;
  live)
    exec python -m src.execution.engine --live
    ;;
  preflight)
    exec python scripts/05_validation_preflight.py
    ;;
  custom)
    if [ "$#" -eq 0 ]; then
      echo "ERRO: BOT_MODE=custom exige um comando no CMD." >&2
      exit 64
    fi
    exec "$@"
    ;;
  *)
    echo "ERRO: BOT_MODE inválido: ${BOT_MODE}" >&2
    exit 64
    ;;
esac
