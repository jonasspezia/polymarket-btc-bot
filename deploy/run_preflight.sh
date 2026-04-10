#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_REF="${IMAGE_REF:-polymarket-btc-bot-ishant5436:local}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERRO: arquivo obrigatório ausente: ${path}" >&2
    exit 64
  fi
}

mkdir -p "${ROOT_DIR}/data/models" "${ROOT_DIR}/data/artifacts" "${ROOT_DIR}/logs" "${ROOT_DIR}/secrets"

require_file "${ROOT_DIR}/.env"
require_file "${ROOT_DIR}/data/models/lgbm_btc_5m.txt"
require_file "${ROOT_DIR}/secrets/polygon_private_key"
require_file "${ROOT_DIR}/secrets/funder_address"
require_file "${ROOT_DIR}/secrets/polymarket_api_key"
require_file "${ROOT_DIR}/secrets/polymarket_api_secret"
require_file "${ROOT_DIR}/secrets/polymarket_api_passphrase"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERRO: docker não encontrado no host." >&2
  exit 127
fi

echo "==> Building image ${IMAGE_REF}"
docker build -t "${IMAGE_REF}" "${ROOT_DIR}"

echo "==> Running preflight"
docker run --rm \
  --name polymarket-btc-bot-preflight \
  -e BOT_MODE=preflight \
  -e POLYGON_PRIVATE_KEY_FILE=/run/secrets/polygon_private_key \
  -e FUNDER_ADDRESS_FILE=/run/secrets/funder_address \
  -e POLYMARKET_API_KEY_FILE=/run/secrets/polymarket_api_key \
  -e POLYMARKET_API_SECRET_FILE=/run/secrets/polymarket_api_secret \
  -e POLYMARKET_API_PASSPHRASE_FILE=/run/secrets/polymarket_api_passphrase \
  -v "${ROOT_DIR}/.env:/app/.env:ro" \
  -v "${ROOT_DIR}/data/models:/app/data/models:ro" \
  -v "${ROOT_DIR}/data/artifacts:/app/data/artifacts" \
  -v "${ROOT_DIR}/logs:/app/logs" \
  -v "${ROOT_DIR}/secrets/polygon_private_key:/run/secrets/polygon_private_key:ro" \
  -v "${ROOT_DIR}/secrets/funder_address:/run/secrets/funder_address:ro" \
  -v "${ROOT_DIR}/secrets/polymarket_api_key:/run/secrets/polymarket_api_key:ro" \
  -v "${ROOT_DIR}/secrets/polymarket_api_secret:/run/secrets/polymarket_api_secret:ro" \
  -v "${ROOT_DIR}/secrets/polymarket_api_passphrase:/run/secrets/polymarket_api_passphrase:ro" \
  "${IMAGE_REF}"
