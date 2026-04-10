#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_NAME="${STACK_NAME:-polymarket-btc-bot-ishant5436}"
IMAGE_REF="${IMAGE_REF:-polymarket-btc-bot-ishant5436:local}"
BOT_MODE="${BOT_MODE:-validation}"
GENERATED_STACK="${ROOT_DIR}/deploy/swarm-stack.generated.yml"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}

trap cleanup EXIT

mkdir -p \
  "${ROOT_DIR}/data/models" \
  "${ROOT_DIR}/data/raw" \
  "${ROOT_DIR}/data/processed" \
  "${ROOT_DIR}/data/artifacts" \
  "${ROOT_DIR}/logs" \
  "${ROOT_DIR}/secrets"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERRO: arquivo obrigatório ausente: ${path}" >&2
    exit 64
  fi
}

require_file "${ROOT_DIR}/.env"
require_file "${ROOT_DIR}/data/models/lgbm_btc_5m.txt"

if [[ "${BOT_MODE}" == "validation" || "${BOT_MODE}" == "live" ]]; then
  require_file "${ROOT_DIR}/secrets/polygon_private_key"
  require_file "${ROOT_DIR}/secrets/funder_address"
  require_file "${ROOT_DIR}/secrets/polymarket_api_key"
  require_file "${ROOT_DIR}/secrets/polymarket_api_secret"
  require_file "${ROOT_DIR}/secrets/polymarket_api_passphrase"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERRO: docker não encontrado no host." >&2
  exit 127
fi

SECRETS_ENV_SNIPPET="${TMP_DIR}/secrets_env.yml"
SECRETS_SERVICE_SNIPPET="${TMP_DIR}/secrets_service.yml"
SECRETS_TOPLEVEL_SNIPPET="${TMP_DIR}/secrets_toplevel.yml"

if [[ "${BOT_MODE}" == "validation" || "${BOT_MODE}" == "live" ]]; then
  cat > "${SECRETS_ENV_SNIPPET}" <<'EOF'
      POLYGON_PRIVATE_KEY_FILE: /run/secrets/polygon_private_key
      FUNDER_ADDRESS_FILE: /run/secrets/funder_address
      POLYMARKET_API_KEY_FILE: /run/secrets/polymarket_api_key
      POLYMARKET_API_SECRET_FILE: /run/secrets/polymarket_api_secret
      POLYMARKET_API_PASSPHRASE_FILE: /run/secrets/polymarket_api_passphrase
EOF

  cat > "${SECRETS_SERVICE_SNIPPET}" <<'EOF'
    secrets:
      - polygon_private_key
      - funder_address
      - polymarket_api_key
      - polymarket_api_secret
      - polymarket_api_passphrase
EOF

  cat > "${SECRETS_TOPLEVEL_SNIPPET}" <<EOF
secrets:
  polygon_private_key:
    file: ${ROOT_DIR}/secrets/polygon_private_key
  funder_address:
    file: ${ROOT_DIR}/secrets/funder_address
  polymarket_api_key:
    file: ${ROOT_DIR}/secrets/polymarket_api_key
  polymarket_api_secret:
    file: ${ROOT_DIR}/secrets/polymarket_api_secret
  polymarket_api_passphrase:
    file: ${ROOT_DIR}/secrets/polymarket_api_passphrase
EOF
else
  : > "${SECRETS_ENV_SNIPPET}"
  : > "${SECRETS_SERVICE_SNIPPET}"
  : > "${SECRETS_TOPLEVEL_SNIPPET}"
fi

echo "==> Building image ${IMAGE_REF}"
docker build -t "${IMAGE_REF}" "${ROOT_DIR}"

echo "==> Rendering stack file ${GENERATED_STACK}"
sed \
  -e "s#__APP_ROOT__#${ROOT_DIR}#g" \
  -e "s#__IMAGE_REF__#${IMAGE_REF}#g" \
  -e "s#__BOT_MODE__#${BOT_MODE}#g" \
  "${ROOT_DIR}/deploy/swarm-stack.yml.tmpl" \
  | sed \
      -e "/__SECRETS_ENV__/r ${SECRETS_ENV_SNIPPET}" \
      -e "/__SECRETS_ENV__/d" \
      -e "/__SECRETS_SERVICE__/r ${SECRETS_SERVICE_SNIPPET}" \
      -e "/__SECRETS_SERVICE__/d" \
      -e "/__SECRETS_TOPLEVEL__/r ${SECRETS_TOPLEVEL_SNIPPET}" \
      -e "/__SECRETS_TOPLEVEL__/d" \
  > "${GENERATED_STACK}"

echo "==> Deploying stack ${STACK_NAME}"
docker stack deploy -c "${GENERATED_STACK}" "${STACK_NAME}"

echo
echo "Deploy concluído."
echo "Serviço: ${STACK_NAME}_polymarket-btc-bot"
echo "Logs: docker service logs -f ${STACK_NAME}_polymarket-btc-bot"
