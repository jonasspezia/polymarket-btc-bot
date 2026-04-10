# Deploy em Docker Swarm

Este pacote adiciona um caminho de deploy em contêiner para o `Ishant5436/polymarket-btc-bot` sem remover o fluxo `systemd` original do upstream.

## Objetivo

- Eliminar a dependência do host em Python `3.12`, `libomp` e wheel nativa do LightGBM.
- Rodar no servidor atual via Docker Swarm.
- Manter segredos fora da imagem.
- Persistir artefatos, logs e modelo treinado fora do contêiner.

## Estrutura esperada no servidor

No diretório raiz do repositório:

```text
.env
data/
  models/
    lgbm_btc_5m.txt
  raw/
  processed/
  artifacts/
logs/
secrets/
  polygon_private_key
  funder_address
  polymarket_api_key
  polymarket_api_secret
  polymarket_api_passphrase
```

## Arquivos obrigatórios

- `.env`
- `data/models/lgbm_btc_5m.txt`
- `secrets/polygon_private_key`
- `secrets/funder_address`
- `secrets/polymarket_api_key`
- `secrets/polymarket_api_secret`
- `secrets/polymarket_api_passphrase`

## Modos suportados

- `BOT_MODE=validation`
  - sobe o engine em `--validation-only`
  - melhor primeiro rollout no servidor
- `BOT_MODE=dry-run`
  - sobe o engine em `--dry-run`
  - não exige segredos do CLOB, mas ainda precisa do modelo
- `BOT_MODE=live`
  - sobe o engine em `--live`
  - exige `.env` coerente, inclusive `LIVE_TRADING_ENABLED=true`
- `BOT_MODE=preflight`
  - uso one-shot via `deploy/run_preflight.sh`

## Primeiro uso recomendado

1. Copie `.env.example` para `.env` e ajuste os parâmetros não sensíveis.
2. Coloque o modelo treinado em `data/models/lgbm_btc_5m.txt`.
3. Grave cada segredo em um arquivo separado dentro de `secrets/`.
4. Rode o preflight:

```bash
./deploy/run_preflight.sh
```

5. Se o preflight passar, faça o rollout em modo de validação:

```bash
BOT_MODE=validation ./deploy/deploy_swarm.sh
```

6. Acompanhe:

```bash
docker service logs -f polymarket-btc-bot-ishant5436_polymarket-btc-bot
```

## Promoção para live

Só promova para `live` quando:

- o preflight passar sem bloqueios;
- o serviço em `validation` estiver lendo mercado e conta corretamente;
- `.env` tiver `LIVE_TRADING_ENABLED=true`;
- você confirmar que o modelo e os thresholds são os desejados para produção;
- o runtime estiver dentro das travas mínimas de governança abaixo.

## Guardrails mínimos para live

Se qualquer um desses limites for violado, o engine agora falha fechado na
subida em produção:

- `MIN_EDGE >= 0.03`
- `MIN_SIDE_PROBABILITY >= 0.55`
- `MAX_ENTRY_PRICE <= 0.70`
- `MAX_SPREAD <= 0.20`
- `MAX_OPEN_POSITIONS = 1`
- `BANKROLL_FRACTION_PER_ORDER <= 0.25`
- `MIN_TIME_REMAINING_SECONDS >= 60`
- `TIME_DECAY_EXIT_SECONDS <= 300`
- `MIN_AVAILABLE_COLLATERAL >= 10`
- `MAX_AVAILABLE_COLLATERAL_DRAWDOWN > 0` e `<= 5`

Com isso:

```bash
BOT_MODE=live ./deploy/deploy_swarm.sh
```

## Fluxo operacional recomendado

Depois de qualquer alteração em estratégia, risco ou discovery:

1. Validar localmente:

```bash
pytest -q
```

2. Publicar no GitHub:

```bash
git push origin main
```

3. Acompanhar GitHub Actions:

```bash
gh run watch --exit-status
```

4. Fazer rollout no Swarm:

```bash
BOT_MODE=validation ./deploy/deploy_swarm.sh
BOT_MODE=live ./deploy/deploy_swarm.sh
```

5. Acompanhar o container e o soak inicial:

```bash
docker service logs -f polymarket-btc-bot-ishant5436_polymarket-btc-bot
docker service ps polymarket-btc-bot-ishant5436_polymarket-btc-bot --no-trunc
```

Recomendação prática: depois do deploy, observar por alguns ciclos de discovery
e pelo menos um relatório periódico de status antes de considerar o rollout
encerrado.

Hoje o workflow `CI` do repositório cobre:

- `pytest -q`
- `docker build -t polymarket-btc-bot-ci .`

O rollout de produção segue manual no Swarm, porque o host precisa do modelo
treinado e dos segredos fora do GitHub.

## Observações

- O modelo **não** vem no repositório upstream. O deploy falha fechado se `data/models/lgbm_btc_5m.txt` não existir.
- Monte o diretório `data/models` inteiro no contêiner. Assim o runtime enxerga também sidecars opcionais como `lgbm_btc_5m.metadata.json` e `lgbm_btc_5m.calibrator.pkl`.
- O `entrypoint` aceita segredos via padrão `*_FILE`, então o stack usa Docker secrets e exporta as variáveis antes de iniciar o Python.
- O healthcheck não depende de porta HTTP; ele valida modelo, processo principal e o último manifesto de execução.
- O stack fixa o serviço no nó manager. Se você quiser espalhar em vários nós, publique a imagem em registry em vez de depender de tag local.
