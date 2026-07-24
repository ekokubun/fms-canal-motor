# fms-canal-motor

Motor de canal endêmico (Gamma-Poisson, bootstrap sobre anos-base), sink Postgres
(`fms_prod`, schema `rio_claro`) e pipeline de faixas etárias/boletim, usados pelos 3
canais FMS de Rio Claro (UPA/`canal-endemico`, APS/`canal-aps`, SINAN/`canal-epidemico`).

Extraído em 2026-07 de 3 repos que mantinham cópias **byte-idênticas** destes mesmos
arquivos (`compute_channels.py`, `carga_postgres.py`, `pipeline.py`, ~8.400 linhas sem
nenhuma divergência real) — cada correção de bug precisava ser replicada manualmente 3×.
Este repo é agora a única fonte de verdade; os 3 repos FMS consomem via
`requirements-vps.txt` (`fms-canal-motor @ git+https://github.com/ekokubun/fms-canal-motor.git@<tag>`).

Ver `contrato_motor_canal.md` e `contrato_carga_postgres.md` em `pasta sem título/arquitetura/`
(outro repo) para o histórico completo da decisão do motor e da atomização.

## Uso

```bash
pip install -e ".[boletim,dev]"

# motor isolado
python -m fms_canal_motor.compute_channels input.csv --pop 210000 --output channel_data.json

# sink Postgres (produção — os 3 crons FMS usam este modo)
fms-carga-postgres --recompute --input input.csv --pop 210000 --fonte upa_187

# pipeline completo (dashboard estático + boletim)
fms-canal-pipeline input.csv --pop 210000 --output index.html
```

## Testes

```bash
pytest
```
