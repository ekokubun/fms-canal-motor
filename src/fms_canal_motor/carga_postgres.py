#!/usr/bin/env python3
"""
carga_postgres.py — Sink do método (canal endêmico) para o Postgres fms_prod.

Fase 4 da arquitetura EpiKinesis: o método Gamma-Poisson roda na VPS e grava o
resultado no Postgres (schema rio_claro), de onde o Metabase lê. NÃO reimplementa
o modelo — reaproveita compute_channels.run_pipeline + pipeline.step2/step3.

Modos
-----
  --recompute --input <csv> --pop <json|int>
        Roda o método de fato (CSV agregado → canais Gamma-Poisson + faixas
        etárias) e grava no Postgres. É o caminho de PRODUÇÃO (cron na VPS).

  --from-json [--json-dir DIR]
        Lê os JSON já computados (channel_data.json / age_channels.json /
        boletim_data.json) e gera as mesmas linhas. Caminho de verificação /
        fallback — não recomputa o modelo.

  --dry-run
        Não conecta ao Postgres: imprime contagem de linhas por tabela e
        amostras. Verificação local (não exige psycopg2 nem o banco).

Conexão (env, senha é secret na VPS)
------------------------------------
  PGHOST (default localhost), PGPORT (5432), PGDATABASE (fms_prod),
  PGUSER (epikinesis), PGPASSWORD

Exemplos
--------
  # verificação local, sem banco:
  python3 carga_postgres.py --from-json --dry-run

  # produção na VPS:
  python3 carga_postgres.py --recompute --input canal_endemico_input.csv \\
      --pop 210000 --schema rio_claro --fonte upa_187
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ── Repo do chamador (canal-endemico/canal-aps/canal-epidemico), não o
# diretório deste arquivo — desde a extração para pacote (fms-canal-motor)
# este módulo roda de dentro do site-packages, mas channel_state.json e os
# CSVs de entrada vivem no repo de onde o processo foi invocado (cwd; os 3
# crons já fazem `-w /app` com o repo montado em /app antes de chamar isto).
REPO_DIR = Path.cwd()

ZONAS_VALIDAS = {'sucesso', 'seguranca', 'alerta', 'epidemico', 'emergencia'}
ROMANOS = {'I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
           'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI','XXII'}

# ════════════════════════════════════════════════════════════════════════════
# Classificação do agravo (nome → tipo/cid/capítulo) — espelha as convenções
# de compute_channels.run_pipeline (capítulos romanos, "SINAN: ", "CID - desc").
# ════════════════════════════════════════════════════════════════════════════

def classify_agravo(nome):
    if nome == 'Todos os atendimentos':
        return ('todos', None, None)
    if nome.startswith('SINAN: '):
        return ('sinan', None, None)
    m = re.match(r'^([IVXLC]+)\s+-\s+', nome)
    if m and m.group(1) in ROMANOS:
        return ('capitulo', None, nome)
    m = re.match(r'^([A-Z]\d{2}(?:\.\d+)?)\b', nome)
    if m:
        return ('cid', m.group(1), None)
    return ('sindrome', None, None)


def _intq(q, i):
    try:
        return int(q[i])
    except (IndexError, TypeError, ValueError):
        return None


# ════════════════════════════════════════════════════════════════════════════
# Extração de linhas a partir dos dicts em memória / JSON
#   obs:   (nome, faixa, ano, se, casos)
#   canal: (nome, faixa, ano, se, p10,p25,p50,p75,p90, shape, rate)
#   clf:   (nome, faixa, ano, se, zona, exceedance)
# ════════════════════════════════════════════════════════════════════════════

def rows_from_channel_data(channel_data, obs, canal, clf, metas):
    """Canais principais (agregado, faixa='Todas'). channel_data = run_pipeline output."""
    for nome, ch in channel_data.get('channels', {}).items():
        metas.setdefault(nome, classify_agravo(nome))
        se_list = ch.get('se_list') or list(range(1, 53))

        # Observações: raw[i] = {'se':n, 'c2023':.., 'c2024':..}
        for entry in ch.get('raw', []):
            se = int(entry.get('se'))
            for k, v in entry.items():
                if k.startswith('c') and k[1:].isdigit():
                    obs.append((nome, 'Todas', int(k[1:]), se, int(v)))

        # Limiares por ano: channels[ano] = [[p10..p90] × 52]; params idem
        params = ch.get('params', {})
        for ano_s, arr in ch.get('channels', {}).items():
            ano = int(ano_s)
            pano = params.get(ano_s, [])
            for i, q in enumerate(arr):
                se = int(se_list[i]) if i < len(se_list) else i + 1
                pr = pano[i] if i < len(pano) else {}
                canal.append((nome, 'Todas', ano, se,
                              _intq(q, 0), _intq(q, 1), _intq(q, 2), _intq(q, 3), _intq(q, 4),
                              (pr or {}).get('shape'), (pr or {}).get('rate')))

        # Classificação + exceedance por ano (listas de 52)
        exc = ch.get('exceedance', {})
        for ano_s, zonas in ch.get('classifications', {}).items():
            ano = int(ano_s)
            eano = exc.get(ano_s, [])
            for i, z in enumerate(zonas):
                se = int(se_list[i]) if i < len(se_list) else i + 1
                e = eano[i] if i < len(eano) else None
                clf.append((nome, 'Todas', ano, se, z, e))


def rows_from_age_inmem(age_results, obs, canal, clf, metas):
    """Faixas etárias — formato em memória do pipeline.step3_age_channels.
    results[agravo][faixa] = {channels:{se:{p10..}}, raw:{ano:{se:n}}, classifications:{ano:{se:zona}}}
    """
    for agravo, faixas in age_results.items():
        metas.setdefault(agravo, classify_agravo(agravo))
        for faixa, d in faixas.items():
            raw = d.get('raw', {})                  # {ano: {se: n}}
            chans = d.get('channels', {})           # {se: {p10..p90}}  (1 conjunto)
            clfs = d.get('classifications', {})     # {ano: {se: zona}}
            anos = [int(a) for a in raw.keys()]

            for ano_s, ses in raw.items():
                ano = int(ano_s)
                for se_s, cnt in ses.items():
                    obs.append((agravo, faixa, ano, int(se_s), int(cnt)))

            # 1 conjunto de limiares → replicado para cada ano observado
            for ano in anos:
                for se_s, c in chans.items():
                    canal.append((agravo, faixa, ano, int(se_s),
                                  c.get('p10'), c.get('p25'), c.get('p50'),
                                  c.get('p75'), c.get('p90'), None, None))

            for ano_s, ses in clfs.items():
                ano = int(ano_s)
                for se_s, z in ses.items():
                    clf.append((agravo, faixa, ano, int(se_s), z, None))


def rows_from_age_compact(age_compact, obs, canal, metas):
    """Faixas etárias — formato compacto do age_channels.json (sem classificação).
    {agravo:{faixa:{years, se_list, channels:{ano:[[5q]×52]}, raw:[{cANO}×52]}}}
    """
    for agravo, faixas in age_compact.items():
        metas.setdefault(agravo, classify_agravo(agravo))
        for faixa, d in faixas.items():
            se_list = d.get('se_list') or list(range(1, 53))
            for i, entry in enumerate(d.get('raw', [])):
                se = int(se_list[i]) if i < len(se_list) else i + 1
                for k, v in entry.items():
                    if k.startswith('c') and k[1:].isdigit():
                        obs.append((agravo, faixa, int(k[1:]), se, int(v)))
            for ano_s, arr in d.get('channels', {}).items():
                ano = int(ano_s)
                for i, q in enumerate(arr):
                    se = int(se_list[i]) if i < len(se_list) else i + 1
                    canal.append((agravo, faixa, ano, se,
                                  _intq(q, 0), _intq(q, 1), _intq(q, 2), _intq(q, 3), _intq(q, 4),
                                  None, None))


# ════════════════════════════════════════════════════════════════════════════
# Obtenção dos dados — recompute (método) ou from-json (arquivos)
# ════════════════════════════════════════════════════════════════════════════

def obter_recompute(input_csv, pop, base_hist_years, skip_channel_estimation):
    """Roda o método de fato. Retorna (channel_data, age_results, prioridades)."""
    os.chdir(REPO_DIR)  # compute_channels/pipeline usam caminhos relativos
    from fms_canal_motor import compute_channels as cc
    from fms_canal_motor import pipeline as pl

    output_json = str(REPO_DIR / 'channel_data.json')  # estável → channel_state.json persiste
    print("[método] compute_channels.run_pipeline ...")
    channel_data = cc.run_pipeline(
        input_csv, pop, output_json,
        agravos='all',
        base_hist_years=base_hist_years,
        skip_channel_estimation=skip_channel_estimation,
    )

    print("[método] faixas etárias (step2/step3) ...")
    incremental = bool(skip_channel_estimation)
    age_data = pl.step2_age_group_data(input_csv, channel_data, incremental=incremental)
    age_results = pl.step3_age_channels(age_data, incremental=incremental) if age_data else {}

    # Prioridade (do boletim) — leve, sem docx
    prioridades = {}
    try:
        for b in pl.step4_boletim(channel_data):
            prioridades[b['name']] = b.get('prioridade')
    except Exception as e:
        print(f"  ⚠ prioridade do boletim indisponível: {e}")

    return channel_data, age_results, prioridades, False  # age em formato in-memory


def obter_from_json(json_dir):
    """Lê os JSON já computados. Retorna (channel_data, age_compact, prioridades)."""
    json_dir = Path(json_dir)
    with open(json_dir / 'channel_data.json', encoding='utf-8') as f:
        channel_data = json.load(f)

    age_compact = {}
    age_path = json_dir / 'age_channels.json'
    if age_path.exists():
        with open(age_path, encoding='utf-8') as f:
            age_compact = json.load(f)

    prioridades = {}
    bol_path = json_dir / 'boletim_data.json'
    if bol_path.exists():
        with open(bol_path, encoding='utf-8') as f:
            for b in json.load(f):
                prioridades[b.get('name')] = b.get('prioridade')

    return channel_data, age_compact, prioridades, True  # age em formato compacto


# ════════════════════════════════════════════════════════════════════════════
# Escrita no Postgres (transação atômica: limpa por fonte + insere)
# ════════════════════════════════════════════════════════════════════════════

def gravar_postgres(schema, fonte, metas, prioridades, obs, canal, clf, meta_exec):
    import psycopg2
    from psycopg2.extras import execute_values
    from psycopg2.extensions import register_adapter, adapt
    import numpy as np

    # psycopg2 não adapta tipos numpy (np.int64/np.float64) que vêm do recompute
    # em memória → registra conversão p/ int/float nativos. Sem isso, np.float64
    # vira o texto "np.float64(...)" no SQL → InvalidSchemaName: schema "np".
    for _t in (np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64):
        register_adapter(_t, lambda v: adapt(int(v)))
    for _t in (np.float16, np.float32, np.float64):
        register_adapter(_t, lambda v: adapt(float(v)))

    conn = psycopg2.connect(
        host=os.environ.get('PGHOST', 'localhost'),
        port=os.environ.get('PGPORT', '5432'),
        dbname=os.environ.get('PGDATABASE', 'fms_prod'),
        user=os.environ.get('PGUSER', 'epikinesis'),
        password=os.environ.get('PGPASSWORD'),
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO {schema}')

                # 1. Upsert dim_agravo → mapa nome→id
                dim_vals = [(nome, t, cid, cap, prioridades.get(nome))
                            for nome, (t, cid, cap) in metas.items()]
                execute_values(cur, """
                    INSERT INTO dim_agravo (nome, tipo, cid_codigo, capitulo, prioridade)
                    VALUES %s
                    ON CONFLICT (nome) DO UPDATE SET
                        tipo       = EXCLUDED.tipo,
                        cid_codigo = EXCLUDED.cid_codigo,
                        capitulo   = EXCLUDED.capitulo,
                        prioridade = COALESCE(EXCLUDED.prioridade, dim_agravo.prioridade)
                    RETURNING agravo_id, nome
                """, dim_vals)
                nome2id = {nome: aid for aid, nome in cur.fetchall()}

                # 2. Limpar dados desta fonte (swap atômico)
                for tbl in ('classificacao_se', 'canal_endemico', 'fato_observacao_se'):
                    cur.execute(f'DELETE FROM {tbl} WHERE fonte = %s', (fonte,))

                # 3. Inserir fatos
                obs_v = [(nome2id[n], fx, fonte, a, s, c) for (n, fx, a, s, c) in obs]
                execute_values(cur, """
                    INSERT INTO fato_observacao_se
                        (agravo_id, faixa_etaria, fonte, ano, se, casos) VALUES %s
                """, obs_v, page_size=2000)

                canal_v = [(nome2id[n], fx, fonte, a, s, p10, p25, p50, p75, p90, sh, rt)
                           for (n, fx, a, s, p10, p25, p50, p75, p90, sh, rt) in canal]
                execute_values(cur, """
                    INSERT INTO canal_endemico
                        (agravo_id, faixa_etaria, fonte, ano, se, p10,p25,p50,p75,p90, shape, rate)
                    VALUES %s
                """, canal_v, page_size=2000)

                clf_v = [(nome2id[n], fx, fonte, a, s, z, e)
                         for (n, fx, a, s, z, e) in clf if z in ZONAS_VALIDAS]
                execute_values(cur, """
                    INSERT INTO classificacao_se
                        (agravo_id, faixa_etaria, fonte, ano, se, zona, exceedance) VALUES %s
                """, clf_v, page_size=2000)

                # 4. Auditoria
                cur.execute("""
                    INSERT INTO execucao
                        (fonte, modelo, mc_samples, base_hist_years, se_atual,
                         fonte_csv, n_agravos, modo)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (fonte, meta_exec.get('model'), meta_exec.get('mc_samples'),
                      meta_exec.get('base_hist_years'), meta_exec.get('se_atual'),
                      meta_exec.get('source'), len(metas), meta_exec.get('modo')))
    finally:
        conn.close()
    print(f"✓ Postgres atualizado: schema={schema} fonte={fonte} | "
          f"{len(metas)} agravos, {len(obs)} obs, {len(canal)} canal, {len(clf)} clf")


# ════════════════════════════════════════════════════════════════════════════
# Verificação (dry-run): contagens + amostras, sem banco
# ════════════════════════════════════════════════════════════════════════════

def resumo_dry_run(metas, obs, canal, clf):
    print("\n" + "=" * 60)
    print("DRY-RUN — nenhuma conexão com o Postgres")
    print("=" * 60)
    print(f"  dim_agravo .........: {len(metas):>7} agravos")
    print(f"  fato_observacao_se .: {len(obs):>7} linhas")
    print(f"  canal_endemico .....: {len(canal):>7} linhas")
    print(f"  classificacao_se ...: {len(clf):>7} linhas")

    faixas = sorted({r[1] for r in obs})
    anos = sorted({r[2] for r in obs})
    tipos = {}
    for _, (t, _, _) in metas.items():
        tipos[t] = tipos.get(t, 0) + 1
    print(f"  faixas .............: {faixas}")
    print(f"  anos ...............: {anos}")
    print(f"  tipos de agravo ....: {tipos}")

    zonas = {r[4] for r in clf}
    invalidas = zonas - ZONAS_VALIDAS
    print(f"  zonas presentes ....: {sorted(zonas)}")
    if invalidas:
        print(f"  ⚠ ZONAS INVÁLIDAS (seriam rejeitadas pelo CHECK): {invalidas}")
    else:
        print(f"  ✓ todas as zonas válidas")

    # Amostra: monta uma linha da vw_canal_completo para 1 agravo conhecido
    alvo = next((n for n in ('SINAN: Dengue', 'Todos os atendimentos') if n in metas),
                next(iter(metas), None))
    if alvo:
        canal_idx = {(n, fx, a, s): (p10,p25,p50,p75,p90)
                     for (n, fx, a, s, p10,p25,p50,p75,p90, _, _) in canal}
        print(f"\n  Amostra vw_canal_completo — '{alvo}' / Todas / 2026 (SE 18–21):")
        print(f"    {'SE':>3} {'obs':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5}  zona")
        for (n, fx, a, s, c) in sorted(obs):
            if n == alvo and fx == 'Todas' and a == 2026 and 18 <= s <= 21:
                q = canal_idx.get((n, fx, a, s))
                if q:
                    p10,p25,p50,p75,p90 = q
                    zona = ('sucesso' if c <= (p25 or 0) else 'seguranca' if c <= (p50 or 0)
                            else 'alerta' if c <= (p75 or 0) else 'epidemico' if c <= (p90 or 0)
                            else 'emergencia')
                    print(f"    {s:>3} {c:>5} {p25:>5} {p50:>5} {p75:>5} {p90:>5}  {zona}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Carga do canal endêmico no Postgres fms_prod")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument('--recompute', action='store_true',
                      help='Roda o método a partir do CSV (produção)')
    modo.add_argument('--from-json', action='store_true',
                      help='Lê os JSON já computados (verificação/fallback)')
    ap.add_argument('--input', help='CSV agregado de entrada (modo --recompute)')
    ap.add_argument('--pop', default='210000', help='População (JSON dict ou int)')
    ap.add_argument('--json-dir', default=str(REPO_DIR),
                    help='Diretório dos JSON (modo --from-json)')
    ap.add_argument('--base-hist-years', default=None,
                    help='Anos de base separados por vírgula (ex: 2023,2024,2025)')
    ap.add_argument('--skip-channel-estimation', action='store_true',
                    help='Modo incremental: reaproveita channel_state.json (congela limiares)')
    ap.add_argument('--schema', default='rio_claro')
    ap.add_argument('--fonte', default='upa_187')
    ap.add_argument('--dry-run', action='store_true',
                    help='Não conecta ao Postgres; imprime contagens e amostras')
    args = ap.parse_args()

    # Inferir modo
    recompute = args.recompute or (bool(args.input) and not args.from_json)
    if recompute and not args.input:
        ap.error("--recompute exige --input <csv>")

    # pop: JSON ou int
    try:
        pop = json.loads(args.pop)
        if isinstance(pop, float):
            pop = int(pop)
    except json.JSONDecodeError:
        pop = int(args.pop)

    bhy = [int(y) for y in args.base_hist_years.split(',')] if args.base_hist_years else None

    # ── Obter dados ─────────────────────────────────────────────────────────
    if recompute:
        channel_data, age, prioridades, age_compact = obter_recompute(
            str(Path(args.input).resolve()), pop, bhy, args.skip_channel_estimation)
        modo_str = 'recompute'
    else:
        channel_data, age, prioridades, age_compact = obter_from_json(args.json_dir)
        modo_str = 'from-json'

    # ── Transformar em linhas ──────────────────────────────────────────────
    metas, obs, canal, clf = {}, [], [], []
    rows_from_channel_data(channel_data, obs, canal, clf, metas)
    if age:
        if age_compact:
            rows_from_age_compact(age, obs, canal, metas)
        else:
            rows_from_age_inmem(age, obs, canal, clf, metas)

    meta = dict(channel_data.get('metadata', {}))
    meta['modo'] = modo_str

    # ── Emitir ──────────────────────────────────────────────────────────────
    if args.dry_run:
        resumo_dry_run(metas, obs, canal, clf)
        print(f"\n  (modo={modo_str}; nada gravado)")
        return

    gravar_postgres(args.schema, args.fonte, metas, prioridades, obs, canal, clf, meta)


if __name__ == '__main__':
    main()
