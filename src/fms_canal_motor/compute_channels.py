#!/usr/bin/env python3
"""
Canal Endêmico Bayesiano Hierárquico — Gamma-Poisson
=====================================================

Modelo:
    X_si | λ_si ~ Poisson(λ_si · e_i)
    λ_si ~ Gamma(a_s, b_s)

    Marginalização → X_si ~ BinNeg(a_s, p_s)  com  p_s = b_s/(b_s + e_i)

MOTOR (decisão 2026-07-19, atomização — ver contrato_motor_canal.md seção 4b): bootstrap de
casos sobre os anos-base, reajuste por Método dos Momentos a cada reamostragem, percentil da
MISTURA das distribuições preditivas via CDF exata (scipy.stats.nbinom) — substitui o par
MLE-por-grid-search + Monte Carlo. Motivo: nas SEs de baixa variância entre anos-base, o
parâmetro de dispersão não é identificável de forma confiável por ajuste pontual único (grid
OU MLE livre) — a verossimilhança fica quase plana ali. O bootstrap não resolve a
não-identificabilidade de fundo, mas evita depender de onde um único otimizador pousa.
`estimate_params_mle`/`estimate_params_mom`/`mc_quantiles` abaixo continuam no arquivo como
referência histórica e para comparação, mas não são mais o caminho usado por
`compute_endemic_channel()` — ver `_bootstrap_channel_se()`.

Saída: JSON compacto para dashboard React/Recharts.

Autor: Pipeline epidemiológico UPAs Rio Claro
"""

import json
import sys
import argparse
import warnings
from math import lgamma, log, exp, inf
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom as _scipy_nbinom
from scipy.stats import betabinom as _scipy_betabinom


class NumpyEncoder(json.JSONEncoder):
    """Converte tipos numpy para tipos nativos Python (JSON serializável)."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Constantes ────────────────────────────────────────────────────────
MC_SAMPLES = 500_000
QUANTILES  = [0.10, 0.25, 0.50, 0.75, 0.90]
ZONE_NAMES = ['sucesso', 'seguranca', 'alerta', 'epidemico', 'emergencia']
RNG_SEED   = 2026
MAX_SE     = 52          # SE 53 excluída por padrão (poucos dados)
ATEND_COL  = 'atend_id'  # nível-atendimento: contar atendimentos, não linhas de CID
JANELA_SE  = 2           # meia-janela de SE vizinhas usada na estimação (0 = SE isolada)
import os as _os
USAR_PROPORCAO = _os.getenv('CANAL_PROPORCAO', '1') != '0'   # canal de agravo como fração

# Agravos SEM leitura epidemiológica: a série é real, mas o que ela mede é
# produção administrativa ou registro de doença crônica, não risco de surto.
# Alarmavam de 21 a 33 semanas em 33 — cerca de 300 dos 1.080 alarmes da APS —
# porque o que subiu neles foi a prática de codificação. Continuam sendo
# calculados e publicados como SÉRIE (útil para gestão e para vigiar a própria
# codificação), mas não recebem classificação de zona epidêmica.
# "Todos os atendimentos" entra aqui por decisão de 2026-08-28: é indicador de
# DEMANDA, não de epidemia.
import re as _re
_SEM_ZONA_RE = _re.compile(r'^(Z\d|E1[014]\b|E78|Z34|Z00|Z10|Z76)')

def sem_zona_epidemica(nome: str) -> bool:
    return (nome == 'Todos os atendimentos'
            or nome.startswith('XXI - ')
            or bool(_SEM_ZONA_RE.match(nome)))


def _betabinom_channel_se(k_train, n_train, n_mon):
    """Faixa preditiva para a PROPORÇÃO do agravo no total de atendimentos da SE.

    Prior beta ajustada por momentos sobre as proporções dos anos-base (a
    sobredispersão entre anos vira a concentração da beta); preditiva
    beta-binomial com o denominador observado na semana-alvo.

    É o que torna o canal imune, por construção, a qualquer deriva de VOLUME —
    codificação, unidade que entra ou sai, captura etária, demanda real. Se o
    total da semana sobe 30% sem mudar a doença, a contagem do agravo sobe 30%
    junto e a fração não se mexe.
    """
    k = np.asarray(k_train, dtype=float)
    n = np.asarray(n_train, dtype=float)
    n_mon = int(round(float(n_mon)))
    if n.sum() <= 0 or n_mon <= 0:
        return [0.0] * 5, 0.0, 0.0
    ph = k.sum() / n.sum()
    if ph <= 0:
        return [0.0] * 5, 0.0, 0.0
    ph = min(ph, 1 - 1e-9)
    pi = k / np.maximum(n, 1.0)
    v_obs = float(pi.var(ddof=1)) if len(pi) > 1 else 0.0
    v_bin = float(np.mean(ph * (1 - ph) / np.maximum(n, 1.0)))
    extra = max(v_obs - v_bin, 1e-12)                 # sobredispersão entre anos
    s_tot = float(np.clip(ph * (1 - ph) / extra - 1.0, 0.5, 1e6))
    a, b = ph * s_tot, (1 - ph) * s_tot
    qs = [float(_scipy_betabinom.ppf(q, n_mon, a, b)) for q in QUANTILES]
    return qs, a, b
FALLBACK_SHAPE = 0.1     # para SE com todos os anos = 0
FALLBACK_RATE  = 1.0
# Anos de implantação — excluídos permanentemente do cálculo dos canais
EXCLUDED_YEARS  = [2021, 2022]
# Base histórica: últimos 5 anos completos, excluindo anos de implantação
# Calculado automaticamente — não editar manualmente
_ano_atual      = __import__('datetime').date.today().year
BASE_HIST_YEARS = [y for y in range(_ano_atual - 5, _ano_atual) if y not in EXCLUDED_YEARS]

# ── Funções auxiliares ────────────────────────────────────────────────

def epi_week(date):
    """Semana epidemiológica MS/OMS (domingo–sábado).
    Retorna (ano_epi, se).
    """
    from datetime import timedelta
    # Ajusta para domingo = dia 0 da semana
    dow = date.isoweekday() % 7  # dom=0, seg=1, ..., sab=6
    # Início da semana epi (domingo)
    sun = date - timedelta(days=dow)
    # Dia de referência: quarta-feira da mesma SE
    wed = sun + timedelta(days=3)
    ano_epi = wed.year
    # Primeiro domingo do ano epi
    jan1 = pd.Timestamp(ano_epi, 1, 1)
    jan1_dow = jan1.isoweekday() % 7
    if jan1_dow <= 3:
        first_sun = jan1 - timedelta(days=jan1_dow)
    else:
        first_sun = jan1 + timedelta(days=7 - jan1_dow)
    se = (sun - first_sun).days // 7 + 1
    if se < 1:
        return epi_week(date - timedelta(days=7))
    return ano_epi, se


def nb_loglik(x_arr, shape, rate, exposure):
    """Log-verossimilhança da Binomial Negativa marginal.

    X ~ NB(r, p) com r=shape, p=rate/(rate+exposure).
    """
    r = shape
    p = rate / (rate + exposure)
    ll = 0.0
    for x in x_arr:
        x = int(x)
        ll += lgamma(x + r) - lgamma(r) - lgamma(x + 1)
        ll += r * log(p) + x * log(1 - p + 1e-300)
    return ll


def estimate_params_mom(cases_arr, exposures_arr):
    """Método dos Momentos para (shape, rate) da Gamma prior.

    Se x_i ~ NB(a, b/(b+e_i)):
        E[X_i] = a * e_i / b
        Var[X_i] = a * e_i / b * (1 + e_i / b)

    Com exposição constante (simplificação):
        mean ≈ a * e_bar / b
        var  ≈ mean * (1 + e_bar / b)
    """
    x = np.array(cases_arr, dtype=float)
    e = np.array(exposures_arr, dtype=float)

    # Taxa observada por 100k
    rates = x / np.maximum(e, 1e-10)
    m = np.mean(rates)
    v = np.var(rates, ddof=1) if len(rates) > 1 else m + 1

    if m <= 0 or v <= 0:
        return FALLBACK_SHAPE, FALLBACK_RATE

    # MoM: shape = m^2 / (v - m), rate = m / (v - m)
    # Mas var da taxa = a/b^2, mean da taxa = a/b
    # → b = m / (v - 0), a = m * b ...
    # Para Gamma(a,b) com E=a/b, Var=a/b^2:
    #   a = m^2 / v,  b = m / v
    if v < 1e-10:
        v = m  # Poisson assumption

    b_hat = m / v
    a_hat = m * b_hat

    # Clamp
    a_hat = max(a_hat, 0.01)
    b_hat = max(b_hat, 0.001)

    return a_hat, b_hat


def estimate_params_mle(cases_arr, exposures_arr, a0=None, b0=None):
    """MLE via grid refinement em torno do MoM.

    Sem scipy, fazemos busca em grade 2D em log-space.
    """
    x = np.array(cases_arr, dtype=float)
    e = np.array(exposures_arr, dtype=float)

    if a0 is None or b0 is None:
        a0, b0 = estimate_params_mom(cases_arr, exposures_arr)

    best_a, best_b = a0, b0
    best_ll = -inf

    # Busca em 3 escalas
    for scale in [2.0, 0.5, 0.1]:
        a_grid = np.exp(np.linspace(
            log(max(best_a * exp(-scale), 0.001)),
            log(best_a * exp(scale)),
            21
        ))
        b_grid = np.exp(np.linspace(
            log(max(best_b * exp(-scale), 0.0001)),
            log(best_b * exp(scale)),
            21
        ))

        for a in a_grid:
            for b in b_grid:
                # Usar exposição média para simplificar
                e_mean = np.mean(e)
                ll = nb_loglik(x, a, b, e_mean)
                if ll > best_ll:
                    best_ll = ll
                    best_a, best_b = a, b

    return best_a, best_b


def mc_quantiles(shape, rate, exposure, quantiles=QUANTILES, n_samples=MC_SAMPLES, rng=None):
    """Quantis preditivos via Monte Carlo Gamma-Poisson."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    # λ ~ Gamma(shape, 1/rate)
    lam = rng.gamma(shape, 1.0 / rate, size=n_samples)
    # X ~ Poisson(λ * exposure)
    x = rng.poisson(lam * exposure)

    return [int(np.quantile(x, q)) for q in quantiles]


def _bootstrap_channel_se(cases_train, exp_train, e_mon, n_boot=300, rng=None,
                           quantiles=QUANTILES):
    """
    MOTOR ÚNICO (decisão 2026-07-19) — bootstrap sobre os anos-base em espaço de TAXA
    (contagem/exposição), reajuste por Método dos Momentos a cada reamostragem, percentil
    da MISTURA das distribuições preditivas via CDF exata (não simulação — sem ruído duplo
    de bootstrap + Monte Carlo).

    cases_train / exp_train: contagens e exposições dos anos-base (mesma ordem, mesmo tamanho).
    e_mon: exposição do ano monitorado — usada só na predição final, pode diferir da exposição
    de ajuste (ex.: leave-one-out ou população mudou).

    Retorna (qs, a_central, b_central): qs = [p10,p25,p50,p75,p90]; a_central/b_central são
    um ajuste MoM único (não-bootstrap) só para registro em `params` — não são usados para
    calcular `qs`, mantidos por compatibilidade de schema com versões anteriores.

    ATENÇÃO: dois bugs reais já foram encontrados e corrigidos nesta lógica durante o
    desenvolvimento (ver arquitetura/prototipo_motor_canal/RESULTADOS.md seção 4) — não
    "simplificar de volta" sem reler aquele documento:
    (1) o MoM tem que ser feito em espaço de TAXA (contagem/exposição), não contagem bruta —
        do contrário o fallback de variância degenerada perde a escala da exposição.
    (2) `a_hat`/`b_hat` NUNCA podem ser clipados independentemente depois de calculados — só
        clipar `b_hat` e derivar `a_hat = m * b_hat` a partir dele, preservando a razão que
        trava a média da NB na média observada.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    cases = np.asarray(cases_train, dtype=float)
    exp_arr = np.asarray(exp_train, dtype=float)
    n_years = len(cases)
    rates_obs = cases / np.maximum(exp_arr, 1e-10)

    if cases.sum() == 0:
        return [0] * len(quantiles), FALLBACK_SHAPE, FALLBACK_RATE

    idx = rng.integers(0, n_years, size=(n_boot, n_years))
    resample_rates = rates_obs[idx]  # (n_boot, n_years)
    m = resample_rates.mean(axis=1)
    v = resample_rates.var(axis=1, ddof=1) if n_years > 1 else np.zeros(n_boot)

    # reamostra degenerada (sorteou o mesmo ano n_years vezes -> var=0) não informa nada sobre
    # dispersão — descartar é mais honesto que fabricar uma variância de fallback.
    valid = v > 1e-12
    if valid.any():
        m, v = m[valid], v[valid]
    else:
        rate_mean = rates_obs.mean()
        m = np.array([rate_mean])
        v = np.array([max(rate_mean, 1e-9)])

    # clipar b_hat ANTES de derivar a_hat = m*b_hat — nunca os dois independentemente
    # depois (isso já causou P90 = exposição inteira num teste real).
    b_hat = np.clip(m / v, 1e-6, 1e6)
    a_hat = m * b_hat
    p_hat = b_hat / (b_hat + e_mon)

    def cdf_mix(x):
        return _scipy_nbinom.cdf(x, a_hat, p_hat).mean()

    qs = []
    for q in quantiles:
        hi = max(int(cases.max()), 10)
        while cdf_mix(hi) < q:
            hi *= 2
            if hi > 10_000_000:  # guarda-corpo: não deveria ocorrer com dados reais
                break
        lo = 0
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf_mix(mid) < q:
                lo = mid + 1
            else:
                hi = mid
        qs.append(int(lo))

    # ajuste central (não-bootstrap) só para o campo `params` do JSON — não usado em `qs`.
    # Mesma convenção de estimate_params_mom() (b=m/v, a=m*b) — não a "v-m" de Gamma
    # clássica, pra ficar consistente com o resto do arquivo.
    m_c = rates_obs.mean()
    v_c = rates_obs.var(ddof=1) if n_years > 1 else m_c + 1
    if m_c <= 0 or v_c <= 0:
        a_c, b_c = FALLBACK_SHAPE, FALLBACK_RATE
    else:
        if v_c < 1e-10:
            v_c = m_c
        b_c = m_c / v_c
        a_c = m_c * b_c
        a_c, b_c = max(a_c, 0.01), max(b_c, 0.001)

    return qs, a_c, b_c


def classify_zone(value, thresholds):
    """Classifica valor em zona epidêmica.
    thresholds = [p10, p25, p50, p75, p90]
    """
    p10, p25, p50, p75, p90 = thresholds
    if value <= p25:
        return 'sucesso'
    elif value <= p50:
        return 'seguranca'
    elif value <= p75:
        return 'alerta'
    elif value <= p90:
        return 'epidemico'
    else:
        return 'emergencia'


# ── Detecção de SE incompleta ─────────────────────────────────────────

def detectar_se_incompleta(df, col_se, col_ano, col_casos, col_data=None):
    """Detecta se a última SE do ano mais recente está incompleta."""
    # Guarda: agravo sem dados utilizáveis (ex.: só SE > MAX_SE, já removidas, ou
    # coluna de ano toda NaN) → nada a detectar. Antes estourava com
    # ValueError: cannot convert float NaN to integer no int(.max()).
    if df.empty or pd.isna(df[col_ano].max()) or pd.isna(df[col_se].max()):
        return {'ultima_se': 0, 'ano': 0, 'completa': True,
                'dias': 0, 'ratio': 0.0, 'decisao': 'INCLUIR'}
    ultimo_ano = int(df[col_ano].max())
    df_ua = df[df[col_ano] == ultimo_ano]
    ultima_se = int(df_ua[col_se].max())
    df_use = df_ua[df_ua[col_se] == ultima_se]

    vol_ultima = df_use[col_casos].sum()
    se_ant = ultima_se - 1
    df_ant = df_ua[df_ua[col_se] == se_ant]
    vol_ant = df_ant[col_casos].sum() if len(df_ant) > 0 and se_ant >= 1 else vol_ultima
    ratio = vol_ultima / max(vol_ant, 1)

    criterios_ok = 0
    dias = 7  # default se não temos data individual
    dia_sem = 7

    if col_data and col_data in df.columns:
        dias = df_use[col_data].dt.date.nunique()
        dia_sem = pd.to_datetime(df_use[col_data]).max().isoweekday()
        if dias >= 6:
            criterios_ok += 1
        if dia_sem >= 5:
            criterios_ok += 1
    else:
        criterios_ok += 2  # sem data individual, assume OK para 2 critérios

    if ratio >= 0.50:
        criterios_ok += 1

    completa = criterios_ok >= 2
    return {
        'ultima_se': ultima_se,
        'ano': ultimo_ano,
        'completa': completa,
        'dias': dias,
        'ratio': round(ratio, 2),
        'decisao': 'INCLUIR' if completa else 'EXCLUIR'
    }


# ── Pipeline principal ────────────────────────────────────────────────

def compute_endemic_channel(
    df_agg,
    populations,
    agravo_name="Todos",
    leave_one_out=False,
    base_hist_years=None,
    use_mle=True,
    monitor_year=None
,
        denominadores=None):
    """
    Computa canal endêmico para um agravo.

    Parâmetros:
        df_agg: DataFrame com colunas [ano, se, casos]
                Já agregado (uma linha por ano×SE).
        populations: dict {ano: população}
        agravo_name: nome do agravo (para metadados)
        leave_one_out: se True, exclui o ano monitorado da estimação
        use_mle: VESTIGIAL desde 2026-07-19 (motor único = bootstrap, ver
                 _bootstrap_channel_se) — mantido só por compatibilidade de assinatura;
                 os dois call sites do arquivo sempre passam True, nunca era efetivamente
                 alternável em produção.
        monitor_year: ano específico a monitorar (None = todos)

    Retorna:
        dict com estrutura para JSON do dashboard
    """
    rng = np.random.default_rng(RNG_SEED)

    years = sorted(df_agg['ano'].unique())
    all_se = sorted(df_agg['se'].unique())
    all_se = [s for s in all_se if s <= MAX_SE]

    if monitor_year:
        monitor_years = [monitor_year]
    else:
        monitor_years = years

    # Construir matriz ano × SE
    matrix = {}
    for _, row in df_agg.iterrows():
        a, s, c = int(row['ano']), int(row['se']), int(row['casos'])
        if s > MAX_SE:
            continue
        matrix[(a, s)] = c

    # Preencher zeros para SE×ano ausentes
    for y in years:
        for s in all_se:
            if (y, s) not in matrix:
                matrix[(y, s)] = 0

    # Exposições (pop / 100_000)
    exposures = {}
    for y in years:
        pop = populations.get(y, populations.get(str(y), 100_000))
        exposures[y] = pop / 100_000

    # RAW data para o dashboard
    raw = []
    for s in all_se:
        entry = {'se': s}
        for y in years:
            entry[f'c{y}'] = matrix.get((y, s), 0)
        raw.append(entry)

    # Canal por ano monitorado (leave-one-out)
    channels = {}
    params = {}

    for mon_year in monitor_years:
        if base_hist_years:
            # Base histórica fixa: calibra sempre com os mesmos anos
            train_years = [y for y in base_hist_years if y in years]
            if len(train_years) < 2:
                train_years = years  # fallback
        elif leave_one_out:
            train_years = [y for y in years if y != mon_year]
        else:
            train_years = years

        if len(train_years) < 2:
            train_years = years  # fallback: usa todos se poucos anos

        channel_se = []
        params_se = []

        for s in all_se:
            # Janela de SE vizinhas: com 3 anos-base a SE isolada dá 3 observações, e
            # com 3 pontos a dispersão não é identificável — a faixa sai absurdamente
            # estreita. Na APS de 2026 o p90 ficava 9% acima do p50: qualquer semana
            # 9% acima da mediana virava 'emergência'. Com ±2 SE são 15 observações e
            # a faixa vai para +23%, que é a variabilidade real. É a mesma ideia da
            # janela de referência do Farrington/Noufaily (±3 semanas).
            # Medido: alarmes de 'Todos' na APS caem de 5 para 0 em 2026 e de 9 para 2
            # em 2025, e a detecção da epidemia de dengue de 2025 na UPA fica em 15
            # semanas — inalterada. JANELA_SE = 0 restaura o comportamento anterior.
            vizinhas = [((s - 1 + d) % MAX_SE) + 1
                        for d in range(-JANELA_SE, JANELA_SE + 1)] if JANELA_SE else [s]
            cases_train, exp_train = [], []
            for v in vizinhas:
                for y in train_years:
                    cases_train.append(matrix.get((y, v), 0))
                    exp_train.append(exposures.get(y, 1.0))

            # Exposição do ano monitorado
            e_mon = exposures.get(mon_year, 1.0)

            if denominadores is not None:
                # Canal de PROPORÇÃO (decisão 2026-08-28): o agravo é medido como
                # fração do total de atendimentos da semana, não como contagem.
                n_train = [denominadores.get((y, v), 0)
                           for v in vizinhas for y in train_years]
                n_mon = denominadores.get((mon_year, s), 0)
                qs, a_s, b_s = _betabinom_channel_se(cases_train, n_train, n_mon)
            else:
                # Motor único (decisão 2026-07-19): bootstrap sobre os anos-base + percentil da
                # mistura — ver _bootstrap_channel_se() e contrato_motor_canal.md seção 4b.
                # a_s/b_s aqui são só o ajuste central de registro, não geram mais `qs` sozinhos.
                qs, a_s, b_s = _bootstrap_channel_se(cases_train, exp_train, e_mon, rng=rng)
            channel_se.append(qs)
            params_se.append({'shape': round(a_s, 4), 'rate': round(b_s, 4)})

        channels[str(mon_year)] = channel_se
        params[str(mon_year)] = params_se

    # Classificações por ano monitorado
    classifications = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        clf = []
        for s_idx, s in enumerate(all_se):
            obs = matrix.get((mon_year, s), 0)
            zone = classify_zone(obs, ch[s_idx])
            clf.append(zone)
        classifications[str(mon_year)] = clf

    # Exceedance ratio (obs / P90)
    exceedance = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        exc = []
        for s_idx, s in enumerate(all_se):
            obs = matrix.get((mon_year, s), 0)
            p90 = ch[s_idx][4]  # index 4 = P90
            ratio = obs / max(p90, 1)
            exc.append(round(ratio, 3))
        exceedance[str(mon_year)] = exc

    # KPIs por ano
    kpis = {}
    for mon_year in monitor_years:
        ch = channels[str(mon_year)]
        cases_year = [matrix.get((mon_year, s), 0) for s in all_se]
        se_above_p90 = sum(
            1 for s_idx, s in enumerate(all_se)
            if matrix.get((mon_year, s), 0) > ch[s_idx][4]
        )
        kpis[str(mon_year)] = {
            'total': sum(cases_year),
            'pico': max(cases_year),
            'pico_se': all_se[cases_year.index(max(cases_year))],
            'se_acima_p90': se_above_p90,
        }

    return {
        'agravo': agravo_name,
        'familia':         'proporcao' if denominadores is not None else 'contagem',
        'years': [int(y) for y in years],
        'se_list': [int(s) for s in all_se],
        'populations': {str(k): int(v) for k, v in populations.items()},
        'raw': raw,
        'channels': channels,
        'params': params,
        'classifications': classifications,
        'exceedance': exceedance,
        'kpis': kpis,
    }


# ── Helpers para pipeline incremental ────────────────────────────────

def _save_channel_state(all_channels, path, base_hist_years, mon_year, denominador=None):
    """Salva params congelados (shape/rate/thresholds + raw histórico) para runs incrementais.

    Chamado apenas no run completo (janeiro / primeiro run).
    Em runs subsequentes, _rebuild_from_state() usa este arquivo.
    """
    state = {
        'generated': pd.Timestamp.now().isoformat(),
        'base_hist_years': base_hist_years,
        'monitor_year': mon_year,
        # Denominador dos canais de proporção (linhas de CID por SE). Precisa ser
        # persistido: no run incremental o CSV só traz o ano monitorado, e sem o
        # histórico os limiares beta-binomiais não podem ser refeitos.
        'denominador_linhas': {f'{a}-{w}': int(v)
                               for (a, w), v in (denominador or {}).items()},
        'channels': {}
    }
    for name, ch in all_channels.items():
        # raw_hist: todas as SEs de todos os anos EXCETO o ano monitorado
        raw_hist = [{k: v for k, v in r.items() if k != f'c{mon_year}'}
                    for r in ch['raw']]
        state['channels'][name] = {
            'agravo':   ch['agravo'],
            'familia':  ch.get('familia', 'contagem'),
            'se_list':  ch['se_list'],
            'channels': ch['channels'],   # thresholds P10-P90 por SE — CONGELADOS
            'params':   ch['params'],     # shape/rate por SE — CONGELADOS
            'raw_hist': raw_hist,         # c2023/c2024/c2025 por SE — CONGELADOS
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, separators=(',', ':'), cls=NumpyEncoder)
    size_kb = Path(path).stat().st_size / 1024
    print(f"   → channel_state.json salvo ({size_kb:.0f} KB, {len(state['channels'])} canais)")


def _rebuild_from_state(state_ch, new_obs_df, populations, mon_year, denominadores=None):
    """Reconstrói canal a partir de params congelados + observações novas do ano monitorado.

    Sem MLE nem Monte Carlo — usa thresholds já calculados.
    new_obs_df: DataFrame [ano, se, casos] com apenas o ano monitorado.
    """
    se_list = state_ch['se_list']
    ch_key  = str(mon_year)

    # Thresholds congelados (P10-P90) — da chave mon_year ou primeira disponível
    frozen = state_ch['channels'].get(ch_key)
    if frozen is None:
        frozen = list(state_ch['channels'].values())[0] if state_ch['channels'] else None
    if frozen is None:
        return None  # agravo sem histórico válido

    # Novas observações de 2026 por SE
    new_obs = {}
    for _, row in new_obs_df[new_obs_df['ano'] == mon_year].iterrows():
        s = int(row['se'])
        if 1 <= s <= MAX_SE:
            new_obs[s] = int(row['casos'])

    # Reconstruir raw: histórico + c_mon_year
    raw = []
    for r_hist in state_ch['raw_hist']:
        s = int(r_hist['se'])
        entry = dict(r_hist)
        entry[f'c{mon_year}'] = new_obs.get(s, 0)
        raw.append(entry)

    # Anos presentes no raw
    years = sorted({int(k[1:]) for r in raw for k in r if k.startswith('c')})

    # Reconstruir obs por ano a partir do raw (para heatmap histórico)
    obs_by_year = {}
    for r in raw:
        s = int(r['se'])
        for k, v in r.items():
            if k.startswith('c') and k[1:].isdigit():
                y = int(k[1:])
                if y not in obs_by_year:
                    obs_by_year[y] = {}
                obs_by_year[y][s] = v

    # Channels e Classifications para TODOS os anos (mesmo frozen — base_hist_years fixo)
    # Necessário para heatmap histórico e seleção de ano no dashboard
    channels_all = {}
    classifications_all = {}
    exceedance_all = {}
    kpis_all = {}

    # Canal de proporção: os limiares NÃO são constantes — dependem do total de
    # atendimentos da semana. Recalcula-se a preditiva beta-binomial com (a,b)
    # congelados e o denominador observado naquela SE.
    _familia = state_ch.get('familia', 'contagem')
    _par = state_ch.get('params', {}).get(str(mon_year)) or \
           (list(state_ch.get('params', {}).values())[0] if state_ch.get('params') else [])

    for y in years:
        obs_y = obs_by_year.get(y, {})
        clf_y, exc_y = [], []
        frozen_y = frozen
        if _familia == 'proporcao' and denominadores is not None and _par:
            frozen_y = []
            for i, s in enumerate(se_list):
                pr = _par[i] if i < len(_par) else {}
                a_, b_ = pr.get('shape'), pr.get('rate')
                n_ = denominadores.get((y, s), 0)
                if not a_ or not b_ or not n_:
                    frozen_y.append(frozen[i])
                else:
                    frozen_y.append([float(_scipy_betabinom.ppf(q, int(n_), a_, b_))
                                     for q in QUANTILES])
        for i, s in enumerate(se_list):
            obs = obs_y.get(s, 0)
            t   = frozen_y[i]
            clf_y.append(classify_zone(obs, t))
            exc_y.append(round(obs / max(t[4], 1), 3))
        channels_all[str(y)]     = frozen_y
        classifications_all[str(y)] = clf_y
        exceedance_all[str(y)]   = exc_y
        cases_y = [obs_y.get(s, 0) for s in se_list]
        max_c   = max(cases_y) if cases_y else 0
        kpis_all[str(y)] = {
            'total':        sum(cases_y),
            'pico':         max_c,
            'pico_se':      se_list[cases_y.index(max_c)] if max_c > 0 else 0,
            'se_acima_p90': sum(1 for i, s in enumerate(se_list)
                                if obs_y.get(s, 0) > frozen_y[i][4])
        }

    return {
        'agravo':          state_ch['agravo'],
        'years':           years,
        'se_list':         se_list,
        'populations':     {str(k): int(v) for k, v in populations.items()},
        'raw':             raw,
        'channels':        channels_all,
        'params':          state_ch['params'],
        'classifications': classifications_all,
        'exceedance':      exceedance_all,
        'kpis':            kpis_all,
    }


# ── Agregação de dados brutos ────────────────────────────────────────

# ── Contagem de casos por SE ──────────────────────────────────────────────────
# O CSV canônico passou a ser de nível-atendimento (uma linha por atendimento×CID,
# com `atend_id`). Todo agravo é uma UNIÃO de CIDs — capítulo, SINAN, síndrome,
# "Todos os atendimentos" — e somar linhas conta o mesmo atendimento uma vez por
# código que ele carrega. Na APS essa razão subiu de 1,10 para 1,65 códigos por
# atendimento entre 2023 e 2026, deriva de registro que o canal lia como epidemia.
# CSVs antigos (sem a coluna) continuam funcionando pela soma de quantidade.

def contar_casos(gdf, col_qty='quantidade', dedup=False):
    """Casos por (ano, se).

    dedup=True  → ATENDIMENTOS distintos. É a medida de DEMANDA e só faz sentido
                  para o agregado: um paciente com três CID é um atendimento.
    dedup=False → LINHAS de CID. É o numerador dos canais de agravo, porque o
                  denominador deles também é linha (ver o canal de proporção):
                  com numerador e denominador na mesma unidade, a deriva de
                  codificação se cancela. Medido na APS de 2026: o capítulo
                  respiratório vai de 13 para 5 semanas de alarme, o capítulo I
                  de 9 para 2 — enquanto a fração de ATENDIMENTOS piorava (16 e
                  10), porque a partir de jul/2025 cada atendimento passou a
                  carregar mais códigos e a chance de tocar qualquer capítulo
                  subiu junto (osteomuscular: 4,6% dos atendimentos em 2023,
                  8,5% em 2026 — mas 4,3% e 5,9% das linhas).
    """
    if dedup and ATEND_COL in gdf.columns:
        agg = gdf.groupby(['ano_epi', 'semana_epi'])[ATEND_COL].nunique().reset_index()
    else:
        agg = gdf.groupby(['ano_epi', 'semana_epi'])[col_qty].sum().reset_index()
    agg.columns = ['ano', 'se', 'casos']
    return agg[agg['se'] <= MAX_SE]


def volume_por(df, col_chave, col_qty='quantidade'):
    """Volume por chave (ranking de top-N), em linhas — mesma unidade dos canais de agravo."""
    return df.groupby(col_chave)[col_qty].sum()


def aggregate_raw_data(df, col_date, col_cid, col_qty='quantidade',
                       group_by='chapter', sinan_only=False):
    """
    Agrega dados brutos em formato SE × ano × agravo.

    Parâmetros:
        df: DataFrame com dados de atendimentos
        col_date: nome da coluna de data
        col_cid: nome da coluna de CID (código ou descrição)
        col_qty: nome da coluna de quantidade
        group_by: 'chapter' (capítulo CID), 'cid' (CID individual),
                  'sinan' (agravos SINAN), 'all' (todos atendimentos)
        sinan_only: se True, filtra apenas CIDs de notificação SINAN

    Retorna:
        dict {agravo_name: DataFrame[ano, se, casos]}
    """
    # Detectar SE
    df = df.copy()
    if 'ano_epi' not in df.columns or 'semana_epi' not in df.columns:
        dates = pd.to_datetime(df[col_date], dayfirst=True, errors='coerce')
        df['ano_epi'] = 0
        df['semana_epi'] = 0
        for idx in df.index:
            if pd.notna(dates[idx]):
                ae, se = epi_week(dates[idx])
                df.at[idx, 'ano_epi'] = ae
                df.at[idx, 'semana_epi'] = se

    # Mapear CID para grupo
    if group_by == 'all':
        df['_grupo'] = 'Todos os atendimentos'
    elif group_by == 'chapter':
        df['_grupo'] = df[col_cid].apply(cid_to_chapter)
    elif group_by == 'sinan':
        df['_grupo'] = df[col_cid].apply(cid_to_sinan)
        df = df[df['_grupo'] != 'Outros']
    else:
        df['_grupo'] = df[col_cid]

    # Agregar
    result = {}
    for grupo, gdf in df.groupby('_grupo'):
        if pd.isna(grupo) or grupo is None or str(grupo).strip() == '':
            continue
        agg = contar_casos(gdf, col_qty, dedup=(group_by == 'all'))
        result[str(grupo)] = agg

    return result


# ── Mapeamento CID descrição → código (tabela CID-10 DATASUS) ────────
# Cobre os CIDs mais prevalentes em UPAs/emergências + todos SINAN

CID_DESC_TO_CODE = {
    # Gerado automaticamente a partir da tabela CID-10 DATASUS
    # Cobertura: 96,5% dos atendimentos das UPAs Rio Claro/SP
    '(OSTEO)ARTROSE EROSIVA': 'M15.4',
    '(OSTEO)ARTROSE PRIMARIA GENERALIZADA': 'M15.0',
    '(SUPER)INFECCAO DELTA AGUDA DE PORTADOR DE HEPATITE B': 'B17.0',
    'ABANDONO': 'T74.0',
    'ABDOME AGUDO': 'R10.0',
    'ABORTAMENTO HABITUAL': 'N96',
    'ABORTO ESPONTANEO': 'O03',
    'ABORTO ESPONTANEO - INCOMPLETO, COM OUTRAS COMPLICACOES OU COM COMPLICACOES NAO ESPECIFICADAS': 'O03.3',
    'ABORTO ESPONTANEO - INCOMPLETO, COMPLICADO POR HEMORRAGIA EXCESSIVA OU TARDIA': 'O03.1',
    'ABORTO ESPONTANEO - INCOMPLETO, COMPLICADO POR INFECCAO DO TRATO GENITAL OU DOS ORGAOS PELVICOS': 'O03.0',
    'ABORTO ESPONTANEO - INCOMPLETO, SEM COMPLICACOES': 'O03.4',
    'ABORTO NAO ESPECIFICADO': 'O06',
    'ABORTO NAO ESPECIFICADO - COMPLETO OU NAO ESPECIFICADO, SEM COMPLICACOES': 'O06.9',
    'ABORTO RETIDO': 'O02.1',
    'ABRASAO DENTARIA': 'K03.1',
    'ABSCESSO AMEBIANO DO FIGADO': 'A06.4',
    'ABSCESSO ANAL': 'K61.0',
    'ABSCESSO ANORRETAL': 'K61.2',
    'ABSCESSO CEREBRAL FEOMICOTICO': 'B43.1',
    'ABSCESSO CUTANEO FURUNCULO E ANTRAZ': 'L02',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DA FACE': 'L02.0',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DA NADEGA': 'L02.3',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DE LOCALIZACAO NAO ESPECIFICADA': 'L02.9',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DE OUTRAS LOCALIZACOES': 'L02.8',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DO PESCOCO': 'L02.1',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DO TRONCO': 'L02.2',
    'ABSCESSO CUTANEO, FURUNCULO E ANTRAZ DO(S) MEMBRO(S)': 'L02.4',
    'ABSCESSO DA BAINHA TENDINEA': 'M65.0',
    'ABSCESSO DA GLANDULA DE BARTHOLIN': 'N75.1',
    'ABSCESSO DA MAMA ASSOCIADA AO PARTO': 'O91.1',
    'ABSCESSO DAS REGIOES ANAL E RETAL': 'K61',
    'ABSCESSO DE BOLSA SINOVIAL': 'M71.0',
    'ABSCESSO DE GLANDULA SALIVAR': 'K11.3',
    'ABSCESSO DO INTESTINO': 'K63.0',
    'ABSCESSO DO MEDIASTINO': 'J85.3',
    'ABSCESSO DO OUVIDO EXTERNO': 'H60.0',
    'ABSCESSO DO PULMAO COM PNEUMONIA': 'J85.1',
    'ABSCESSO E CISTO FEOMICOTICO SUBCUTANEOS': 'B43.2',
    'ABSCESSO E GRANULOMA INTRACRANIANOS': 'G06.0',
    'ABSCESSO EXTRADURAL E SUBDURAL NAO ESPECIFICADOS': 'G06.2',
    'ABSCESSO INTRA-ESFINCTERIANO': 'K61.4',
    'ABSCESSO PERIAMIGDALIANO': 'J36',
    'ABSCESSO PERIAPICAL COM FISTULA': 'K04.6',
    'ABSCESSO PERIAPICAL SEM FISTULA': 'K04.7',
    'ABSCESSO RENAL E PERINEFRETICO': 'N15.1',
    'ABSCESSO RETAL': 'K61.1',
    'ABSCESSO RETROFARINGEO E PARAFARINGEO': 'J39.0',
    'ABSCESSO VULVAR': 'N76.4',
    'ABSCESSO, FURUNCULO E ANTRAZ DO NARIZ': 'J34.0',
    'ABUSO DE SUBSTANCIAS QUE NAO PRODUZEM DEPENDENCIA': 'F55',
    'ABUSO PSICOLOGICO': 'T74.3',
    'ABUSO SEXUAL': 'T74.2',
    'ACALASIA DO CARDIA': 'K22.0',
    'ACANTOSE NIGRICANS': 'L83',
    'ACHADO ANORMAL DE EXAME QUIMICO DO SANGUE, NAO ESPECIFICADO': 'R79.9',
    'ACHADOS ANORMAIS AO EXAME CITOLOGICO E HISTOLOGICO DA URINA': 'R82.8',
    'ACHADOS ANORMAIS AO EXAME MICROBIOLOGICO DA URINA': 'R82.7',
    'ACHADOS ANORMAIS DE EXAMES DIAGNOSTICOS POR IMAGEM DE OUTRAS ESTRUTURAS SOMATICAS ESPECIFICADAS': 'R93.8',
    'ACHADOS ANORMAIS DE EXAMES PARA DIAGNOSTICO POR IMAGEM DO CORACAO E DA CIRCULACAO CORONARIANA': 'R93.1',
    'ACHADOS ANORMAIS DE EXAMES PARA DIAGNOSTICO POR IMAGEM DOS ORGAOS URINARIOS': 'R93.4',
    'ACHADOS ANORMAIS DE MATERIAL PROVENIENTE DE OUTROS ORGAOS APARELHOS SISTEMAS E TECIDOS': 'R89',
    'ACHADOS ANORMAIS DE MATERIAL PROVENIENTE DOS ORGAOS GENITAIS FEMININOS': 'R87',
    'ACHADOS ANORMAIS DE MATERIAL PROVENIENTE DOS ORGAOS GENITAIS MASCULINOS': 'R86',
    'ACHADOS ANORMAIS, DE EXAMES PARA DIAGNOSTICO POR IMAGEM, DA MAMA': 'R92',
    'ACHADOS ANORMAIS, DE EXAMES PARA DIAGNOSTICO POR IMAGEM, DO PULMAO': 'R91',
    'ACIDENTE COM UM VEICULO A MOTOR OU NAO-MOTORIZADO TIPO(S) DE VEICULO(S) NAO ESPECIFICADO(S)': 'V89',
    'ACIDENTE DE TRANSITO DE TIPO ESPECIFICADO MAS SENDO DESCONHECIDO O MODO DE TRANSPORTE DA VITIMA': 'V87',
    'ACIDENTE DE TRANSPORTE NAO ESPECIFICADO': 'V99',
    'ACIDENTE VASCULAR CEREBRAL, NAO ESPECIFICADO COMO HEMORRAGICO OU ISQUEMICO': 'I64',
    'ACIDENTES VASCULARES CEREBRAIS ISQUEMICOS TRANSITORIOS E SINDROMES CORRELATAS': 'G45',
    'ACIDOSE': 'E87.2',
    'ACNE': 'L70',
    'ACNE INFANTIL': 'L70.4',
    'ACNE VULGAR': 'L70.0',
    'ACNE, NAO ESPECIFICADA': 'L70.9',
    'ACONSELHAMENTO E SUPERVISAO PARA ABUSO DE ALCOOL': 'Z71.4',
    'ACONSELHAMENTO E SUPERVISAO PARA ABUSO DE DROGAS': 'Z71.5',
    'ACONSELHAMENTO GERAL SOBRE CONTRACEPCAO': 'Z30.0',
    'ACONSELHAMENTO NAO ESPECIFICADO': 'Z71.9',
    'ACONSELHAMENTO NAO ESPECIFICADO EM MATERIA DE SEXUALIDADE': 'Z70.9',
    'ACONSELHAMENTO RELATIVO AO COMPORTAMENTO E A ORIENTACAO SEXUAL DO SUJEITO': 'Z70.1',
    'ACONSELHAMENTO RELATIVO AS ATITUDES COMPORTAMENTO E ORIENTACAO EM MATERIA DE SEXUALIDADE': 'Z70',
    'ACRODERMATITE PAPULAR INFANTIL [SINDROME DE GIANOTTI-CROSTI]': 'L44.4',
    'ACTINOMICOSE': 'A42',
    'ACTINOMICOSE CERVICOFACIAL': 'A42.2',
    'ACTINOMICOSE NAO ESPECIFICADA': 'A42.9',
    'ADENOMEGALIA OU AUMENTO DE VOLUME DOS GANGLIOS LINFATICOS, NAO ESPECIFICADO': 'R59.9',
    'ADERENCIAS DO PERITONIO PELVICO POS-PROCEDIMENTOS': 'N99.4',
    'ADERENCIAS INTESTINAIS (BRIDAS) COM OBSTRUCAO': 'K56.5',
    'ADERENCIAS PELVIPERITONAIS FEMININAS': 'N73.6',
    'ADERENCIAS PERITONIAIS': 'K66.0',
    'ADMINISTRACAO POR OUTROS MEIOS DE MEDICAMENTO OU SUBSTANCIA BIOLOGICAS CONTAMINADOS': 'Y64.8',
    'AFECCAO HEMORRAGICA NAO ESPECIFICADA': 'D69.9',
    'AFECCAO NAO ESPECIFICADA DA PROSTATA': 'N42.9',
    'AFECCAO PLEURAL NAO ESPECIFICADA': 'J94.9',
    'AFECCAO RESPIRATORIA NAO ESPECIFICADA DEVIDA A PRODUTOS QUIMICOS, GASES, FUMACA E VAPORES': 'J68.9',
    'AFECCOES ALVEOLARES E PARIETO-ALVEOLARES': 'J84.0',
    'AFECCOES ATROFICAS DA PELE': 'L90',
    'AFECCOES BOLHOSAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L14',
    'AFECCOES DA PELE E DO TECIDO SUBCUTANEO RELACIONADAS COM A RADIACAO, NAO ESPECIFICADAS': 'L59.9',
    'AFECCOES DA PELE E DO TECIDO SUBCUTANEO, NAO ESPECIFICADOS': 'L98.9',
    'AFECCOES DAS GLANDULAS SUDORIPARAS ECRINAS': 'L74',
    'AFECCOES DAS GLANDULAS SUDORIPARAS ECRINAS, NAO ESPECIFICADAS': 'L74.9',
    'AFECCOES DAS UNHAS': 'L60',
    'AFECCOES DAS UNHAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L62',
    'AFECCOES DAS UNHAS EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L62.8',
    'AFECCOES DAS UNHAS, NAO ESPECIFICADAS': 'L60.9',
    'AFECCOES DEGENERATIVAS DO GLOBO OCULAR': 'H44.5',
    'AFECCOES ERITEMATOSAS, NAO ESPECIFICADAS': 'L53.9',
    'AFECCOES EXOFTALMICAS': 'H05.2',
    'AFECCOES FOLICULARES, NAO ESPECIFICADAS': 'L73.9',
    'AFECCOES GRANULOMATOSAS DA PELE E DO TECIDO SUBCUTANEO': 'L92',
    'AFECCOES GRANULOMATOSAS DA PELE E DO TECIDO SUBCUTANEO, NAO ESPECIFICADOS': 'L92.9',
    'AFECCOES HIPERTROFICAS DA PELE': 'L91',
    'AFECCOES HIPERTROFICAS DA PELE, NAO ESPECIFICADAS': 'L91.9',
    'AFECCOES INFLAMATORIAS DOS MAXILARES': 'K10.2',
    'AFECCOES LIGADAS A GRAVIDEZ, NAO ESPECIFICADAS': 'O26.9',
    'AFECCOES LOCALIZADAS DO TECIDO CONJUNTIVO, NAO ESPECIFICADAS': 'L94.9',
    'AFECCOES NAO ESPECIFICADAS ASSOCIADAS COM OS ORGAOS GENITAIS FEMININOS E COM O CICLO MENSTRUAL': 'N94.9',
    'AFECCOES OCULARES DEVIDAS AO VIRUS DO HERPES': 'B00.5',
    'AFECCOES PAPULO-DESCAMATIVAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L45',
    'AFECCOES PAPULO-DESCAMATIVAS, NAO ESPECIFICADAS': 'L44.9',
    'AFECCOES RESPIRATORIAS CRONICAS DEVIDAS A PRODUTOS QUIMICOS, GASES, FUMACAS E VAPORES': 'J68.4',
    'AFECCOES RESPIRATORIAS DEVIDAS A AGENTES EXTERNOS NAO ESPECIFICADOS': 'J70.9',
    'AFECCOES RESPIRATORIAS DEVIDAS A INALACAO DE PRODUTOS QUIMICOS GASES FUMACAS E VAPORES': 'J68',
    'AFECCOES RESPIRATORIAS POS-PROCEDIMENTOS NAO CLASSIFICADAS EM OUTRA PARTE': 'J95',
    'AFOGAMENTO E SUBMERSAO CONSEQUENTE A QUEDA DENTRO DE UMA PISCINA': 'W68',
    'AFOGAMENTO E SUBMERSAO EM PISCINA': 'W67',
    'AFOGAMENTO E SUBMERSAO NAO MORTAL': 'T75.1',
    'AFONIA': 'R49.1',
    'AFTAS BUCAIS RECIDIVANTES': 'K12.0',
    'AGITACAO E INQUIETACAO': 'R45.1',
    'AGORAFOBIA': 'F40.0',
    'AGRANULOCITOSE': 'D70',
    'AGRESSAO POR MEIO DE DISPARO DE ARMA DE FOGO DE MAO': 'X93',
    'AGRESSAO POR MEIO DE DISPARO DE ARMA DE FOGO DE MAO - LOCAL NAO ESPECIFICADO': 'X93.9',
    'AGRESSAO POR MEIO DE DISPARO DE ARMA DE FOGO DE MAO - RUA E ESTRADA': 'X93.4',
    'AGRESSAO POR MEIO DE DROGAS MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS': 'X85',
    'AGRESSAO POR MEIO DE ENFORCAMENTO ESTRANGULAMENTO E SUFOCACAO': 'X91',
    'AGRESSAO POR MEIO DE FORCA CORPORAL': 'Y04',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - AREAS DE COMERCIO E DE SERVICOS': 'Y04.5',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - HABITACAO COLETIVA': 'Y04.1',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - LOCAL NAO ESPECIFICADO': 'Y04.9',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - OUTROS LOCAIS ESPECIFICADOS': 'Y04.8',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - RESIDENCIA': 'Y04.0',
    'AGRESSAO POR MEIO DE FORCA CORPORAL - RUA E ESTRADA': 'Y04.4',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE': 'X99',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE - HABITACAO COLETIVA': 'X99.1',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE - LOCAL NAO ESPECIFICADO': 'X99.9',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE - RESIDENCIA': 'X99.0',
    'AGRESSAO POR MEIO DE OBJETO CORTANTE OU PENETRANTE - RUA E ESTRADA': 'X99.4',
    'AGRESSAO POR MEIO DE OUTROS PRODUTOS QUIMICOS E SUBSTANCIAS NOCIVAS ESPECIFICADOS': 'X89',
    'AGRESSAO POR MEIO DE PRODUTOS QUIMICOS E SUBSTANCIAS NOCIVAS NAO ESPECIFICADOS': 'X90',
    'AGRESSAO POR MEIO DE PROJECAO OU COLOCACAO DA VITIMA DIANTE DE UM OBJETO EM MOVIMENTO': 'Y02',
    'AGRESSAO POR MEIO DE UM OBJETO CONTUNDENTE': 'Y00',
    'AGRESSAO POR MEIOS NAO ESPECIFICADOS': 'Y09',
    'AGRESSAO POR MEIOS NAO ESPECIFICADOS - LOCAL NAO ESPECIFICADO': 'Y09.9',
    'AGRESSAO POR MEIOS NAO ESPECIFICADOS - OUTROS LOCAIS ESPECIFICADOS': 'Y09.8',
    'AGRESSAO POR MEIOS NAO ESPECIFICADOS - RESIDENCIA': 'Y09.0',
    'AGRESSAO POR MEIOS NAO ESPECIFICADOS - RUA E ESTRADA': 'Y09.4',
    'AGRESSAO POR OUTROS MEIOS ESPECIFICADOS': 'Y08',
    'AGRESSAO POR OUTROS MEIOS ESPECIFICADOS - RESIDENCIA': 'Y08.0',
    'AGRESSAO POR OUTROS MEIOS ESPECIFICADOS - RUA E ESTRADA': 'Y08.4',
    'AGRESSAO SEXUAL POR MEIO DE FORCA FISICA': 'Y05',
    'AGRESSAO SEXUAL POR MEIO DE FORCA FISICA - RESIDENCIA': 'Y05.0',
    'AJUSTAMENTO E MANUSEIO DE DISPOSITIVO DE ACESSO VASCULAR': 'Z45.2',
    'AJUSTAMENTO E MANUSEIO DE DISPOSITIVO IMPLANTADO': 'Z45',
    'AJUSTAMENTO E MANUSEIO DE DISPOSITIVO IMPLANTADO NAO ESPECIFICADO': 'Z45.9',
    'AJUSTAMENTO E MANUSEIO DE OUTROS DISPOSITIVOS IMPLANTADOS': 'Z45.8',
    'ALERGIA NAO ESPECIFICADA': 'T78.4',
    'ALESQUERIOSE': 'B48.2',
    'ALGUMAS COMPLICACOES PRECOCES DOS TRAUMATISMOS NAO CLASSIFICADAS EM OUTRA PARTE': 'T79',
    'ALOPECIA ANDROGENICA': 'L64',
    'ALOPECIA ANDROGENICA, NAO ESPECIFICADA': 'L64.9',
    'ALOPECIA AREATA': 'L63',
    'ALOPECIA AREATA, NAO ESPECIFICADA': 'L63.9',
    'ALOPECIA CICATRICIAL (PERDA DE CABELOS OU PELOS CICATRICIAL)': 'L66',
    'ALOPECIA MUCINOSA': 'L65.2',
    'ALTERACAO DO HABITO INTESTINAL': 'R19.4',
    'ALTERACOES AGUDAS DA PELE DEVIDAS A RADIACAO ULTRAVIOLETA, NAO ESPECIFICADAS': 'L56.9',
    'ALTERACOES DA SECRECAO SALIVAR': 'K11.7',
    'ALTERACOES MORFOLOGICAS CONGENITAS DOS CABELOS NAO CLASSIFICADAS EM OUTRA PARTE': 'Q84.1',
    'ALTERACOES NAS MEMBRANAS DA CORNEA': 'H18.3',
    'ALTERACOES POS-ERUPTIVAS DA COR DOS TECIDOS DUROS DOS DENTES': 'K03.7',
    'ALUCINACOES AUDITIVAS': 'R44.0',
    'ALUCINACOES NAO ESPECIFICADAS': 'R44.3',
    'ALUCINACOES VISUAIS': 'R44.1',
    'ALUCINOSE ORGANICA': 'F06.0',
    'ALVEOLITE MAXILAR': 'K10.3',
    'AMAUROSE FUGAZ': 'G45.3',
    'AMBLIOPIA POR ANOPSIA': 'H53.0',
    'AMEACA DE ABORTO': 'O20.0',
    'AMEBIASE': 'A06',
    'AMEBIASE CUTANEA': 'A06.7',
    'AMEBIASE INTESTINAL CRONICA': 'A06.1',
    'AMEBIASE NAO ESPECIFICADA': 'A06.9',
    'AMENORREIA PRIMARIA': 'N91.0',
    'AMENORREIA SECUNDARIA': 'N91.1',
    'AMENORREIA, NAO ESPECIFICADA': 'N91.2',
    'AMIGDALITE AGUDA DEVIDA A OUTROS MICROORGANISMOS ESPECIFICADOS': 'J03.8',
    'AMIGDALITE AGUDA NAO ESPECIFICADA': 'J03.9',
    'AMIGDALITE CRONICA': 'J35.0',
    'AMIGDALITE ESTREPTOCOCICA': 'J03.0',
    'AMNESIA ANTEROGRADA': 'R41.1',
    'AMNESIA GLOBAL TRANSITORIA': 'G45.4',
    'AMNESIA RETROGRADA': 'R41.2',
    'AMPUTACAO TRAUMATICA AO NIVEL DO PUNHO E DA MAO': 'S68',
    'AMPUTACAO TRAUMATICA DA ORELHA': 'S08.1',
    'AMPUTACAO TRAUMATICA DA PERNA': 'S88',
    'AMPUTACAO TRAUMATICA DA PERNA AO NIVEL NAO ESPECIFICADO': 'S88.9',
    'AMPUTACAO TRAUMATICA DE APENAS UM ARTELHO': 'S98.1',
    'AMPUTACAO TRAUMATICA DE DOIS OU MAIS DEDOS SOMENTE (COMPLETA) (PARCIAL)': 'S68.2',
    'AMPUTACAO TRAUMATICA DE UM OUTRO DEDO APENAS (COMPLETA) (PARCIAL)': 'S68.1',
    'AMPUTACAO TRAUMATICA DO PE AO NIVEL NAO ESPECIFICADO': 'S98.4',
    'AMPUTACAO TRAUMATICA DO POLEGAR (COMPLETA) (PARCIAL)': 'S68.0',
    'AMPUTACAO TRAUMATICA DO PUNHO E DA MAO, NIVEL NAO ESPECIFICADO': 'S68.9',
    'AMPUTACAO TRAUMATICA DO TORNOZELO E DO PE': 'S98',
    'AMPUTACOES TRAUMATICAS ENVOLVENDO MULTIPLAS REGIOES DO CORPO': 'T05',
    'ANCILOSTOMIASE NAO ESPECIFICADA': 'B76.9',
    'ANEMIA AGUDA POS-HEMORRAGICA': 'D62',
    'ANEMIA APLASTICA CONSTITUCIONAL': 'D61.0',
    'ANEMIA APLASTICA IDIOPATICA': 'D61.3',
    'ANEMIA APLASTICA NAO ESPECIFICADA': 'D61.9',
    'ANEMIA DEVIDA A DEFICIENCIA DE GLICOSE-6-FOSFATO-DESIDROGENASE [G-6-PD]': 'D55.0',
    'ANEMIA DEVIDA A TRANSTORNO ENZIMATICO NAO ESPECIFICADA': 'D55.9',
    'ANEMIA EM NEOPLASIAS': 'D63.0',
    'ANEMIA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'D63.8',
    'ANEMIA FALCIFORME COM CRISE': 'D57.0',
    'ANEMIA FALCIFORME SEM CRISE': 'D57.1',
    'ANEMIA HEMOLITICA ADQUIRIDA': 'D59',
    'ANEMIA HEMOLITICA ADQUIRIDA NAO ESPECIFICADA': 'D59.9',
    'ANEMIA HEMOLITICA HEREDITARIA NAO ESPECIFICADA': 'D58.9',
    'ANEMIA NAO ESPECIFICADA': 'D64.9',
    'ANEMIA NUTRICIONAL NAO ESPECIFICADA': 'D53.9',
    'ANEMIA POR DEFICIENCIA DE FERRO': 'D50',
    'ANEMIA POR DEFICIENCIA DE FERRO NAO ESPECIFICADA': 'D50.9',
    'ANEMIA POR DEFICIENCIA DE FERRO SECUNDARIA A PERDA DE SANGUE (CRONICA)': 'D50.0',
    'ANEMIA POR DEFICIENCIA DE FOLATO': 'D52',
    'ANEMIA POR DEFICIENCIA DE FOLATO NA DIETA': 'D52.0',
    'ANEMIA POR DEFICIENCIA DE FOLATO NAO ESPECIFICADA': 'D52.9',
    'ANEMIA POR DEFICIENCIA DE PROTEINAS': 'D53.0',
    'ANEMIA POR DEFICIENCIA DE VITAMINA B12': 'D51',
    'ANEMIA POR DEFICIENCIA DE VITAMINA B12 DEVIDA A DEFICIENCIA DE FATOR INTRINSECO': 'D51.0',
    'ANEMIA POR DEFICIENCIA DE VITAMINA B12 DEVIDA A MA-ABSORCAO SELETIVA DE VITAMINA B12 COM PROTEINURIA': 'D51.1',
    'ANEMIA POR DEFICIENCIA DE VITAMINA B12 NAO ESPECIFICADA': 'D51.9',
    'ANEMIA REFRATARIA COM EXCESSO DE BLASTOS': 'D46.2',
    'ANEMIA REFRATARIA SEM SIDEROBLASTOS': 'D46.0',
    'ANEMIA REFRATARIA, NAO ESPECIFICADA': 'D46.4',
    'ANESTESIA CUTANEA': 'R20.0',
    'ANEURISMA AORTICO DE LOCALIZACAO NAO ESPECIFICADA, SEM MENCAO DE RUPTURA': 'I71.9',
    'ANEURISMA CEREBRAL NAO-ROTO': 'I67.1',
    'ANEURISMA DA AORTA ABDOMINAL, SEM MENCAO DE RUPTURA': 'I71.4',
    'ANEURISMA DA AORTA TORACICA, SEM MENCAO DE RUPTURA': 'I71.2',
    'ANEURISMA DA AORTA TORACO-ABDOMINAL, SEM MENCAO DE RUPTURA': 'I71.6',
    'ANEURISMA DE OUTRAS ARTERIAS ESPECIFICADAS': 'I72.8',
    'ANEURISMA E DISSECCAO DA AORTA': 'I71',
    'ANGINA INSTAVEL': 'I20.0',
    'ANGINA PECTORIS': 'I20',
    'ANGINA PECTORIS, NAO ESPECIFICADA': 'I20.9',
    'ANGIOPATIA PERIFERICA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'I79.2',
    'ANGROSTRONGILIASE DEVIDA A PARASTRONGYLUS CANTONENSIS': 'B83.2',
    'ANODONTIA': 'K00.0',
    'ANOMALIA CROMOSSOMICA NAO ESPECIFICADA': 'Q99.9',
    'ANOMALIAS CONGENITAS OBSTRUTIVAS DA PELVE RENAL E MALFORMACOES CONGENITAS DO URETER': 'Q62',
    'ANOMALIAS DA FUNCAO PUPILAR': 'H57.0',
    'ANOMALIAS DENTOFACIAIS (INCLUSIVE A MALOCLUSAO)': 'K07',
    'ANOMALIAS IMPORTANTES (MAJOR) DO TAMANHO DA MANDIBULA': 'K07.0',
    'ANOREXIA': 'R63.0',
    'ANORMALIDADE DAS HEMACIAS': 'R71',
    'ANORMALIDADE DOS LEUCOCITOS NAO CLASSIFICADA EM OUTRA PARTE': 'R72',
    'ANORMALIDADE DOS NIVEIS DE ENZIMAS SERICAS, NAO ESPECIFICADA': 'R74.9',
    'ANORMALIDADES DA CONTRACAO UTERINA': 'O62',
    'ANORMALIDADES DA MARCHA E DA MOBILIDADE': 'R26',
    'ANORMALIDADES DA RESPIRACAO': 'R06',
    'ANORMALIDADES DO BATIMENTO CARDIACO': 'R00',
    'ANORMALIDADES DOS NIVEIS DE ENZIMAS SERICAS': 'R74',
    'ANOSMIA': 'R43.0',
    'ANSIEDADE GENERALIZADA': 'F41.1',
    'ANTICONCEPCAO': 'Z30',
    'ANURIA E OLIGURIA': 'R34',
    'APATIA E DESINTERESSE': 'R45.3',
    'APENDICITE AGUDA': 'K35',
    'APENDICITE AGUDA COM ABSCESSO PERITONIAL': 'K35.1',
    'APENDICITE AGUDA COM PERITONITE GENERALIZADA': 'K35.0',
    'APENDICITE AGUDA SEM OUTRA ESPECIFICACAO': 'K35.9',
    'APERTADO COLHIDO COMPRIMIDO OU ESMAGADO DENTRO DE OU ENTRE OBJETOS': 'W23',
    'APLASIA PURA ADQUIRIDA CRONICA DA SERIE VERMELHA': 'D60.0',
    'APLASIA PURA DA SERIE VERMELHA ADQUIRIDA (ERITROBLASTOPENIA)': 'D60',
    'APNEIA DE SONO': 'G47.3',
    'ARRITMIA CARDIACA NAO ESPECIFICADA': 'I49.9',
    'ARTERITE NAO ESPECIFICADA': 'I77.6',
    'ARTRITE E PIOLIARTRITE ESTAFILOCOCICAS': 'M00.0',
    'ARTRITE E POLIARTRITE DEVIDAS A OUTRO AGENTE BACTERIANO ESPECIFICADO': 'M00.8',
    'ARTRITE EM OUTRAS DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'M01.3',
    'ARTRITE EM OUTRAS DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'M01.8',
    'ARTRITE EM OUTRAS DOENCAS VIRAIS CLASSIFICADAS EM OUTRA PARTE': 'M01.5',
    'ARTRITE JUVENIL': 'M08',
    'ARTRITE JUVENIL NA DOENCA DE CHRON [ENTERITE REGIONAL]': 'M09.1',
    'ARTRITE JUVENIL NA PSORIASE': 'M09.0',
    'ARTRITE JUVENIL NAO ESPECIFICADA': 'M08.9',
    'ARTRITE MUTILANTE': 'M07.1',
    'ARTRITE NAO ESPECIFICADA': 'M13.9',
    'ARTRITE PIOGENICA': 'M00',
    'ARTRITE PIOGENICA, NAO ESPECIFICADA': 'M00.9',
    'ARTRITE REUMATOIDE COM COMPROMETIMENTO DE OUTROS ORGAOS E SISTEMAS': 'M05.3',
    'ARTRITE REUMATOIDE JUVENIL': 'M08.0',
    'ARTRITE REUMATOIDE NAO ESPECIFICADA': 'M06.9',
    'ARTRITE REUMATOIDE SORO-NEGATIVA': 'M06.0',
    'ARTRITE REUMATOIDE SORO-POSITIVA': 'M05',
    'ARTRITE REUMATOIDE SORO-POSITIVA NAO ESPECIFICADA': 'M05.9',
    'ARTRODESE': 'Z98.1',
    'ARTROPATIA DIABETICA': 'M14.2',
    'ARTROPATIA HEMOFILICA': 'M36.2',
    'ARTROPATIA NA DOENCA DE CROHN [ENTERITE REGIONAL]': 'M07.4',
    'ARTROPATIA POS-IMUNIZACAO': 'M02.2',
    'ARTROPATIA REACIONAL NAO ESPECIFICADA': 'M02.9',
    'ARTROPATIA TRAUMATICA': 'M12.5',
    'ARTROPATIAS EM OUTRAS DOENCAS ESPECIFICADAS CLASSIFICADAS EM OUTRA PARTE': 'M14.8',
    'ARTROPATIAS POS-INFECCIOSAS E REACIONAIS EM DOENCAS INFECCIOSAS CLASSIFICADAS EM OUTRA PARTE': 'M03',
    'ARTROPATIAS PSORIASICAS E ENTEROPATICAS': 'M07',
    'ARTROPATIAS REACIONAIS': 'M02',
    'ARTROSE MULTIPLA SECUNDARIA': 'M15.3',
    'ARTROSE NAO ESPECIFICADA': 'M19.9',
    'ARTROSE NAO ESPECIFICADA DA PRIMEIRA ARTICULACAO CARPOMETACARPIANA': 'M18.9',
    'ARTROSE POS-TRAUMATICA BILATERAL DA PRIMEIRA ARTICULACAO CARPOMETACARPIANA': 'M18.2',
    'ARTROSE POS-TRAUMATICA DE OUTRAS ARTICULACOES': 'M19.1',
    'ARTROSE PRIMARIA DE OUTRAS ARTICULACOES': 'M19.0',
    'ASCARIDIASE': 'B77',
    'ASCARIDIASE NAO ESPECIFICADA': 'B77.9',
    'ASCITE': 'R18',
    'ASMA': 'J45',
    'ASMA MISTA': 'J45.8',
    'ASMA NAO ESPECIFICADA': 'J45.9',
    'ASMA NAO-ALERGICA': 'J45.1',
    'ASMA PREDOMINANTEMENTE ALERGICA': 'J45.0',
    'ASPERGILOSE AMIGDALIANA': 'B44.2',
    'ASPIRACAO NEONATAL DE LEITE E ALIMENTO REGURGITADOS': 'P24.3',
    'ASSEPSIA INSUFICIENTE DURANTE INJECAO OU VACINACAO (IMUNIZACAO)': 'Y62.3',
    'ASSIMETRIA FACIAL': 'Q67.0',
    'ASSISTENCIA A GRAVIDEZ POR MOTIVO DE ABORTAMENTO HABITUAL': 'O26.2',
    'ASSISTENCIA E EXAME IMEDIATAMENTE APOS O PARTO': 'Z39.0',
    'ASSISTENCIA PRESTADA A MAE POR CICATRIZ UTERINA DEVIDA A UMA CIRURGIA ANTERIOR': 'O34.2',
    'ASSISTENCIA PRESTADA A MAE POR LESAO (SUSPEITADA) CAUSADA AO FETO POR ALCOOLISMO MATERNO': 'O35.4',
    'ASSISTENCIA PRESTADA A MAE POR MOTIVO DE APRESENTACAO ANORMAL CONHECIDA OU SUSPEITADA DO FETO': 'O32',
    'ASSISTENCIA PRESTADA A MAE POR POSICAO FETAL INSTAVEL': 'O32.0',
    'ASTIGMATISMO': 'H52.2',
    'ATAXIA CONGENITA NAO-PROGRESSIVA': 'G11.0',
    'ATAXIA NAO ESPECIFICADA': 'R27.0',
    'ATENCAO A ORIFICIOS ARTIFICIAIS': 'Z43',
    'ATEROSCLEROSE': 'I70',
    'ATEROSCLEROSE DAS ARTERIAS DAS EXTREMIDADES': 'I70.2',
    'ATEROSCLEROSE DE OUTRAS ARTERIAS': 'I70.8',
    'ATRASO DE CONSOLIDACAO DE FRATURA': 'M84.2',
    'ATRASO DO DESENVOLVIMENTO DEVIDO A DESNUTRICAO PROTEICO-CALORICA': 'E45',
    'ATRESIA DAS COANAS': 'Q30.0',
    'ATRITO DENTARIO EXCESSIVO': 'K03.0',
    'ATROFIA DA PROSTATA': 'N42.2',
    'ATROFIA DO REBORDO ALVEOLAR SEM DENTES': 'K08.2',
    'ATROFIA MUSCULAR ESPINAL E SINDROMES CORRELATAS': 'G12',
    'AUMENTO DA GLICEMIA': 'R73',
    'AUMENTO DE VOLUME DOS GANGLIOS LINFATICOS': 'R59',
    'AUMENTO DE VOLUME GENERALIZADO DE GANGLIOS LINFATICOS': 'R59.1',
    'AUMENTO DE VOLUME LOCALIZADO DE GANGLIOS LINFATICOS': 'R59.0',
    'AUMENTO DOS NIVEIS DE TRANSAMINASES E DA DESIDROGENASE LATICA (DHL)': 'R74.0',
    'AUSENCIA ADQUIRIDA DE ORGAOS NAO CLASSIFICADOS EM OUTRA PARTE': 'Z90',
    'AUSENCIA ADQUIRIDA DE PE E TORNOZELO': 'Z89.4',
    'AUSENCIA CONGENITA COMPLETA DO(S) MEMBRO(S) SUPERIOR(ES)': 'Q71.0',
    'AUSENCIA CONGENITA DO BRACO E DO ANTEBRACO, COM MAO PRESENTE': 'Q71.1',
    'AUSENCIA DE UM DOS MEMBROS DA FAMILIA': 'Z63.3',
    'AUSENCIA E APLASIA DO TESTICULO': 'Q55.0',
    'AUSENCIA OU PERDA DO DESEJO SEXUAL': 'F52.0',
    'AUTISMO ATIPICO': 'F84.1',
    'AUTISMO INFANTIL': 'F84.0',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL': 'X65',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - AREAS DE COMERCIO E DE SERVICOS': 'X65.5',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - FAZENDA': 'X65.7',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - HABITACAO COLETIVA': 'X65.1',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - LOCAL NAO ESPECIFICADO': 'X65.9',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - OUTROS LOCAIS ESPECIFICADOS': 'X65.8',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - RESIDENCIA': 'X65.0',
    'AUTO-INTOXICACAO VOLUNTARIA POR ALCOOL - RUA E ESTRADA': 'X65.4',
    'AUTOSSENSIBILIZACAO CUTANEA': 'L30.2',
    'BABESIOSE': 'B60.0',
    'BALANITE EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N51.2',
    'BALANOPOSTITE': 'N48.1',
    'BALANTIDIASE': 'A07.0',
    'BAQUETEAMENTO DOS DEDOS': 'R68.3',
    'BARTONELOSE': 'A44',
    'BARTONELOSE CUTANEA E CUTANEO-MUCOSA': 'A44.1',
    'BARTONELOSE NAO ESPECIFICADA': 'A44.9',
    'BEXIGA NEUROPATICA FLACIDA NAO CLASSIFICADA EM OUTRA PARTE': 'N31.2',
    'BEXIGA NEUROPATICA NAO-INIBIDA NAO CLASSIFICADA EM OUTRA PARTE': 'N31.0',
    'BEXIGA NEUROPATICA REFLEXA NAO CLASSIFICADA EM OUTRA PARTE': 'N31.1',
    'BLASTOMICOSE CUTANEA': 'B40.3',
    'BLEFARITE': 'H01.0',
    'BLEFAROCALASIA': 'H02.3',
    'BLEFAROCONJUNTIVITE': 'H10.5',
    'BLOQUEIO ATRIOVENTRICULAR DE PRIMEIRO GRAU': 'I44.0',
    'BLOQUEIO ATRIOVENTRICULAR DE SEGUNDO GRAU': 'I44.1',
    'BLOQUEIO ATRIOVENTRICULAR E DO RAMO ESQUERDO': 'I44',
    'BLOQUEIO ATRIOVENTRICULAR TOTAL': 'I44.2',
    'BOCA SECA, NAO ESPECIFICADA': 'R68.2',
    'BOCIO (ENDEMICO) NAO ESPECIFICADO, POR DEFICIENCIA DE IODO': 'E01.2',
    'BOCIO NAO-TOXICO DIFUSO': 'E04.0',
    'BOCIO NAO-TOXICO, NAO ESPECIFICADO': 'E04.9',
    'BOTULISMO': 'A05.1',
    'BOUBA': 'A66',
    'BRADICARDIA NAO ESPECIFICADA': 'R00.1',
    'BRONCOPNEUMONIA NAO ESPECIFICADA': 'J18.0',
    'BRONQUECTASIA': 'J47',
    'BRONQUIOLITE AGUDA': 'J21',
    'BRONQUIOLITE AGUDA DEVIDA A OUTROS MICROORGANISMOS ESPECIFICADOS': 'J21.8',
    'BRONQUIOLITE AGUDA DEVIDA A VIRUS SINCICIAL RESPIRATORIO': 'J21.0',
    'BRONQUITE AGUDA': 'J20',
    'BRONQUITE AGUDA DEVIDA A ESTREPTOCOCOS': 'J20.2',
    'BRONQUITE AGUDA DEVIDA A HAEMOPHILUS INFLUENZAE': 'J20.1',
    'BRONQUITE AGUDA DEVIDA A MYCOPLASMA PNEUMONIAE': 'J20.0',
    'BRONQUITE AGUDA DEVIDA A OUTROS MICROORGANISMOS ESPECIFICADOS': 'J20.8',
    'BRONQUITE AGUDA DEVIDA A RINOVIRUS': 'J20.6',
    'BRONQUITE AGUDA DEVIDA A VIRUS COXSACKIE': 'J20.3',
    'BRONQUITE AGUDA DEVIDA A VIRUS PARAINFLUENZA': 'J20.4',
    'BRONQUITE AGUDA DEVIDA A VIRUS SINCICIAL RESPIRATORIO': 'J20.5',
    'BRONQUITE AGUDA NAO ESPECIFICADA': 'J21.9',
    'BRONQUITE CRONICA MISTA, SIMPLES E MUCOPURULENTA': 'J41.8',
    'BRONQUITE CRONICA NAO ESPECIFICADA': 'J42',
    'BRONQUITE CRONICA SIMPLES': 'J41.0',
    'BRONQUITE CRONICA SIMPLES E A MUCOPURULENTA': 'J41',
    'BRONQUITE E PNEUMONITE DEVIDA A PRODUTOS QUIMICOS, GASES, FUMACAS E VAPORES': 'J68.0',
    'BRONQUITE NAO ESPECIFICADA COMO AGUDA OU CRONICA': 'J40',
    'BULIMIA NERVOSA': 'F50.2',
    'BURSITE DA MAO': 'M70.1',
    'BURSITE DO OLECRANO': 'M70.2',
    'BURSITE DO OMBRO': 'M75.5',
    'BURSITE PRE-PATELAR': 'M70.4',
    'BURSITE REUMATOIDE': 'M06.2',
    'BURSITE TIBIAL COLATERAL [PELLEGRINI-STIEDA]': 'M76.4',
    'BURSITE TROCANTERICA': 'M70.6',
    'BURSOPATIA NAO ESPECIFICADA': 'M71.9',
    'CAIBRAS DEVIDAS AO CALOR': 'T67.2',
    'CAIBRAS E ESPASMOS': 'R25.2',
    'CALAZIO': 'H00.1',
    'CALCIFICACAO E OSSIFICACAO DO MUSCULO': 'M61',
    'CALCULO DO TRATO URINARIO INFERIOR, PORCAO NAO ESPECIFICADA': 'N21.9',
    'CALCULO URETRAL': 'N21.1',
    'CALCULO URINARIO NA ESQUISTOSSOMOSE [BILHARZIOSE] [SCHISTOSOMIASE]': 'N22.0',
    'CALCULOSE DA VESICULA BILIAR COM COLICISTITE AGUDA': 'K80.0',
    'CALCULOSE DA VESICULA BILIAR COM OUTRAS FORMAS DE COLECISTITE': 'K80.1',
    'CALCULOSE DA VESICULA BILIAR SEM COLECISTITE': 'K80.2',
    'CALCULOSE DE VIA BILIAR COM COLANGITE': 'K80.3',
    'CALCULOSE DE VIA BILIAR COM COLECISTITE': 'K80.4',
    'CALCULOSE DE VIA BILIAR SEM COLANGITE OU COLECISTITE': 'K80.5',
    'CALCULOSE DO RIM': 'N20.0',
    'CALCULOSE DO RIM COM CALCULO DO URETER': 'N20.2',
    'CALCULOSE DO RIM E DO URETER': 'N20',
    'CALCULOSE DO TRATO URINARIO EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N22.8',
    'CALCULOSE DO TRATO URINARIO INFERIOR': 'N21',
    'CALCULOSE DO TRATO URINARIO INFERIOR EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N22',
    'CALCULOSE DO URETER': 'N20.1',
    'CALCULOSE NA BEXIGA': 'N21.0',
    'CALCULOSE URINARIA, NAO ESPECIFICADA': 'N20.9',
    'CALOS E CALOSIDADES': 'L84',
    'CANCRO MOLE': 'A57',
    'CANDIDIASE': 'B37',
    'CANDIDIASE DA PELE E DAS UNHAS': 'B37.2',
    'CANDIDIASE DA VULVA E DA VAGINA': 'B37.3',
    'CANDIDIASE DE OUTRAS LOCALIZACOES': 'B37.8',
    'CANDIDIASE DE OUTRAS LOCALIZACOES UROGENITAIS': 'B37.4',
    'CANDIDIASE NAO ESPECIFICADA': 'B37.9',
    'CANDIDIASE NEONATAL': 'P37.5',
    'CAPSULITE ADESIVA DO OMBRO': 'M75.0',
    'CAQUEXIA': 'R64',
    'CARBUNCULO': 'A22',
    'CARBUNCULO CUTANEO': 'A22.0',
    'CARBUNCULO, FORMA NAO ESPECIFICADA': 'A22.9',
    'CARCINOMA DE CELULAS HEPATICAS': 'C22.0',
    'CARCINOMA IN SITU DA BEXIGA': 'D09.0',
    'CARCINOMA IN SITU DA CAVIDADE ORAL DO ESOFAGO E DO ESTOMAGO': 'D00',
    'CARCINOMA IN SITU DA MAMA': 'D05',
    'CARCINOMA IN SITU DA MAMA, NAO ESPECIFICADO': 'D05.9',
    'CARCINOMA IN SITU DA PELE': 'D04',
    'CARCINOMA IN SITU DA PELE DO COURO CABELUDO E DO PESCOCO': 'D04.4',
    'CARCINOMA IN SITU DA PELE DOS MEMBROS SUPERIORES, INCLUINDO OMBRO': 'D04.6',
    'CARCINOMA IN SITU DA PROSTATA': 'D07.5',
    'CARCINOMA IN SITU DE OUTRAS LOCALIZACOES ESPECIFICADAS': 'D09.7',
    'CARCINOMA IN SITU DE OUTRAS PARTES DO INTESTINO E AS NAO ESPECIFICADAS': 'D01.4',
    'CARCINOMA IN SITU DE OUTROS ORGAOS DIGESTIVOS': 'D01',
    'CARCINOMA IN SITU DE OUTROS ORGAOS ESPECIFICADOS DO APARELHO DIGESTIVO': 'D01.7',
    'CARCINOMA IN SITU DE OUTROS ORGAOS URINARIOS E OS NAO ESPECIFICADOS': 'D09.1',
    'CARCINOMA IN SITU DO ANUS E CANAL ANAL': 'D01.3',
    'CARCINOMA IN SITU DO APARELHO RESPIRATORIO, NAO ESPECIFICADO': 'D02.9',
    'CARCINOMA IN SITU DO COLO DO UTERO, NAO ESPECIFICADO': 'D06.9',
    'CARCINOMA IN SITU DO COLON': 'D01.0',
    'CARCINOMA IN SITU DO ESTOMAGO': 'D00.2',
    'CARCINOMA IN SITU DO FIGADO, VESICULA BILIAR E VIAS BILIARES': 'D01.5',
    'CARCINOMA IN SITU DO OUVIDO MEDIO E DO APARELHO RESPIRATORIO': 'D02',
    'CARCINOMA IN SITU DO RETO': 'D01.2',
    'CARCINOMA IN SITU DOS BRONQUIOS E PULMOES': 'D02.2',
    'CARCINOMA IN SITU DOS LABIOS, CAVIDADE ORAL E FARINGE': 'D00.0',
    'CARCINOMA LOBULAR IN SITU': 'D05.0',
    'CARDIOMEGALIA': 'I51.7',
    'CARDIOMIOPATIA DILATADA': 'I42.0',
    'CARDIOMIOPATIA NAO ESPECIFICADA': 'I42.9',
    'CARDIOMIOPATIAS': 'I42',
    'CARDIOPATIA CIFOESCOLIOTICA': 'I27.1',
    'CARDIOPATIA PULMONAR NAO ESPECIFICADA': 'I27.9',
    'CARIE DENTARIA': 'K02',
    'CARIE DENTARIA, SEM OUTRA ESPECIFICACAO': 'K02.9',
    'CARIES DA DENTINA': 'K02.1',
    'CARIES LIMITADAS AO ESMALTE': 'K02.0',
    'CATARATA NAO ESPECIFICADA': 'H26.9',
    'CATARATA SENIL': 'H25',
    'CATARATA SENIL NAO ESPECIFICADA': 'H25.9',
    'CAUSAS DESCONHECIDAS E NAO ESPECIFICADAS DE MORBIDADE': 'R69',
    'CAXUMBA (PAROTIDITE EPIDEMICA)': 'B26',
    'CAXUMBA [PAROTIDITE EPIDEMICA] COM OUTRAS COMPLICACOES': 'B26.8',
    'CAXUMBA [PAROTIDITE EPIDEMICA] SEM COMPLICACOES': 'B26.9',
    'CEFALEIA': 'R51',
    'CEFALEIA CAUSADA POR ANESTESIA RAQUIDIANA OU PERIDURAL ADMINISTRADAS DURANTE A GRAVIDEZ': 'O29.4',
    'CEFALEIA CRONICA POS-TRAUMATICA': 'G44.3',
    'CEFALEIA INDUZIDA POR DROGAS, NAO CLASSIFICADA EM OUTRA PARTE': 'G44.4',
    'CEFALEIA POS-ANESTESIA RAQUIDIANA E PERIDURAL, DURANTE O TRABALHO DE PARTO E PARTO': 'O74.5',
    'CEFALEIA PROVOCADA POR UMA ANESTESIA RAQUIDIANA OU PERIDURAL, DURANTE O PUERPERIO': 'O89.4',
    'CEFALEIA TENSIONAL': 'G44.2',
    'CEFALEIA VASCULAR, NAO CLASSIFICADA EM OUTRA PARTE': 'G44.1',
    'CEFALO-HEMATOMA DEVIDO A TRAUMATISMO DE PARTO': 'P12.0',
    'CEGUEIRA E VISAO SUBNORMAL': 'H54',
    'CEGUEIRA EM UM OLHO': 'H54.4',
    'CEGUEIRA EM UM OLHO E VISAO SUBNORMAL EM OUTRO': 'H54.1',
    'CEGUEIRA, AMBOS OS OLHOS': 'H54.0',
    'CELULITE (FLEGMAO)': 'L03',
    'CELULITE DA FACE': 'L03.2',
    'CELULITE DE DEDOS DAS MAOS E DOS PES': 'L03.0',
    'CELULITE DE OUTRAS PARTES DO(S) MEMBRO(S)': 'L03.1',
    'CELULITE DE OUTROS LOCAIS': 'L03.8',
    'CELULITE DO OUVIDO EXTERNO': 'H60.1',
    'CELULITE DO TRONCO': 'L03.3',
    'CELULITE E ABSCESSO DA BOCA': 'K12.2',
    'CELULITE NAO ESPECIFICADA': 'L03.9',
    'CERATITE': 'H16',
    'CERATITE E CERATOCONJUNTIVITE PELO VIRUS DO HERPES SIMPLES': 'H19.1',
    'CERATITE NAO ESPECIFICADA': 'H16.9',
    'CERATITES INTERSTICIAL E PROFUNDA': 'H16.3',
    'CERATOCONE': 'H18.6',
    'CERATOCONJUNTIVITE': 'H16.2',
    'CERATOCONJUNTIVITE DEVIDA A ADENOVIRUS': 'B30.0',
    'CERATOSE ACTINICA': 'L57.0',
    'CERATOSE ADQUIRIDA [CERATODERMIA] PALMAR E PLANTAR': 'L85.1',
    'CERATOSE FOLICULAR ADQUIRIDA': 'L11.0',
    'CERATOSE PUNCTATA (PALMAR E PLANTAR)': 'L85.2',
    'CERATOSE SEBORREICA': 'L82',
    'CERUME IMPACTADO': 'H61.2',
    'CERVICALGIA': 'M54.2',
    'CHOQUE ANAFILATICO DEVIDO A INTOLERANCIA ALIMENTAR': 'T78.0',
    'CHOQUE ANAFILATICO NAO ESPECIFICADO': 'T78.2',
    'CHOQUE CARDIOGENICO': 'R57.0',
    'CHOQUE HIPOVOLEMICO': 'R57.1',
    'CHOQUE NAO CLASSIFICADO EM OUTRA PARTE': 'R57',
    'CHOQUE NAO ESPECIFICADO': 'R57.9',
    'CHOQUE TRAUMATICO': 'T79.4',
    'CIANOSE': 'R23.0',
    'CIATICA': 'M54.3',
    'CICATRIZ QUELOIDE': 'L91.0',
    'CICATRIZES DA CONJUNTIVA': 'H11.2',
    'CICATRIZES E FIBROSE CUTANEA': 'L90.5',
    'CICLISTA TRAUMATIZADO EM COLISAO COM OUTRO VEICULO A PEDAL': 'V11',
    'CICLISTA TRAUMATIZADO EM COLISAO COM OUTRO VEICULO NAO-MOTORIZADO': 'V16',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM AUTOMOVEL, PICK-UP OU CAMINHONETE': 'V13',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM OBJETO FIXO OU PARADO': 'V17',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM PEDESTRE OU UM ANIMAL': 'V10',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM TREM OU UM VEICULO FERROVIARIO': 'V15',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM VEICULO A MOTOR DE DUAS OU TRES RODAS': 'V12',
    'CICLISTA TRAUMATIZADO EM COLISAO COM UM VEICULO DE TRANSPORTE PESADO OU UM ONIBUS': 'V14',
    'CICLISTA TRAUMATIZADO EM UM ACIDENTE DE TRANSPORTE SEM COLISAO': 'V18',
    'CICLISTA [QUALQUER] TRAUMATIZADO EM OUTROS ACIDENTES DE TRANSPORTE ESPECIFICADOS': 'V19.8',
    'CICLISTA [QUALQUER] TRAUMATIZADO EM UM ACIDENTE DE TRANSITO NAO ESPECIFICADO': 'V19.9',
    'CIFOSE E LORDOSE': 'M40',
    'CIFOSE POSTURAL': 'M40.0',
    'CIRCUNSTANCIA RELATIVA AS CONDICOES DE TRABALHO': 'Y96',
    'CIRCUNSTANCIA RELATIVA AS CONDICOES NOSOCOMIAIS (HOSPITALARES)': 'Y95',
    'CIRROSE BILIAR PRIMARIA': 'K74.3',
    'CIRROSE BILIAR SECUNDARIA': 'K74.4',
    'CIRROSE BILIAR, SEM OUTRA ESPECIFICACAO': 'K74.5',
    'CIRROSE HEPATICA ALCOOLICA': 'K70.3',
    'CIRURGIA PROFILATICA NAO ESPECIFICADA': 'Z40.9',
    'CISTICERCOSE': 'B69',
    'CISTICERCOSE DO SISTEMA NERVOSO CENTRAL': 'B69.0',
    'CISTICERCOSE NAO ESPECIFICADA': 'B69.9',
    'CISTITE': 'N30',
    'CISTITE AGUDA': 'N30.0',
    'CISTITE INTERSTICIAL (CRONICA)': 'N30.1',
    'CISTITE TUBERCULOSA': 'N33.0',
    'CISTITE, NAO ESPECIFICADA': 'N30.9',
    'CISTO BILIAR': 'K83.5',
    'CISTO DA GLANDULA DE BARTHOLIN': 'N75.0',
    'CISTO DO COLEDOCO': 'Q44.4',
    'CISTO DO RIM, ADQUIRIDO': 'N28.1',
    'CISTO E MUCOCELE DO NARIZ E DO SEIO PARANASAL': 'J34.1',
    'CISTO EPIDERMICO': 'L72.0',
    'CISTO FOLICULAR DO OVARIO': 'N83.0',
    'CISTO FOLICULAR, NAO ESPECIFICADO DA PELE E DO TECIDO SUBCUTANEO': 'L72.9',
    'CISTO OSSEO ANEURISMATICO': 'M85.5',
    'CISTO OSSEO SOLITARIO': 'M85.4',
    'CISTO OVARIANO DE DESENVOLVIMENTO': 'Q50.1',
    'CISTO PILONIDAL': 'L05',
    'CISTO PILONIDAL COM ABSCESSO': 'L05.0',
    'CISTO PILONIDAL SEM ABSCESSO': 'L05.9',
    'CISTO RADICULAR': 'K04.8',
    'CISTO SINOVIAL DO ESPACO POPLITEO [BAKER]': 'M71.2',
    'CISTO SOLITARIO DA MAMA': 'N60.0',
    'CISTO VULVAR': 'N90.7',
    'CISTOCELE': 'N81.1',
    'CISTOS CEREBRAIS': 'G93.0',
    'CISTOS DA IRIS, DO CORPO CILIAR E DA CAMARA ANTERIOR': 'H21.3',
    'CISTOS DA REGIAO BUCAL NAO CLASSIFICADOS EM OUTRA PARTE': 'K09',
    'CISTOS FOLICULARES DA PELE E DO TECIDO SUBCUTANEO': 'L72',
    'CISTOSTOMIA': 'Z93.5',
    'COARTACAO DA AORTA': 'Q25.1',
    'COCCIDIOIDOMICOSE CUTANEA': 'B38.3',
    'COLANGITE': 'K83.0',
    'COLAPSO PULMONAR': 'J98.1',
    'COLECISTITE AGUDA': 'K81.0',
    'COLECISTITE CRONICA': 'K81.1',
    'COLECISTITE, SEM OUTRA ESPECIFICACAO': 'K81.9',
    'COLELITIASE': 'K80',
    'COLERA': 'A00',
    'COLERA DEVIDA A VIBRIO CHOLERAE 01, BIOTIPO CHOLERAE': 'A00.0',
    'COLERA DEVIDA A VIBRIO CHOLERAE 01, BIOTIPO EL TOR': 'A00.1',
    'COLERA NAO ESPECIFICADA': 'A00.9',
    'COLESTEATOMA DO OUVIDO EXTERNO': 'H60.4',
    'COLICA NEFRETICA NAO ESPECIFICADA': 'N23',
    'COLITE AMEBIANA NAO-DISENTERICA': 'A06.2',
    'COLITE ULCERATIVA': 'K51',
    'COLITE ULCERATIVA, SEM OUTRA ESPECIFICACAO': 'K51.9',
    'COLOCACAO E AJUSTAMENTO DE APARELHO NAO ESPECIFICADO': 'Z46.9',
    'COLOCACAO E AJUSTAMENTO DE BRACO ARTIFICIAL (PARCIAL) (TOTAL)': 'Z44.0',
    'COLOCACAO E AJUSTAMENTO DE ILEOSTOMIA E DE OUTROS DISPOSITIVOS INTESTINAIS': 'Z46.5',
    'COLOCACAO E AJUSTAMENTO DE OUTROS APARELHOS': 'Z46',
    'COLOCACAO E AJUSTAMENTO DE OUTROS APARELHOS ESPECIFICADOS': 'Z46.8',
    'COLOCACAO E AJUSTAMENTO DE PROTESE URINARIA': 'Z46.6',
    'COLOSTOMIA': 'Z93.3',
    'COMA HIPOGLICEMICO NAO-DIABETICO': 'E15',
    'COMPLEXO DE SUBLUXACAO (VERTEBRAL)': 'M99.1',
    'COMPLICACAO DO PUERPERIO NAO ESPECIFICADA': 'O90.9',
    'COMPLICACAO MECANICA DE CATETER (DE DEMORA) URINARIO': 'T83.0',
    'COMPLICACAO MECANICA DE CATETER VASCULAR DE DIALISE': 'T82.4',
    'COMPLICACAO MECANICA DE DISPOSITIVO DE FIXACAO INTERNA DE OSSOS DOS MEMBROS': 'T84.1',
    'COMPLICACAO MECANICA DE DISPOSITIVO INTRA-UTERINO (ANTICONCEPCIONAL)': 'T83.3',
    'COMPLICACAO MECANICA DE DISPOSITIVOS PROTETICOS, IMPLANTE E ENXERTO GASTROINTESTINAIS': 'T85.5',
    'COMPLICACAO MECANICA DE ESTIMULADOR ELETRONICO IMPLANTADO NO SISTEMA NERVOSO': 'T85.1',
    'COMPLICACAO MECANICA DE OUTROS DISPOSITIVOS E IMPLANTES URINARIOS': 'T83.1',
    'COMPLICACAO MECANICA DE PROTESE ARTICULAR INTERNA': 'T84.0',
    'COMPLICACAO MECANICA DE PROTESE VALVULAR CARDIACA': 'T82.0',
    'COMPLICACAO NAO ESPECIFICADA DE CUIDADOS MEDICOS E CIRURGICOS': 'T88.9',
    'COMPLICACAO NAO ESPECIFICADA DE DISPOSITIVO PROTETICO, IMPLANTE E ENXERTO ORTOPEDICOS INTERNOS': 'T84.9',
    'COMPLICACAO NAO ESPECIFICADA DE PROCEDIMENTO': 'T81.9',
    'COMPLICACAO NAO ESPECIFICADA SUBSEQUENTE A INFUSAO, TRANSFUSAO E INJECAO TERAPEUTICA': 'T80.9',
    'COMPLICACOES ASSOCIADAS A FECUNDACAO ARTIFICIAL': 'N98',
    'COMPLICACOES CONSEQUENTES A INFUSAO TRANSFUSAO OU INJECAO TERAPEUTICA': 'T80',
    'COMPLICACOES DE ANESTESIA ADMINISTRADA DURANTE A GRAVIDEZ': 'O29',
    'COMPLICACOES DE DISPOSITIVOS PROTETICOS IMPLANTES E ENXERTOS CARDIACOS E VASCULARES': 'T82',
    'COMPLICACOES DE DISPOSITIVOS PROTETICOS IMPLANTES E ENXERTOS GENITURINARIOS INTERNOS': 'T83',
    'COMPLICACOES DE DISPOSITIVOS PROTETICOS IMPLANTES E ENXERTOS ORTOPEDICOS INTERNOS': 'T84',
    'COMPLICACOES DE OUTRAS PARTES REIMPLANTADAS DO CORPO': 'T87.2',
    'COMPLICACOES DE PROCEDIMENTOS NAO CLASSIFICADAS EM OUTRA PARTE': 'T81',
    'COMPLICACOES DO PUERPERIO NAO CLASSIFICADAS EM OUTRA PARTE': 'O90',
    'COMPLICACOES ESPECIFICAS DE GESTACAO MULTIPLA': 'O31',
    'COMPLICACOES NAO ESPECIFICADA DE DISPOSITIVO PROTETICO, IMPLANTE E ENXERTO CARDIACOS E VASCULARES': 'T82.9',
    'COMPLICACOES RELACIONADAS COM A TENTATIVA DE TRANSFERENCIA DO EMBRIAO': 'N98.3',
    'COMPLICACOES VENOSAS NA GRAVIDEZ': 'O22',
    'COMPORTAMENTO SEXUAL DE ALTO RISCO': 'Z72.5',
    'COMPRESSAO NAO ESPECIFICADA DE MEDULA ESPINAL': 'G95.2',
    'COMPRESSAO VENOSA': 'I87.1',
    'COMPRESSOES DAS RAIZES E DOS PLEXOS NERVOSOS EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G55.8',
    'COMPRESSOES DAS RAIZES E DOS PLEXOS NERVOSOS EM OUTRAS DORSOPATIAS': 'G55.3',
    'COMPRESSOES DAS RAIZES E DOS PLEXOS NERVOSOS EM TRANSTORNOS DOS DISCOS INTERVERTEBRAIS': 'G55.1',
    'COMPROMETIMENTO DO PERITONIO EM DOENCAS INFECCIOSAS CLASSIFICADAS EM OUTRA PARTE': 'K67',
    'CONCRECOES APENDICULARES': 'K38.1',
    'CONCUSSAO CEREBRAL': 'S06.0',
    'CONCUSSAO E EDEMA DA MEDULA CERVICAL': 'S14.0',
    'CONDROMALACIA': 'M94.2',
    'CONDROMALACIA DA ROTULA': 'M22.4',
    'CONJUNTIVITE': 'H10',
    'CONJUNTIVITE AGUDA ATOPICA': 'H10.1',
    'CONJUNTIVITE AGUDA NAO ESPECIFICADA': 'H10.3',
    'CONJUNTIVITE CAUSADA POR CLAMIDIAS': 'A74.0',
    'CONJUNTIVITE CRONICA': 'H10.4',
    'CONJUNTIVITE DEVIDA A ADENOVIRUS': 'B30.1',
    'CONJUNTIVITE E DACRIOCISTITE NEONATAIS': 'P39.1',
    'CONJUNTIVITE EM DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'H13.1',
    'CONJUNTIVITE EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H13.2',
    'CONJUNTIVITE MUCOPURULENTA': 'H10.0',
    'CONJUNTIVITE NAO ESPECIFICADA': 'H10.9',
    'CONJUNTIVITE VIRAL': 'B30',
    'CONJUNTIVITE VIRAL NAO ESPECIFICADA': 'B30.9',
    'CONSTIPACAO': 'K59.0',
    'CONTATO COM ABELHAS VESPAS E VESPOES': 'X23',
    'CONTATO COM AGULHA HIPODERMICA': 'W46',
    'CONTATO COM AGULHA HIPODERMICA - AREAS DE COMERCIO E DE SERVICOS': 'W46.5',
    'CONTATO COM AGULHA HIPODERMICA - ESCOLAS, OUTRAS INSTITUICOES E AREAS DE ADMINISTRACAO PUBLICA': 'W46.2',
    'CONTATO COM ANIMAIS OU PLANTAS VENENOSOS SEM ESPECIFICACAO': 'X29',
    'CONTATO COM ARANHAS VENENOSAS': 'X21',
    'CONTATO COM BEBIDAS ALIMENTOS GORDURA E OLEO DE COZINHA QUENTES': 'X10',
    'CONTATO COM CENTOPEIAS E MIRIAPODES VENENOSAS (TROPICAIS)': 'X24',
    'CONTATO COM E EXPOSICAO A DOENCA TRANSMISSIVEL NAO ESPECIFICADA': 'Z20.9',
    'CONTATO COM E EXPOSICAO A INFECCOES DE TRANSMISSAO PREDOMINANTEMENTE SEXUAL': 'Z20.2',
    'CONTATO COM E EXPOSICAO A OUTRAS DOENCAS TRANSMISSIVEIS': 'Z20.8',
    'CONTATO COM E EXPOSICAO A RAIVA': 'Z20.3',
    'CONTATO COM E EXPOSICAO AO VIRUS DA IMUNODEFICIENCIA HUMANA [HIV]': 'Z20.6',
    'CONTATO COM ELEVADORES E INSTRUMENTOS DE TRANSMISSAO NAO CLASSIFICADOS EM OUTRA PARTE': 'W24',
    'CONTATO COM ESCORPIOES': 'X22',
    'CONTATO COM ESPINHOS DE PLANTAS OU COM FOLHAS AGUCADAS': 'W60',
    'CONTATO COM ESPINHOS DE PLANTAS OU COM FOLHAS AGUCADAS - HABITACAO COLETIVA': 'W60.1',
    'CONTATO COM ESPINHOS DE PLANTAS OU COM FOLHAS AGUCADAS - LOCAL NAO ESPECIFICADO': 'W60.9',
    'CONTATO COM ESPINHOS DE PLANTAS OU COM FOLHAS AGUCADAS - RESIDENCIA': 'W60.0',
    'CONTATO COM FACA ESPADA E PUNHAL': 'W26',
    'CONTATO COM FACA, ESPADA E PUNHAL - LOCAL NAO ESPECIFICADO': 'W26.9',
    'CONTATO COM FACA, ESPADA E PUNHAL - OUTROS LOCAIS ESPECIFICADOS': 'W26.8',
    'CONTATO COM FACA, ESPADA E PUNHAL - RESIDENCIA': 'W26.0',
    'CONTATO COM OBJETO CONTUNDENTE INTENCAO NAO DETERMINADA': 'Y29',
    'CONTATO COM OBJETO CORTANTE OU PENETRANTE INTENCAO NAO DETERMINADA': 'Y28',
    'CONTATO COM OUTRAS MAQUINAS E COM AS NAO ESPECIFICADAS - AREAS DE COMERCIO E DE SERVICOS': 'W31.5',
    'CONTATO COM OUTRAS MAQUINAS E COM AS NAO ESPECIFICADAS - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W31.6',
    'CONTATO COM OUTRAS MAQUINAS E COM AS NAO ESPECIFICADAS - RESIDENCIA': 'W31.0',
    'CONTATO COM OUTROS ARTROPODES VENENOSOS': 'X25',
    'CONTATO COM OUTROS ARTROPODES VENENOSOS - HABITACAO COLETIVA': 'X25.1',
    'CONTATO COM OUTROS ARTROPODES VENENOSOS - RUA E ESTRADA': 'X25.4',
    'CONTATO COM OUTROS UTENSILIOS MANUAIS E APARELHOS DOMESTICOS EQUIPADOS COM MOTOR': 'W29',
    'CONTATO COM SERPENTES E LAGARTOS VENENOSOS': 'X20',
    'CONTATO COM VIDRO CORTANTE': 'W25',
    'CONTATO COM VIDRO CORTANTE - LOCAL NAO ESPECIFICADO': 'W25.9',
    'CONTATO COM VIDRO CORTANTE - RESIDENCIA': 'W25.0',
    'CONTATOS COM SERVICOS DE SAUDE POR OUTRAS CIRCUNSTANCIAS ESPECIFICADAS': 'Z76.8',
    'CONTRATURA DE MUSCULO': 'M62.4',
    'CONTUSAO DA COXA': 'S70.1',
    'CONTUSAO DA GARGANTA': 'S10.0',
    'CONTUSAO DA MAMA': 'S20.0',
    'CONTUSAO DA PALPEBRA E DA REGIAO PERIOCULAR': 'S00.1',
    'CONTUSAO DA PAREDE ABDOMINAL': 'S30.1',
    'CONTUSAO DE ARTELHO SEM LESAO DA UNHA': 'S90.1',
    'CONTUSAO DE ARTELHO(S) COM LESAO DA UNHA': 'S90.2',
    'CONTUSAO DE DEDO(S) COM LESAO DA UNHA': 'S60.1',
    'CONTUSAO DE DEDO(S) SEM LESAO DA UNHA': 'S60.0',
    'CONTUSAO DE OUTRAS PARTES DO PUNHO E DA MAO': 'S60.2',
    'CONTUSAO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA PERNA': 'S80.1',
    'CONTUSAO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DO ANTEBRACO': 'S50.1',
    'CONTUSAO DE OUTRAS PARTES E PARTES NAO ESPECIFICADAS DO PE': 'S90.3',
    'CONTUSAO DO COTOVELO': 'S50.0',
    'CONTUSAO DO DORSO E DA PELVE': 'S30.0',
    'CONTUSAO DO GLOBO OCULAR E DOS TECIDOS DA ORBITA': 'S05.1',
    'CONTUSAO DO JOELHO': 'S80.0',
    'CONTUSAO DO OMBRO E DO BRACO': 'S40.0',
    'CONTUSAO DO QUADRIL': 'S70.0',
    'CONTUSAO DO TORAX': 'S20.2',
    'CONTUSAO DO TORNOZELO': 'S90.0',
    'CONTUSAO DOS ORGAOS GENITAIS EXTERNOS': 'S30.2',
    'CONVALESCENCA': 'Z54',
    'CONVALESCENCA APOS CIRURGIA': 'Z54.0',
    'CONVALESCENCA APOS QUIMIOTERAPIA': 'Z54.2',
    'CONVALESCENCA APOS RADIOTERAPIA': 'Z54.1',
    'CONVALESCENCA APOS TRATAMENTO NAO ESPECIFICADO': 'Z54.9',
    'CONVULSOES DISSOCIATIVAS': 'F44.5',
    'CONVULSOES DO RECEM-NASCIDO': 'P90',
    'CONVULSOES FEBRIS': 'R56.0',
    'CONVULSOES NAO CLASSIFICADAS EM OUTRA PARTE': 'R56',
    'COQUELUCHE': 'A37',
    'COQUELUCHE NAO ESPECIFICADA': 'A37.9',
    'COQUELUCHE POR BORDETELLA PERTUSSIS': 'A37.0',
    'CORONAVIRUS, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97.2',
    'CORPO ESTRANHO (ANTIGO) RETIDO CONSEQUENTE A FERIMENTO PERFURANTE DA ORBITA': 'H05.5',
    'CORPO ESTRANHO EM OUTRAS PARTES E PARTES MULTIPLAS DO APARELHO DIGESTIVO': 'T18.8',
    'CORPO ESTRANHO EM OUTRAS PARTES E PARTES MULTIPLAS DO TRATO RESPIRATORIO': 'T17.8',
    'CORPO ESTRANHO EM OUTROS LOCAIS E EM LOCAIS MULTIPLOS DA PARTE EXTERNA DO OLHO': 'T15.8',
    'CORPO ESTRANHO EM PARTE NAO ESPECIFICADA DA REGIAO EXTERNA DO OLHO': 'T15.9',
    'CORPO ESTRANHO EM PARTE NAO ESPECIFICADA DO APARELHO DIGESTIVO': 'T18.9',
    'CORPO ESTRANHO NA BOCA': 'T18.0',
    'CORPO ESTRANHO NA CORNEA': 'T15.0',
    'CORPO ESTRANHO NA FARINGE': 'T17.2',
    'CORPO ESTRANHO NA LARINGE': 'T17.3',
    'CORPO ESTRANHO NA NARINA': 'T17.1',
    'CORPO ESTRANHO NA PARTE EXTERNA DO OLHO': 'T15',
    'CORPO ESTRANHO NA TRAQUEIA': 'T17.4',
    'CORPO ESTRANHO NA VULVA E VAGINA': 'T19.2',
    'CORPO ESTRANHO NO ANUS E RETO': 'T18.5',
    'CORPO ESTRANHO NO APARELHO DIGESTIVO': 'T18',
    'CORPO ESTRANHO NO COLON': 'T18.4',
    'CORPO ESTRANHO NO ESOFAGO': 'T18.1',
    'CORPO ESTRANHO NO ESTOMAGO': 'T18.2',
    'CORPO ESTRANHO NO INTESTINO DELGADO': 'T18.3',
    'CORPO ESTRANHO NO OUVIDO': 'T16',
    'CORPO ESTRANHO NO SACO CONJUNTIVAL': 'T15.1',
    'CORPO ESTRANHO NO SEIO NASAL': 'T17.0',
    'CORPO ESTRANHO NO TRATO GENITURINARIO': 'T19',
    'CORPO ESTRANHO NO TRATO RESPIRATORIO': 'T17',
    'CORPO ESTRANHO NO TRATO RESPIRATORIO, PARTE NAO ESPECIFICADA': 'T17.9',
    'CORPO ESTRANHO NO UTERO [QUALQUER PARTE]': 'T19.3',
    'CORPO ESTRANHO RESIDUAL NO TECIDO MOLE': 'M79.5',
    'CORPO ESTRANHO RETIDO (ANTIGO) INTRA-OCULAR DE NATUREZA MAGNETICA': 'H44.6',
    'CORPO ESTRANHO RETIDO (ANTIGO) INTRA-OCULAR DE NATUREZA NAO-MAGNETICA': 'H44.7',
    'CORPO FLUTUANTE NO JOELHO': 'M23.4',
    'CORROSAO DA BOCA E FARINGE': 'T28.5',
    'CORROSAO DE OUTRAS PARTES DO OLHO E ANEXOS': 'T26.8',
    'CORROSAO DE PRIMEIRO GRAU DO OMBRO E DO MEMBRO SUPERIOR, EXCETO PUNHO E MAO': 'T22.5',
    'CORROSAO DE PRIMEIRO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.5',
    'CORROSAO DE PRIMEIRO GRAU, PARTE DO CORPO NAO ESPECIFICADA': 'T30.5',
    'CORROSAO DE SEGUNDO GRAU DA CABECA E DO PESCOCO': 'T20.6',
    'CORROSAO DE SEGUNDO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.6',
    'CORROSAO DE SEGUNDO GRAU DO TORNOZELO E DO PE': 'T25.6',
    'CORROSAO DE SEGUNDO GRAU DO TRONCO': 'T21.6',
    'CORROSAO DE TERCEIRO GRAU DO PUNHO E DA MAO': 'T23.7',
    'CORROSAO DE TERCEIRO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.7',
    'CORROSAO DO OLHO E ANEXOS, PARTE NAO ESPECIFICADA': 'T26.9',
    'CORROSAO DO PUNHO E DA MAO, GRAU NAO ESPECIFICADO': 'T23.4',
    'CORROSAO DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE, GRAU NAO ESPECIFICADO': 'T24.4',
    'CORROSAO DO TRONCO, GRAU NAO ESPECIFICADO': 'T21.4',
    'CORROSOES CLASSIFICADAS SEGUNDO A EXTENSAO DA SUPERFICIE CORPORAL ATINGIDA': 'T32',
    'CORROSOES MULTIPLAS, GRAU NAO ESPECIFICADO': 'T29.4',
    'CORROSOES MULTIPLAS, SEM MENCIONAR CORROSAO(OES) ULTRAPASSANDO O PRIMEIRO GRAU': 'T29.5',
    'CORTE, PUNCAO, PERFURACAO OU HEMORRAGIA ACIDENTAIS DURANTE CATETERISMO CARDIACO': 'Y60.5',
    'CORTE, PUNCAO, PERFURACAO OU HEMORRAGIA ACIDENTAIS DURANTE HEMODIALISE OU OUTRAS PERFUSOES': 'Y60.2',
    'CORTE, PUNCAO, PERFURACAO OU HEMORRAGIA ACIDENTAIS DURANTE INFUSAO OU TRANSFUSAO': 'Y60.1',
    'CORTE, PUNCAO, PERFURACAO OU HEMORRAGIA ACIDENTAIS DURANTE INJECAO OU VACINACAO (IMUNIZACAO)': 'Y60.3',
    'CORTE, PUNCAO, PERFURACAO OU HEMORRAGIA ACIDENTAIS DURANTE INTERVENCAO CIRURGICA': 'Y60.0',
    'COSTELA CERVICAL': 'Q76.5',
    'COXARTROSE (ARTROSE DO QUADRIL)': 'M16',
    'COXARTROSE NAO ESPECIFICADA': 'M16.9',
    'COXARTROSE PRIMARIA BILATERAL': 'M16.0',
    'CRANIOSSINOSTOSE': 'Q75.0',
    'CRIPTOCOCOSE PULMONAR': 'B45.0',
    'CRIPTOSPORIDIOSE': 'A07.2',
    'CRISE ADDISONIANA': 'E27.2',
    'CRISE DE GRANDE MAL, NAO ESPECIFICADA (COM OU SEM PEQUENO MAL)': 'G40.6',
    'CRISES CIANOTICAS DO RECEM-NASCIDO': 'P28.2',
    'CROMOMICOSE CUTANEA': 'B43.0',
    'CUIDADO MEDICO NAO ESPECIFICADO': 'Z51.9',
    'CUIDADO PALIATIVO': 'Z51.5',
    'CUIDADOS A CISTOSTOMIA': 'Z43.5',
    'CUIDADOS A COLOSTOMIA': 'Z43.3',
    'CUIDADOS A CURATIVOS E SUTURAS CIRURGICAS': 'Z48.0',
    'CUIDADOS A GASTROSTOMIA': 'Z43.1',
    'CUIDADOS A ILEOSTOMIA': 'Z43.2',
    'CUIDADOS A ORIFICIO ARTIFICIAL NAO ESPECIFICADO': 'Z43.9',
    'CUIDADOS A OUTROS ORIFICIOS ARTIFICIAIS DAS VIAS URINARIAS': 'Z43.6',
    'CUIDADOS A TRAQUEOSTOMIA': 'Z43.0',
    'CUIDADOS ENVOLVENDO DIALISE': 'Z49',
    'CUIDADOS ENVOLVENDO O USO DE PROCEDIMENTOS DE REABILITACAO': 'Z50',
    'DACRIOADENITE': 'H04.0',
    'DEDO EM GATILHO': 'M65.3',
    'DEDO(S) DO PE EM MALHO (ADQUIRIDO)': 'M20.4',
    'DEFEITO DE COAGULACAO NAO ESPECIFICADO': 'D68.9',
    'DEFEITO DE CONSOLIDACAO DA FRATURA': 'M84.0',
    'DEFEITOS DO CAMPO VISUAL': 'H53.4',
    'DEFEITOS POR REDUCAO DO MEMBRO INFERIOR': 'Q72',
    'DEFEITOS QUALITATIVOS DAS PLAQUETAS': 'D69.1',
    'DEFICIENCIA ADQUIRIDA DE FATOR DE COAGULACAO': 'D68.4',
    'DEFICIENCIA DE CALCIO DA DIETA': 'E58',
    'DEFICIENCIA DE FERRO': 'E61.1',
    'DEFICIENCIA DE NIACINA [PELAGRA]': 'E52',
    'DEFICIENCIA DE OUTRAS VITAMINAS DO GRUPO B': 'E53',
    'DEFICIENCIA DE OUTROS ELEMENTOS NUTRIENTES': 'E61',
    'DEFICIENCIA DE TIAMINA': 'E51',
    'DEFICIENCIA DE VITAMINA A COM ULCERACAO E XEROSE DA CORNEA': 'E50.3',
    'DEFICIENCIA DE VITAMINA D': 'E55',
    'DEFICIENCIA HEREDITARIA DO FATOR VIII': 'D66',
    'DEFICIENCIA NAO ESPECIFICADA DE VITAMINA B': 'E53.9',
    'DEFICIENCIAS IMUNITARIAS COMBINADAS': 'D81',
    'DEFORMIDADE ADQUIRIDA DA PELVE': 'M95.5',
    'DEFORMIDADE ADQUIRIDA DO NARIZ': 'M95.0',
    'DEFORMIDADE ADQUIRIDA DO PESCOCO': 'M95.3',
    'DEFORMIDADE ADQUIRIDA DO TORAX E DAS COSTELAS': 'M95.4',
    'DEFORMIDADE ADQUIRIDA NAO ESPECIFICADA DE DEDO(S) DO PE': 'M20.6',
    'DEFORMIDADE ADQUIRIDA NAO ESPECIFICADA DE MEMBRO': 'M21.9',
    'DEFORMIDADE CONGENITA DA MAO': 'Q68.1',
    'DEFORMIDADE CONGENITA DO JOELHO': 'Q68.2',
    'DEFORMIDADE EM FLEXAO': 'M21.2',
    'DEFORMIDADE(S) DO(S) DEDO(S) DAS MAOS': 'M20.0',
    'DEFORMIDADES ADQUIRIDAS DOS DEDOS DAS MAOS E DOS PES': 'M20',
    'DEFORMIDADES CONGENITAS DA COLUNA VERTEBRAL': 'Q67.5',
    'DEFORMIDADES CONGENITAS DO PE': 'Q66',
    'DEGENERACAO CEREBRAL SENIL, NAO CLASSIFICADAS EM OUTRA PARTE': 'G31.1',
    'DEGENERACAO DA MACULA E DO POLO POSTERIOR': 'H35.3',
    'DEGENERACAO DA POLPA': 'K04.2',
    'DEGENERACAO ESTRIONIGRICA': 'G23.2',
    'DEGENERACAO GORDUROSA DO FIGADO NAO CLASSIFICADA EM OUTRA PARTE': 'K76.0',
    'DEGENERACAO MULTISSISTEMICA': 'G90.3',
    'DEGENERACOES E DEPOSITOS DA CONJUNTIVA': 'H11.1',
    'DEISCENCIA DE FERIDA CIRURGICA NAO CLASSIFICADA EM OUTRA PARTE': 'T81.3',
    'DELECAO DO BRACO CURTO DO CROMOSSOMO 5': 'Q93.4',
    'DELIRIUM NAO ESPECIFICADO': 'F05.9',
    'DELIRIUM NAO INDUZIDO PELO ALCOOL OU POR OUTRAS SUBSTANCIAS PSICOATIVAS': 'F05',
    'DELIRIUM NAO SUPERPOSTO A UMA DEMENCIA, ASSIM DESCRITO': 'F05.0',
    'DELIRIUM SUPERPOSTO A UMA DEMENCIA': 'F05.1',
    'DEMENCIA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'F02',
    'DEMENCIA EM OUTRAS DOENCAS ESPECIFICADAS CLASSIFICADAS EM OUTRA PARTE': 'F02.8',
    'DEMENCIA NA DOENCA DE ALZHEIMER': 'F00',
    'DEMENCIA NA DOENCA DE ALZHEIMER DE INICIO PRECOCE': 'F00.0',
    'DEMENCIA NA DOENCA DE ALZHEIMER, FORMA ATIPICA OU MISTA': 'F00.2',
    'DEMENCIA NA DOENCA DE CREUTZFELDT-JAKOB': 'F02.1',
    'DEMENCIA NA DOENCA DE PARKINSON': 'F02.3',
    'DEMENCIA NAO ESPECIFICADA': 'F03',
    'DEMENCIA NAO ESPECIFICADA NA DOENCA DE ALZHEIMER': 'F00.9',
    'DEMENCIA VASCULAR': 'F01',
    'DEMENCIA VASCULAR DE INICIO AGUDO': 'F01.0',
    'DEMENCIA VASCULAR NAO ESPECIFICADA': 'F01.9',
    'DEMENCIA VASCULAR SUBCORTICAL': 'F01.2',
    'DENGUE [DENGUE CLASSICO]': 'A90',
    'DENTES IMPACTADOS': 'K01.1',
    'DENTES INCLUSOS': 'K01.0',
    'DENTES INCLUSOS E IMPACTADOS': 'K01',
    'DENTES MANCHADOS': 'K00.3',
    'DEPENDENCIA DE DIALISE RENAL': 'Z99.2',
    'DEPENDENCIA DE MAQUINA E APARELHO CAPACITANTE NAO ESPECIFICADO': 'Z99.9',
    'DEPLECAO DE VOLUME': 'E86',
    'DEPOSITOS NOS DENTES': 'K03.6',
    'DEPRESSAO POS-ESQUIZOFRENICA': 'F20.4',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A ADESIVOS': 'L23.1',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A ALIMENTOS EM CONTATO COM A PELE': 'L23.6',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A CORANTES': 'L23.4',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A COSMETICOS': 'L23.2',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A METAIS': 'L23.0',
    'DERMATITE ALERGICA DE CONTATO DEVIDA A OUTROS PRODUTOS QUIMICOS': 'L23.5',
    'DERMATITE ALERGICA DE CONTATO DEVIDO A DROGAS EM CONTATO COM A PELE': 'L23.3',
    'DERMATITE ALERGICA DE CONTATO DEVIDO A OUTROS AGENTES': 'L23.8',
    'DERMATITE ALERGICA DE CONTATO DEVIDO A PLANTAS, EXCETO ALIMENTOS': 'L23.7',
    'DERMATITE ALERGICA DE CONTATO, DE CAUSA NAO ESPECIFICADA': 'L23.9',
    'DERMATITE ATOPICA': 'L20',
    'DERMATITE ATOPICA, NAO ESPECIFICADA': 'L20.9',
    'DERMATITE DAS FRALDAS': 'L22',
    'DERMATITE DE CONTATO NAO ESPECIFICADA': 'L25',
    'DERMATITE DE CONTATO NAO ESPECIFICADA DEVIDA A ALIMENTOS EM CONTATO COM A PELE': 'L25.4',
    'DERMATITE DE CONTATO NAO ESPECIFICADA DEVIDA A CORANTES': 'L25.2',
    'DERMATITE DE CONTATO NAO ESPECIFICADA DEVIDA A COSMETICOS': 'L25.0',
    'DERMATITE DE CONTATO NAO ESPECIFICADA DEVIDA A OUTROS AGENTES': 'L25.8',
    'DERMATITE DE CONTATO NAO ESPECIFICADA DEVIDA A OUTROS PRODUTOS QUIMICOS': 'L25.3',
    'DERMATITE DE CONTATO NAO ESPECIFICADA, DE CAUSA NAO ESPECIFICADA': 'L25.9',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A ALIMENTOS EM CONTATO COM A PELE': 'L24.6',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A COSMETICOS': 'L24.3',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A DETERGENTES': 'L24.0',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A DROGAS EM CONTATO COM A PELE': 'L24.4',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A OUTROS PRODUTOS QUIMICOS': 'L24.5',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDA A SOLVENTES': 'L24.2',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDO A OUTROS AGENTES': 'L24.8',
    'DERMATITE DE CONTATO POR IRRITANTES DEVIDO A PLANTAS, EXCETO ALIMENTOS': 'L24.7',
    'DERMATITE DE CONTATO POR IRRITANTES, DE CAUSA NAO ESPECIFICADA': 'L24.9',
    'DERMATITE DEVIDA A INGESTAO DE ALIMENTOS': 'L27.2',
    'DERMATITE DEVIDA A OUTRAS SUBSTANCIAS DE USO INTERNO': 'L27.8',
    'DERMATITE DEVIDA A SUBSTANCIA NAO ESPECIFICADA DE USO INTERNO': 'L27.9',
    'DERMATITE DEVIDA A SUBSTANCIAS DE USO INTERNO': 'L27',
    'DERMATITE ESFOLIATIVA': 'L26',
    'DERMATITE FACTICIA': 'L98.1',
    'DERMATITE HERPETIFORME': 'L13.0',
    'DERMATITE INFECTADA': 'L30.3',
    'DERMATITE NAO ESPECIFICADA': 'L30.9',
    'DERMATITE NUMULAR': 'L30.0',
    'DERMATITE PERIORAL': 'L71.0',
    'DERMATITE POR CERCARIAS': 'B65.3',
    'DERMATITE POR FOTOCONTATO [DERMATITE DO BERLOQUE]': 'L56.2',
    'DERMATITE PUSTULAR SUBCORNEANA': 'L13.1',
    'DERMATITE SEBORREICA': 'L21',
    'DERMATITE SEBORREICA INFANTIL': 'L21.1',
    'DERMATITE SEBORREICA, NAO ESPECIFICADA': 'L21.9',
    'DERMATITE VESICULAR DEVIDO AO VIRUS DO HERPES': 'B00.1',
    'DERMATITES ALERGICAS DE CONTATO': 'L23',
    'DERMATITES DE CONTATO POR IRRITANTES': 'L24',
    'DERMATOFITOSE': 'B35',
    'DERMATOFITOSE NAO ESPECIFICADA': 'B35.9',
    'DERMATOPOLIOMIOSITE': 'M33',
    'DERMATOSE PURPURICA PIGMENTADA': 'L81.7',
    'DERMATOSES NAO INFECCIOSAS DA PALPEBRA': 'H01.1',
    'DERRAME ARTICULAR': 'M25.4',
    'DERRAME PERICARDICO (NAO-INFLAMATORIO)': 'I31.3',
    'DERRAME PLEURAL EM AFECCOES CLASSIFICADAS EM OUTRA PARTE': 'J91',
    'DERRAME PLEURAL NAO CLASSIFICADO EM OUTRA PARTE': 'J90',
    'DESAPARECIMENTO OU FALECIMENTO DE UM MEMBRO DA FAMILIA': 'Z63.4',
    'DESCOLAMENTO DA RETINA COM DEFEITO RETINIANO': 'H33.0',
    'DESCOLAMENTO DA RETINA POR TRACAO': 'H33.4',
    'DESCOLAMENTOS E DEFEITOS DA RETINA': 'H33',
    'DESCONFORTO RESPIRATORIO NAO ESPECIFICADO DO RECEM-NASCIDO': 'P22.9',
    'DESEQUILIBRIO DE CONSTITUINTES DA INGESTAO DE ALIMENTOS': 'E63.1',
    'DESIDRATACAO DO RECEM-NASCIDO': 'P74.1',
    'DESLOCAMENTO DO CRISTALINO': 'H27.1',
    'DESLOCAMENTO E SUBLUXACAO DE ARTICULACAO RECIDIVANTES': 'M24.4',
    'DESLOCAMENTO E SUBLUXACAO PATOLOGICAS DE ARTICULACAO, NAO CLASSIFICADA EM OUTRA PARTE': 'M24.3',
    'DESLOCAMENTO RECIDIVANTE DA ROTULA': 'M22.0',
    'DESNUTRICAO PROTEICO-CALORICA DE GRAUS MODERADO E LEVE': 'E44',
    'DESNUTRICAO PROTEICO-CALORICA GRAVE NAO ESPECIFICADA': 'E43',
    'DESNUTRICAO PROTEICO-CALORICA LEVE': 'E44.1',
    'DESNUTRICAO PROTEICO-CALORICA MODERADA': 'E44.0',
    'DESNUTRICAO PROTEICO-CALORICA NAO ESPECIFICADA': 'E46',
    'DESORIENTACAO NAO ESPECIFICADA': 'R41.0',
    'DESVIO DO SEPTO NASAL': 'J34.2',
    'DIABETES INSIPIDO': 'E23.2',
    'DIABETES INSIPIDO NEFROGENICO': 'N25.1',
    'DIABETES MELLITUS INSULINO-DEPENDENTE': 'E10',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM CETOACIDOSE': 'E10.1',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMA': 'E10.0',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES CIRCULATORIAS PERIFERICAS': 'E10.5',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES MULTIPLAS': 'E10.7',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES NAO ESPECIFICADAS': 'E10.8',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES NEUROLOGICAS': 'E10.4',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES OFTALMICAS': 'E10.3',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM COMPLICACOES RENAIS': 'E10.2',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - COM OUTRAS COMPLICACOES ESPECIFICADAS': 'E10.6',
    'DIABETES MELLITUS INSULINO-DEPENDENTE - SEM COMPLICACOES': 'E10.9',
    'DIABETES MELLITUS NAO ESPECIFICADO': 'E14',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM CETOACIDOSE': 'E14.1',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMA': 'E14.0',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES CIRCULATORIAS PERIFERICAS': 'E14.5',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES MULTIPLAS': 'E14.7',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES NAO ESPECIFICADAS': 'E14.8',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES NEUROLOGICAS': 'E14.4',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES OFTALMICAS': 'E14.3',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM COMPLICACOES RENAIS': 'E14.2',
    'DIABETES MELLITUS NAO ESPECIFICADO - COM OUTRAS COMPLICACOES ESPECIFICADAS': 'E14.6',
    'DIABETES MELLITUS NAO ESPECIFICADO - SEM COMPLICACOES': 'E14.9',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE': 'E11',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM CETOACIDOSE': 'E11.1',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMA': 'E11.0',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES CIRCULATORIAS PERIFERICAS': 'E11.5',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES MULTIPLAS': 'E11.7',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES NAO ESPECIFICADAS': 'E11.8',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES NEUROLOGICAS': 'E11.4',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES OFTALMICAS': 'E11.3',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM COMPLICACOES RENAIS': 'E11.2',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - COM OUTRAS COMPLICACOES ESPECIFICADAS': 'E11.6',
    'DIABETES MELLITUS NAO-INSULINO-DEPENDENTE - SEM COMPLICACOES': 'E11.9',
    'DIABETES MELLITUS RELACIONADO COM A DESNUTRICAO': 'E12',
    'DIABETES MELLITUS RELACIONADO COM A DESNUTRICAO - SEM COMPLICACOES': 'E12.9',
    'DIARREIA E GASTROENTERITE DE ORIGEM INFECCIOSA PRESUMIVEL': 'A09',
    'DIARREIA FUNCIONAL': 'K59.1',
    'DIARREIA NEONATAL NAO-INFECCIOSA': 'P78.3',
    'DIASTASE DE MUSCULO': 'M62.0',
    'DIFICULDADE NEONATAL NA AMAMENTACAO NO PEITO': 'P92.5',
    'DIFICULDADE PARA ANDAR NAO CLASSIFICADA EM OUTRA PARTE': 'R26.2',
    'DIFICULDADES DE ALIMENTACAO E ERROS NA ADMINISTRACAO DE ALIMENTOS': 'R63.3',
    'DIFTERIA LARINGEA': 'A36.2',
    'DIFTERIA NASOFARINGEA': 'A36.1',
    'DILATACAO CONGENITA DO ESOFAGO': 'Q39.5',
    'DIPLEGIA DOS MEMBROS SUPERIORES': 'G83.0',
    'DIPLOPIA': 'H53.2',
    'DISARTRIA E ANARTRIA': 'R47.1',
    'DISCITE NAO ESPECIFICADA': 'M46.4',
    'DISENTERIA AMEBIANA AGUDA': 'A06.0',
    'DISFAGIA': 'R13',
    'DISFAGIA SIDEROPENICA': 'D50.1',
    'DISFASIA E AFASIA': 'R47.0',
    'DISFONIA': 'R49.0',
    'DISFUNCAO DO LABIRINTO': 'H83.2',
    'DISFUNCAO HIPOTALAMICA NAO CLASSIFICADA EM OUTRA PARTE': 'E23.3',
    'DISFUNCAO NEUROMUSCULAR NAO ESPECIFICADA DA BEXIGA': 'N31.9',
    'DISFUNCAO OVARIANA NAO ESPECIFICADA': 'E28.9',
    'DISFUNCAO SEXUAL NAO CAUSADA POR TRANSTORNO OU DOENCA ORGANICA': 'F52',
    'DISFUNCAO SEXUAL NAO DEVIDA A TRANSTORNO OU A DOENCA ORGANICA, NAO ESPECIFICADA': 'F52.9',
    'DISFUNCAO TESTICULAR': 'E29',
    'DISFUNCAO TESTICULAR NAO ESPECIFICADA': 'E29.9',
    'DISFUNCOES NEUROMUSCULARES DA BEXIGA NAO CLASSIFICADOS EM OUTRA PARTE': 'N31',
    'DISIDROSE [POMPHOLYX]': 'L30.1',
    'DISMENORREIA NAO ESPECIFICADA': 'N94.6',
    'DISMENORREIA PRIMARIA': 'N94.4',
    'DISMENORREIA SECUNDARIA': 'N94.5',
    'DISOSTOSE OCULO-MANDIBULAR': 'Q75.5',
    'DISPAREUNIA': 'N94.1',
    'DISPAREUNIA NAO-ORGANICA': 'F52.6',
    'DISPEPSIA': 'K30',
    'DISPLASIA BRONCOPULMONAR ORIGINADA NO PERIODO PERINATAL': 'P27.1',
    'DISPLASIA CERVICAL GRAVE, NAO CLASSIFICADA EM OUTRA PARTE': 'N87.2',
    'DISPLASIA ESPONDILOEPIFISARIA': 'Q77.7',
    'DISPLASIA VAGINAL MODERADA': 'N89.1',
    'DISPLASIAS MAMARIAS BENIGNAS': 'N60',
    'DISPNEIA': 'R06.0',
    'DISPOSITIVOS (APARELHOS) ORTOPEDICOS ASSOCIADO A INCIDENTES ADVERSOS': 'Y79',
    'DISPOSITIVOS (APARELHOS) USADOS EM GASTROENTEROLOGIA E EM UROLOGIA ASSOCIADOS A INCIDENTES ADVERSOS': 'Y73',
    'DISTENSAO E ENTORSE DA COLUNA CERVICAL': 'S13.4',
    'DISTENSAO MUSCULAR': 'M62.6',
    'DISTIMIA': 'F34.1',
    'DISTONIA': 'G24',
    'DISTONIA INDUZIDA POR DROGAS': 'G24.0',
    'DISTROFIA MUSCULAR': 'G71.0',
    'DISTROFIA UNGUEAL': 'L60.3',
    'DISTURBIO DE ANSIEDADE SOCIAL DA INFANCIA': 'F93.2',
    'DISTURBIO DE CONDUTA DO TIPO SOCIALIZADO': 'F91.2',
    'DISTURBIO DESAFIADOR E DE OPOSICAO': 'F91.3',
    'DISTURBIO DO SONO, NAO ESPECIFICADO': 'G47.9',
    'DISTURBIO METABOLICO NAO ESPECIFICADO': 'E88.9',
    'DISTURBIO MISTO DO EQUILIBRIO ACIDO-BASICO': 'E87.4',
    'DISTURBIO NAO ESPECIFICADO DO DESENVOLVIMENTO DENTARIO': 'K00.9',
    'DISTURBIO NAO ESPECIFICADO DO METABOLISMO DA BILIRRUBINA': 'E80.7',
    'DISTURBIO NAO ESPECIFICADO DO METABOLISMO DA PURINA E PIRIMIDINA': 'E79.9',
    'DISTURBIO NAO ESPECIFICADO DO METABOLISMO DE LIPOPROTEINAS': 'E78.9',
    'DISTURBIO VISUAL NAO ESPECIFICADO': 'H53.9',
    'DISTURBIOS DA ERUPCAO DENTARIA': 'K00.6',
    'DISTURBIOS DA FALA NAO CLASSIFICADOS EM OUTRA PARTE': 'R47',
    'DISTURBIOS DA SENSIBILIDADE CUTANEA': 'R20',
    'DISTURBIOS DA VOZ': 'R49',
    'DISTURBIOS DE CONDUTA': 'F91',
    'DISTURBIOS DO DESENVOLVIMENTO E DA ERUPCAO DOS DENTES': 'K00',
    'DISTURBIOS DO EQUILIBRIO DE POTASSIO DO RECEM-NASCIDO': 'P74.3',
    'DISTURBIOS DO INICIO E DA MANUTENCAO DO SONO [INSONIAS]': 'G47.0',
    'DISTURBIOS DO METABOLISMO DA PORFIRINA E DA BILIRRUBINA': 'E80',
    'DISTURBIOS DO METABOLISMO DE GLICOPROTEINAS': 'E77',
    'DISTURBIOS DO METABOLISMO DE MINERAIS': 'E83',
    'DISTURBIOS DO METABOLISMO DE PURINA E PIRIMIDINA': 'E79',
    'DISTURBIOS DO METABOLISMO DO CALCIO': 'E83.5',
    'DISTURBIOS DO METABOLISMO DO COBRE': 'E83.0',
    'DISTURBIOS DO OLFATO E DO PALADAR': 'R43',
    'DISTURBIOS DO SONO': 'G47',
    'DISTURBIOS DO SONO POR SONOLENCIA EXCESSIVA [HIPERSONIA]': 'G47.1',
    'DISTURBIOS NA FORMACAO DOS DENTES': 'K00.4',
    'DISTURBIOS VISUAIS': 'H53',
    'DISTURBIOS VISUAIS SUBJETIVOS': 'H53.1',
    'DISURIA': 'R30.0',
    'DIVERTICULO DE MECKEL': 'Q43.0',
    'DIVERTICULO DO ESOFAGO ADQUIRIDO': 'K22.5',
    'DIVERTICULO GASTRICO': 'K31.4',
    'DOADOR DE MEDULA OSSEA': 'Z52.3',
    'DOADOR DE OUTROS ORGAOS OU TECIDOS': 'Z52.8',
    'DOADORES DE ORGAOS E TECIDOS': 'Z52',
    'DOENCA ALCOOLICA DO FIGADO': 'K70',
    'DOENCA ALCOOLICA DO FIGADO, SEM OUTRA ESPECIFICACAO': 'K70.9',
    'DOENCA ATEROSCLEROTICA DO CORACAO': 'I25.1',
    'DOENCA BOLHOSA CRONICA DA INFANCIA': 'L12.2',
    'DOENCA BOLHOSA, NAO ESPECIFICADA': 'L13.9',
    'DOENCA CARDIACA E RENAL HIPERTENSIVA': 'I13',
    'DOENCA CARDIACA E RENAL HIPERTENSIVA COM INSUFICIENCIA CARDIACA (CONGESTIVA)': 'I13.0',
    'DOENCA CARDIACA E RENAL HIPERTENSIVA COM INSUFICIENCIA CARDIACA (CONGESTIVA) E INSUFICIENCIA RENAL': 'I13.2',
    'DOENCA CARDIACA E RENAL HIPERTENSIVA COM INSUFICIENCIA RENAL': 'I13.1',
    'DOENCA CARDIACA E RENAL HIPERTENSIVA, NAO ESPECIFICADA': 'I13.9',
    'DOENCA CARDIACA HIPERTENSIVA': 'I11',
    'DOENCA CARDIACA HIPERTENSIVA COM INSUFICIENCIA CARDIACA (CONGESTIVA)': 'I11.0',
    'DOENCA CARDIACA HIPERTENSIVA SEM INSUFICIENCIA CARDIACA (CONGESTIVA)': 'I11.9',
    'DOENCA CARDIOVASCULAR ATEROSCLEROTICA, DESCRITA DESTA MANEIRA': 'I25.0',
    'DOENCA CARDIOVASCULAR NAO ESPECIFICADA': 'I51.6',
    'DOENCA CAUSADA PELO MOVIMENTO': 'T75.3',
    'DOENCA CELIACA': 'K90.0',
    'DOENCA CEREBROVASCULAR NAO ESPECIFICADA': 'I67.9',
    'DOENCA DA LINGUA, SEM OUTRA ESPECIFICACAO': 'K14.9',
    'DOENCA DA VESICULA BILIAR, SEM OUTRA ESPECIFICACAO': 'K82.9',
    'DOENCA DAS VIAS BILIARES, SEM OUTRA ESPECIFICACAO': 'K83.9',
    'DOENCA DE ALZHEIMER': 'G30',
    'DOENCA DE ALZHEIMER DE INICIO PRECOCE': 'G30.0',
    'DOENCA DE ALZHEIMER DE INICIO TARDIO': 'G30.1',
    'DOENCA DE ALZHEIMER NAO ESPECIFICADA': 'G30.9',
    'DOENCA DE BEHCET': 'M35.2',
    'DOENCA DE CHAGAS': 'B57',
    'DOENCA DE CHAGAS (CRONICA) COM COMPROMETIMENTO CARDIACO': 'B57.2',
    'DOENCA DE CHAGAS (CRONICA) COM COMPROMETIMENTO DE OUTROS ORGAOS': 'B57.5',
    'DOENCA DE CHAGAS (CRONICA) COM COMPROMETIMENTO DO APARELHO DIGESTIVO': 'B57.3',
    'DOENCA DE CREUTZFELDT-JAKOB': 'A81.0',
    'DOENCA DE CROHN (ENTERITE REGIONAL)': 'K50',
    'DOENCA DE CROHN DE LOCALIZACAO NAO ESPECIFICADA': 'K50.9',
    'DOENCA DE CROHN DO INTESTINO DELGADO': 'K50.0',
    'DOENCA DE CROHN DO INTESTINO GROSSO': 'K50.1',
    'DOENCA DE GLANDULA SALIVAR, SEM OUTRA ESPECIFICACAO': 'K11.9',
    'DOENCA DE HODGKIN': 'C81',
    'DOENCA DE HODGKIN, NAO ESPECIFICADA': 'C81.9',
    'DOENCA DE KIENBOCK DO ADULTO': 'M93.1',
    'DOENCA DE MENIERE': 'H81.0',
    'DOENCA DE PARKINSON': 'G20',
    'DOENCA DE REFLUXO GASTROESOFAGICO': 'K21',
    'DOENCA DE REFLUXO GASTROESOFAGICO COM ESOFAGITE': 'K21.0',
    'DOENCA DE REFLUXO GASTROESOFAGICO SEM ESOFAGITE': 'K21.9',
    'DOENCA DE STILL DO ADULTO': 'M06.1',
    'DOENCA DE VON WILLEBRAND': 'D68.0',
    'DOENCA DEGENERATIVA DO SISTEMA NERVOSO, NAO ESPECIFICADA': 'G31.9',
    'DOENCA DISSEMINADA DEVIDA AO VIRUS DO HERPES': 'B00.7',
    'DOENCA DIVERTICULAR CONCOMITANTE DOS INTESTINOS DELGADO E GROSSO SEM PERFURACAO OU ABSCESSO': 'K57.5',
    'DOENCA DIVERTICULAR DO INTESTINO': 'K57',
    'DOENCA DIVERTICULAR DO INTESTINO DELGADO COM PERFURACAO E ABSCESSO': 'K57.0',
    'DOENCA DIVERTICULAR DO INTESTINO DELGADO SEM PERFURACAO OU ABSCESSO': 'K57.1',
    'DOENCA DIVERTICULAR DO INTESTINO GROSSO COM PERFURACAO E ABSCESSO': 'K57.2',
    'DOENCA DIVERTICULAR DO INTESTINO GROSSO SEM PERFURACAO OU ABSCESSO': 'K57.3',
    'DOENCA DIVERTICULAR DO INTESTINO, DE LOCALIZACAO NAO ESPECIFICADA, COM PERFURACAO E ABSCESSO': 'K57.8',
    'DOENCA DIVERTICULAR DO INTESTINO, DE LOCALIZACAO NAO ESPECIFICADA, SEM PERFURACAO OU ABSCESSO': 'K57.9',
    'DOENCA DO ANUS E DO RETO, SEM OUTRA ESPECIFICACAO': 'K62.9',
    'DOENCA DO APARELHO DIGESTIVO, SEM OUTRA ESPECIFICACAO': 'K92.9',
    'DOENCA DO APENDICE, SEM OUTRA ESPECIFICACAO': 'K38.9',
    'DOENCA DO ESOFAGO, SEM OUTRA ESPECIFICACAO': 'K22.9',
    'DOENCA DO INTESTINO, SEM OUTRA ESPECIFICACAO': 'K63.9',
    'DOENCA DO METABOLISMO DO FERRO': 'E83.1',
    'DOENCA DO NEURONIO MOTOR': 'G12.2',
    'DOENCA DO PANCREAS, SEM OUTRA ESPECIFICACAO': 'K86.9',
    'DOENCA DOS LEGIONARIOS': 'A48.1',
    'DOENCA DOS LEGIONARIOS NAO-PNEUMONICA [FEBRE DE PONTIAC]': 'A48.2',
    'DOENCA DOS MAXILARES, SEM OUTRA ESPECIFICACAO': 'K10.9',
    'DOENCA DOS TECIDOS DUROS DOS DENTES, NAO ESPECIFICADA': 'K03.9',
    'DOENCA HEPATICA TOXICA': 'K71',
    'DOENCA HEPATICA TOXICA COM COLESTASE': 'K71.0',
    'DOENCA HEPATICA TOXICA COM FIBROSE E CIRROSE HEPATICAS': 'K71.7',
    'DOENCA HEPATICA TOXICA COM HEPATITE AGUDA': 'K71.2',
    'DOENCA HEPATICA TOXICA COM HEPATITE CRONICA PERSISTENTE': 'K71.3',
    'DOENCA HEPATICA TOXICA COM HEPATITE NAO CLASSIFICADA EM OUTRA PARTE': 'K71.6',
    'DOENCA HEPATICA TOXICA COM OUTROS TRANSTORNOS DO FIGADO': 'K71.8',
    'DOENCA HEPATICA TOXICA, SEM OUTRA ESPECIFICACAO': 'K71.9',
    'DOENCA HEPATICA, SEM OUTRA ESPECIFICACAO': 'K76.9',
    'DOENCA INFLAMATORIA AGUDA DO UTERO': 'N71.0',
    'DOENCA INFLAMATORIA DO COLO DO UTERO': 'N72',
    'DOENCA INFLAMATORIA DO UTERO EXCETO O COLO': 'N71',
    'DOENCA INFLAMATORIA NAO ESPECIFICADA DA PELVE FEMININA': 'N73.9',
    'DOENCA INFLAMATORIA NAO ESPECIFICADA DA PROSTATA': 'N41.9',
    'DOENCA INFLAMATORIA NAO ESPECIFICADA DO UTERO': 'N71.9',
    'DOENCA INTESTINAL NAO ESPECIFICADA POR PROTOZOARIOS': 'A07.9',
    'DOENCA ISQUEMICA AGUDA DO CORACAO NAO ESPECIFICADA': 'I24.9',
    'DOENCA ISQUEMICA CRONICA DO CORACAO': 'I25',
    'DOENCA MIELOPROLIFERATIVA CRONICA': 'D47.1',
    'DOENCA NAO ESPECIFICADA DA GLANDULA DE BARTHOLIN': 'N75.9',
    'DOENCA NAO ESPECIFICADA DA MEDULA ESPINAL': 'G95.9',
    'DOENCA NAO ESPECIFICADA DA VALVA MITRAL': 'I05.9',
    'DOENCA NAO ESPECIFICADA DA VALVA TRICUSPIDE': 'I07.9',
    'DOENCA NAO ESPECIFICADA DAS VIAS AEREAS SUPERIORES': 'J39.9',
    'DOENCA NAO ESPECIFICADA DO SANGUE E DOS ORGAOS HEMATOPOETICOS': 'D75.9',
    'DOENCA NAO ESPECIFICADA DOS VASOS PULMONARES': 'I28.9',
    'DOENCA PARASITARIA NAO ESPECIFICADA': 'B89',
    'DOENCA PELO HIV RESULTANDO EM CANDIDIASE': 'B20.4',
    'DOENCA PELO HIV RESULTANDO EM DOENCA INFECCIOSA OU PARASITARIA NAO ESPECIFICADA': 'B20.9',
    'DOENCA PELO HIV RESULTANDO EM INFECCOES MICOBACTERIANAS': 'B20.0',
    'DOENCA PELO HIV RESULTANDO EM OUTRA AFECCOES ESPECIFICADAS': 'B23.8',
    'DOENCA PELO HIV RESULTANDO EM OUTRAS DOENCAS INFECCIOSAS E PARASITARIAS': 'B20.8',
    'DOENCA PELO HIV RESULTANDO EM OUTRAS INFECCOES BACTERIANAS': 'B20.1',
    'DOENCA PELO HIV RESULTANDO EM OUTRAS INFECCOES VIRAIS': 'B20.3',
    'DOENCA PELO HIV RESULTANDO EM OUTRAS MICOSES': 'B20.5',
    'DOENCA PELO HIV RESULTANDO EM PNEUMONIA POR PNEUMOCYSTIS JIROVECII': 'B20.6',
    'DOENCA PELO VIRUS DA IMUNODEFICIENCIA HUMANA [HIV] NAO ESPECIFICADA': 'B24',
    'DOENCA PERIODONTAL, SEM OUTRA ESPECIFICACAO': 'K05.6',
    'DOENCA POR ARRANHADURA DO GATO': 'A28.1',
    'DOENCA POR CITOMEGALOVIRUS': 'B25',
    'DOENCA PULMONAR INTERSTICIAL NAO ESPECIFICADAS': 'J84.9',
    'DOENCA PULMONAR OBSTRUTIVA CRONICA COM EXACERBACAO AGUDA NAO ESPECIFICADA': 'J44.1',
    'DOENCA PULMONAR OBSTRUTIVA CRONICA COM INFECCAO RESPIRATORIA AGUDA DO TRATO RESPIRATORIO INFERIOR': 'J44.0',
    'DOENCA PULMONAR OBSTRUTIVA CRONICA NAO ESPECIFICADA': 'J44.9',
    'DOENCA RENAL EM ESTADIO FINAL': 'N18.0',
    'DOENCA RENAL HIPERTENSIVA': 'I12',
    'DOENCA RENAL HIPERTENSIVA COM INSUFICIENCIA RENAL': 'I12.0',
    'DOENCA RENAL HIPERTENSIVA SEM INSUFICIENCIA RENAL': 'I12.9',
    'DOENCA RENAL TUBULO-INTERSTICIAL NAO ESPECIFICADA': 'N15.9',
    'DOENCA RESPIRATORIA CRONICA ORIGINADA NO PERIODO PERINATAL': 'P27',
    'DOENCA VIRAL CONGENITA NAO ESPECIFICADA': 'P35.9',
    'DOENCAS CRONICAS DAS AMIGDALAS E DAS ADENOIDES': 'J35',
    'DOENCAS DA GLANDULA DE BARTHOLIN': 'N75',
    'DOENCAS DA LINGUA': 'K14',
    'DOENCAS DA POLPA E DOS TECIDOS PERIAPICAIS': 'K04',
    'DOENCAS DAS AMIGDALAS E DAS ADENOIDES NAO ESPECIFICADAS': 'J35.9',
    'DOENCAS DAS CORDAS VOCAIS E DA LARINGE NAO CLASSIFICADAS EM OUTRA PARTE': 'J38',
    'DOENCAS DAS GLANDULAS SALIVARES': 'K11',
    'DOENCAS DAS VIAS AEREAS DEVIDA A POEIRAS ORGANICAS ESPECIFICAS': 'J66',
    'DOENCAS DO BACO': 'D73',
    'DOENCAS DO ESTOMAGO E DO DUODENO, SEM OUTRA ESPECIFICACAO': 'K31.9',
    'DOENCAS DO TIMO': 'E32',
    'DOENCAS DOS CAPILARES': 'I78',
    'DOENCAS DOS LABIOS': 'K13.0',
    'DOENCAS EXTRAPIRAMIDAIS E TRANSTORNOS DOS MOVIMENTOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G26',
    'DOENCAS EXTRAPIRAMIDAIS E TRANSTORNOS DOS MOVIMENTOS, NAO ESPECIFICADOS': 'G25.9',
    'DOENCAS IMUNOPROLIFERATIVAS MALIGNAS': 'C88',
    'DOENCAS INFECCIOSAS, OUTRAS E AS NAO ESPECIFICADAS': 'B99',
    'DOENCAS INFLAMATORIAS DA PROSTATA': 'N41',
    'DOENCAS NAO ESPECIFICADAS DOS CAPILARES': 'I78.9',
    'DOENCAS POR VIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34',
    'DOENCAS SEXUALMENTE TRANSMITIDAS, NAO ESPECIFICADAS': 'A64',
    'DOENCAS SISTEMICAS DO TECIDO CONJUNTIVO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M36',
    'DOENCAS VASCULARES PERIFERICAS NAO ESPECIFICADA': 'I73.9',
    'DOR ABDOMINAL E PELVICA': 'R10',
    'DOR AGUDA': 'R52.0',
    'DOR ARTICULAR': 'M25.5',
    'DOR ASSOCIADA A MICCAO': 'R30',
    'DOR CRONICA INTRATAVEL': 'R52.1',
    'DOR DE GARGANTA': 'R07.0',
    'DOR DE GARGANTA E NO PEITO': 'R07',
    'DOR EM MEMBRO': 'M79.6',
    'DOR FACIAL ATIPICA': 'G50.1',
    'DOR LOCALIZADA EM OUTRAS PARTES DO ABDOME INFERIOR': 'R10.3',
    'DOR LOCALIZADA NO ABDOME SUPERIOR': 'R10.1',
    'DOR LOMBAR BAIXA': 'M54.5',
    'DOR NA COLUNA TORACICA': 'M54.6',
    'DOR NAO CLASSIFICADA EM OUTRA PARTE': 'R52',
    'DOR NAO ESPECIFICADA': 'R52.9',
    'DOR OCULAR': 'H57.1',
    'DOR PELVICA E PERINEAL': 'R10.2',
    'DOR PRECORDIAL': 'R07.2',
    'DOR TORACICA AO RESPIRAR': 'R07.1',
    'DOR TORACICA, NAO ESPECIFICADA': 'R07.4',
    'DORSALGIA': 'M54',
    'DORSALGIA NAO ESPECIFICADA': 'M54.9',
    'DORSOPATIA NAO ESPECIFICADA': 'M53.9',
    'DUODENITE': 'K29.8',
    'ECTASIA DE DUTOS MAMARIOS': 'N60.4',
    'ECTROPIO DA PALPEBRA': 'H02.1',
    'ECZEMA HERPETICO': 'B00.0',
    'EDEMA ANGIONEUROTICO': 'T78.3',
    'EDEMA CEREBRAL TRAUMATICO': 'S06.1',
    'EDEMA DA LARINGE': 'J38.4',
    'EDEMA DEVIDO AO CALOR': 'T67.7',
    'EDEMA E PROTEINURIA GESTACIONAIS (INDUZIDOS PELA GRAVIDEZ) SEM HIPERTENSAO': 'O12',
    'EDEMA GENERALIZADO': 'R60.1',
    'EDEMA GESTACIONAL': 'O12.0',
    'EDEMA LOCALIZADO': 'R60.0',
    'EDEMA NAO CLASSIFICADO EM OUTRA PARTE': 'R60',
    'EDEMA NAO ESPECIFICADO': 'R60.9',
    'EDEMA PULMONAR, NAO ESPECIFICADO DE OUTRA FORMA': 'J81',
    'EFEITO ADVERSO NAO ESPECIFICADO': 'T78.9',
    'EFEITO ADVERSO NAO ESPECIFICADO DE DROGA OU MEDICAMENTO': 'T88.7',
    'EFEITO TOXICO DE ALCOOL NAO ESPECIFICADO': 'T51.9',
    'EFEITO TOXICO DE BASES (ALCALIS) CAUSTICAS(OS) E SUBSTANCIAS SEMELHANTES': 'T54.3',
    'EFEITO TOXICO DE CONTATO COM ANIMAIS VENENOSOS': 'T63',
    'EFEITO TOXICO DE CONTATO COM ANIMAL VENENOSO NAO ESPECIFICADO': 'T63.9',
    'EFEITO TOXICO DE CORROSIVOS': 'T54',
    'EFEITO TOXICO DE DERIVADOS DO PETROLEO': 'T52.0',
    'EFEITO TOXICO DE DERIVADOS HALOGENICOS DE HIDROCARBONETOS ALIFATICOS E AROMATICOS NAO ESPECIFICADOS': 'T53.9',
    'EFEITO TOXICO DE FORMALDEIDO': 'T59.2',
    'EFEITO TOXICO DE FRUTOS DO MAR NAO ESPECIFICADOS': 'T61.9',
    'EFEITO TOXICO DE GASES, FUMACAS E VAPORES NAO ESPECIFICADOS': 'T59.9',
    'EFEITO TOXICO DE INGESTAO DE OUTRAS (PARTES DE) PLANTAS': 'T62.2',
    'EFEITO TOXICO DE INSETICIDAS ORGANOFOSFORADOS E CARBAMATOS': 'T60.0',
    'EFEITO TOXICO DE OUTRAS SUBSTANCIAS E AS NAO ESPECIFICADAS': 'T65',
    'EFEITO TOXICO DE OUTRAS SUBSTANCIAS ESPECIFICADAS': 'T65.8',
    'EFEITO TOXICO DE OUTRAS SUBSTANCIAS INORGANICAS': 'T57',
    'EFEITO TOXICO DE OUTRAS SUBSTANCIAS NOCIVAS INGERIDAS COMO ALIMENTO': 'T62.8',
    'EFEITO TOXICO DE OUTROS ALCOOIS': 'T51.8',
    'EFEITO TOXICO DE OUTROS FRUTOS DO MAR': 'T61.8',
    'EFEITO TOXICO DE OUTROS GASES FUMACAS E VAPORES': 'T59',
    'EFEITO TOXICO DE OUTROS INSETICIDAS': 'T60.2',
    'EFEITO TOXICO DE OUTROS SOLVENTES ORGANICOS': 'T52.8',
    'EFEITO TOXICO DE PESTICIDAS': 'T60',
    'EFEITO TOXICO DE RODENTICIDAS': 'T60.4',
    'EFEITO TOXICO DE SABOES E DETERGENTES': 'T55',
    'EFEITO TOXICO DE SUBSTANCIA INORGANICA NAO ESPECIFICADA': 'T57.9',
    'EFEITO TOXICO DE SUBSTANCIA NAO ESPECIFICADA': 'T65.9',
    'EFEITO TOXICO DE TINTURAS E CORANTES, NAO CLASSIFICADAS EM OUTRA PARTE': 'T65.6',
    'EFEITO TOXICO DO CLORO GASOSO': 'T59.4',
    'EFEITO TOXICO DO DICLOROMETANO': 'T53.4',
    'EFEITO TOXICO DO ETANOL': 'T51.0',
    'EFEITO TOXICO DO GAS LACRIMOGENEO': 'T59.3',
    'EFEITO TOXICO DO METANOL': 'T51.1',
    'EFEITO TOXICO DO MONOXIDO DE CARBONO': 'T58',
    'EFEITO TOXICO DO TABACO E DA NICOTINA': 'T65.2',
    'EFEITO TOXICO DO VENENO DE ARANHA': 'T63.3',
    'EFEITO TOXICO DO VENENO DE ESCORPIAO': 'T63.2',
    'EFEITO TOXICO DO VENENO DE OUTROS ARTROPODES': 'T63.4',
    'EFEITO TOXICO DO VENENO DE SERPENTE': 'T63.0',
    'EFEITOS ADVERSOS DA VACINA ANTITETANICA': 'Y58.4',
    'EFEITOS ADVERSOS DA VACINA BCG': 'Y58.0',
    'EFEITOS ADVERSOS DE ADSTRINGENTES E DETERGENTES LOCAIS': 'Y56.2',
    'EFEITOS ADVERSOS DE ANDROGENOS E ANABOLIZANTES CONGENERES': 'Y42.7',
    'EFEITOS ADVERSOS DE ANTAGONISTAS DE ANTICOAGULANTES, VITAMINA K E OUTROS COAGULANTES': 'Y44.3',
    'EFEITOS ADVERSOS DE ANTIBIOTICO SISTEMICO, NAO ESPECIFICADO': 'Y40.9',
    'EFEITOS ADVERSOS DE ANTIBIOTICOS SISTEMICOS': 'Y40',
    'EFEITOS ADVERSOS DE ANTICONCEPCIONAIS [CONTRACEPTIVOS] ORAIS': 'Y42.4',
    'EFEITOS ADVERSOS DE ANTIDIARREICOS': 'Y53.6',
    'EFEITOS ADVERSOS DE ANTIPSICOTICOS E NEUROLEPTICOS FENOTIAZINICOS': 'Y49.3',
    'EFEITOS ADVERSOS DE BENZODIAZEPINICOS': 'Y47.1',
    'EFEITOS ADVERSOS DE DEPRESSORES DO APETITE [ANOREXICOS]': 'Y57.0',
    'EFEITOS ADVERSOS DE DROGA E MEDICAMENTO NAO ESPECIFICADO': 'Y57.9',
    'EFEITOS ADVERSOS DE DROGAS ANTI-RESFRIADO COMUM': 'Y55.5',
    'EFEITOS ADVERSOS DE DROGAS ANTITROMBOTICAS [INIBIDORES DA AGREGACAO DE PLAQUETAS]': 'Y44.4',
    'EFEITOS ADVERSOS DE EXCIPIENTES FARMACEUTICOS': 'Y57.4',
    'EFEITOS ADVERSOS DE INSULINA E HIPOGLICEMICOS ORAIS (ANTIDIABETICOS)': 'Y42.3',
    'EFEITOS ADVERSOS DE LAXATIVOS ESTIMULANTES': 'Y53.2',
    'EFEITOS ADVERSOS DE OUTRAS DROGAS ANTI-HIPERTENSIVAS, NAO CLASSIFICADAS EM OUTRA PARTE': 'Y52.5',
    'EFEITOS ADVERSOS DE OUTRAS DROGAS E MEDICAMENTOS': 'Y57.8',
    'EFEITOS ADVERSOS DE OUTRAS DROGAS E MEDICAMENTOS E AS NAO ESPECIFICADAS': 'Y57',
    'EFEITOS ADVERSOS DE OUTRAS SUBSTANCIAS PSICOTROPICAS, NAO CLASSIFICADOS EM OUTRA PARTE': 'Y49.8',
    'EFEITOS ADVERSOS DE OUTRAS SUBSTANCIAS QUE ATUAM PRIMARIAMENTE SOBRE O APARELHO GASTROINTESTINAL': 'Y53.8',
    'EFEITOS ADVERSOS DE OUTRAS VACINAS BACTERIANAS E AS NAO ESPECIFICADAS': 'Y58.9',
    'EFEITOS ADVERSOS DE OUTRAS VACINAS E SUBSTANCIAS BIOLOGICAS E AS NAO ESPECIFICADAS': 'Y59',
    'EFEITOS ADVERSOS DE OUTRAS VACINAS E SUBSTANCIAS BIOLOGICAS ESPECIFICADAS': 'Y59.8',
    'EFEITOS ADVERSOS DE OUTROS ANTICONVULSIVANTES (ANTIEPILEPTICOS) E OS NAO ESPECIFICADOS': 'Y46.6',
    'EFEITOS ADVERSOS DE PENICILINAS': 'Y40.0',
    'EFEITOS ADVERSOS DE SEDATIVOS HIPNOTICOS E TRANQUILIZANTES (ANSIOLITICOS)': 'Y47',
    'EFEITOS ADVERSOS DE SUBSTANCIA FARMACOLOGICA DE ACAO SISTEMICA, NAO ESPECIFICADA': 'Y43.9',
    'EFEITOS ADVERSOS DE SUBSTANCIA PSICOTROPICA, NAO ESPECIFICADA': 'Y49.9',
    'EFEITOS ADVERSOS DE SUBSTANCIAS DE ACAO PRIMARIAMENTE SISTEMICA': 'Y43',
    'EFEITOS ADVERSOS DE SUBSTANCIAS PSICOTROPICAS NAO CLASSIFICADAS EM OUTRA PARTE': 'Y49',
    'EFEITOS ADVERSOS DE VACINA OU SUBSTANCIA BIOLOGICA, NAO ESPECIFICADA': 'Y59.9',
    'EFEITOS ADVERSOS DE VACINAS ANTIVIRAIS': 'Y59.0',
    'EFEITOS ADVERSOS DE VACINAS BACTERIANAS': 'Y58',
    'EFEITOS ADVERSOS DE VITAMINAS, NAO CLASSIFICADAS EM OUTRA PARTE': 'Y57.7',
    'EFEITOS ADVERSOS NAO CLASSIFICADOS EM OUTRA PARTE': 'T78',
    'EFEITOS DA CORRENTE ELETRICA': 'T75.4',
    'EFEITOS DA FOME': 'T73.0',
    'EFEITOS DO CALOR E DA LUZ': 'T67',
    'EFEITOS DO RUIDO SOBRE O OUVIDO INTERNO': 'H83.3',
    'EJACULACAO PRECOCE': 'F52.4',
    'ELIPTOCITOSE HEREDITARIA': 'D58.1',
    'EMBOLIA E TROMBOSE ARTERIAIS': 'I74',
    'EMBOLIA E TROMBOSE DE ARTERIA NAO ESPECIFICADA': 'I74.9',
    'EMBOLIA E TROMBOSE DE ARTERIAS DOS MEMBROS INFERIORES': 'I74.3',
    'EMBOLIA E TROMBOSE DE ARTERIAS DOS MEMBROS NAO ESPECIFICADAS': 'I74.4',
    'EMBOLIA E TROMBOSE DE ARTERIAS DOS MEMBROS SUPERIORES': 'I74.2',
    'EMBOLIA E TROMBOSE DE OUTRAS ARTERIAS': 'I74.8',
    'EMBOLIA E TROMBOSE DE OUTRAS VEIAS ESPECIFICADAS': 'I82.8',
    'EMBOLIA E TROMBOSE DE VEIA CAVA': 'I82.2',
    'EMBOLIA E TROMBOSE VENOSAS DE VEIA NAO ESPECIFICADA': 'I82.9',
    'EMBOLIA GASOSA (TRAUMATICA)': 'T79.0',
    'EMBOLIA GORDUROSA (TRAUMATICA)': 'T79.1',
    'EMBOLIA PULMONAR': 'I26',
    'EMBOLIA PULMONAR COM MENCAO DE COR PULMONALE AGUDO': 'I26.0',
    'EMBOLIA PULMONAR SEM MENCAO DE COR PULMONALE AGUDO': 'I26.9',
    'EMISSAO DE PRESCRICAO DE REPETICAO': 'Z76.0',
    'ENCEFALITE DEVIDA AO VIRUS DO HERPES': 'B00.4',
    'ENCEFALITE NAO ESPECIFICADA POR VIRUS TRANSMITIDA POR MOSQUITOS': 'A83.9',
    'ENCEFALITE PELO VIRUS DO HERPES ZOSTER': 'B02.0',
    'ENCEFALITE POR CAXUMBA [PAROTIDITE EPIDEMICA]': 'B26.2',
    'ENCEFALITE VIRAL, NAO ESPECIFICADA': 'A86',
    'ENCEFALOPATIA DE WERNICKE': 'E51.2',
    'ENCEFALOPATIA HIPERTENSIVA': 'I67.4',
    'ENCEFALOPATIA NAO ESPECIFICADA': 'G93.4',
    'ENCEFALOPATIA TOXICA': 'G92',
    'ENDOCARDITE AGUDA E SUBAGUDA': 'I33',
    'ENDOCARDITE AGUDA NAO ESPECIFICADA': 'I33.9',
    'ENDOMETRIOSE': 'N80',
    'ENDOMETRIOSE DO OVARIO': 'N80.1',
    'ENDOMETRIOSE DO PERITONIO PELVICO': 'N80.3',
    'ENDOMETRIOSE DO UTERO': 'N80.0',
    'ENDOMETRIOSE NAO ESPECIFICADA': 'N80.9',
    'ENFISEMA': 'J43',
    'ENFISEMA CENTROLOBULAR': 'J43.2',
    'ENFISEMA INTERSTICIAL': 'J98.2',
    'ENFISEMA NAO ESPECIFICADO': 'J43.9',
    'ENFISEMA PANLOBULAR': 'J43.1',
    'ENFISEMA SUBCUTANEO DE ORIGEM TRAUMATICA': 'T79.7',
    'ENFORCAMENTO ESTRANGULAMENTO E SUFOCACAO INTENCAO NAO DETERMINADA': 'Y20',
    'ENTERITE POR CAMPYLOBACTER': 'A04.5',
    'ENTERITE POR ROTAVIRUS': 'A08.0',
    'ENTERITE POR SALMONELA': 'A02.0',
    'ENTEROCOLITE DEVIDA A CLOSTRIDIUM DIFFICILE': 'A04.7',
    'ENTEROCOLITE ULCERATIVA (CRONICA)': 'K51.0',
    'ENTEROVIRUS, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97.1',
    'ENTESOPATIA DO MEMBRO INFERIOR NAO ESPECIFICADA': 'M76.9',
    'ENTESOPATIA NAO ESPECIFICADA': 'M77.9',
    'ENTESOPATIAS DOS MEMBROS INFERIORES EXCLUINDO PE': 'M76',
    'ENTORSE E DISTENSAO DA ARTICULACAO ACROMIOCLAVICULAR': 'S43.5',
    'ENTORSE E DISTENSAO DA ARTICULACAO ESTERNOCLAVICULAR': 'S43.6',
    'ENTORSE E DISTENSAO DA ARTICULACAO SACROILIACA': 'S33.6',
    'ENTORSE E DISTENSAO DA COLUNA LOMBAR': 'S33.5',
    'ENTORSE E DISTENSAO DA COLUNA TORACICA': 'S23.3',
    'ENTORSE E DISTENSAO DAS COSTELAS E DO ESTERNO': 'S23.4',
    'ENTORSE E DISTENSAO DE ARTICULACAO DO OMBRO': 'S43.4',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES DO TORAX E DE PARTES NAO ESPECIFICADAS': 'S23.5',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DA COLUNA LOMBAR E DA PELVE': 'S33.7',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DA MAO': 'S63.7',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DO JOELHO': 'S83.6',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA CINTURA ESCAPULAR': 'S43.7',
    'ENTORSE E DISTENSAO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DO PE': 'S93.6',
    'ENTORSE E DISTENSAO DO COTOVELO': 'S53.4',
    'ENTORSE E DISTENSAO DO MAXILAR': 'S03.4',
    'ENTORSE E DISTENSAO DO PUNHO': 'S63.5',
    'ENTORSE E DISTENSAO DO QUADRIL': 'S73.1',
    'ENTORSE E DISTENSAO DO TORNOZELO': 'S93.4',
    'ENTORSE E DISTENSAO DO(S) ARTELHO(S)': 'S93.5',
    'ENTORSE E DISTENSAO DO(S) DEDO(S)': 'S63.6',
    'ENTORSE E DISTENSAO ENVOLVENDO LIGAMENTO COLATERAL (PERONIAL) (TIBIAL) DO JOELHO': 'S83.4',
    'ENTORSE E DISTENSAO ENVOLVENDO LIGAMENTO CRUZADO (ANTERIOR) (POSTERIOR) DO JOELHO': 'S83.5',
    'ENTROPIO E TRIQUIASE DA PALPEBRA': 'H02.0',
    'ENURESE DE ORIGEM NAO-ORGANICA': 'F98.0',
    'ENVENENAMENTO (INTOXICACAO) ACIDENTAL POR E EXPOSICAO A PESTICIDAS': 'X48',
    'ENVENENAMENTO (INTOXICACAO) ACIDENTAL POR E EXPOSICAO AO ALCOOL': 'X45',
    'ENVENENAMENTO (INTOXICACAO) POR E EXPOSICAO A PESTICIDAS INTENCAO NAO DETERMINADA': 'Y18',
    'ENVOLVIMENTO COM ALCOOL NAO ESPECIFICADO DE OUTRA FORMA': 'Y91.9',
    'ENXAQUECA': 'G43',
    'ENXAQUECA COM AURA [ENXAQUECA CLASSICA]': 'G43.1',
    'ENXAQUECA COMPLICADA': 'G43.3',
    'ENXAQUECA SEM AURA [ENXAQUECA COMUM]': 'G43.0',
    'ENXAQUECA, SEM ESPECIFICACAO': 'G43.9',
    'EOSINOFILIA': 'D72.1',
    'EPICONDILITE LATERAL': 'M77.1',
    'EPICONDILITE MEDIAL': 'M77.0',
    'EPIDERMOLISE BOLHOSA': 'Q81',
    'EPIDERMOLISE BOLHOSA ADQUIRIDA': 'L12.3',
    'EPIGLOTITE AGUDA': 'J05.1',
    'EPILEPSIA': 'G40',
    'EPILEPSIA, NAO ESPECIFICADA': 'G40.9',
    'EPISCLERITE': 'H15.1',
    'EPISODIO DEPRESSIVO GRAVE COM SINTOMAS PSICOTICOS': 'F32.3',
    'EPISODIO DEPRESSIVO GRAVE SEM SINTOMAS PSICOTICOS': 'F32.2',
    'EPISODIO DEPRESSIVO LEVE': 'F32.0',
    'EPISODIO DEPRESSIVO MODERADO': 'F32.1',
    'EPISODIO DEPRESSIVO NAO ESPECIFICADO': 'F32.9',
    'EPISODIO MANIACO': 'F30',
    'EPISODIOS DEPRESSIVOS': 'F32',
    'EPISTAXIS': 'R04.0',
    'EQUIMOSES ESPONTANEAS': 'R23.3',
    'ERISIPELA': 'A46',
    'ERISIPELOIDE': 'A26',
    'ERISIPELOIDE CUTANEO': 'A26.0',
    'ERISIPELOIDE NAO ESPECIFICADO': 'A26.9',
    'ERITEMA E OUTRAS ERUPCOES CUTANEAS NAO ESPECIFICADAS': 'R21',
    'ERITEMA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L54',
    'ERITEMA INFECCIOSO [QUINTA DOENCA]': 'B08.3',
    'ERITEMA MARGINADO': 'L53.2',
    'ERITEMA MARGINADO NA FEBRE REUMATICA AGUDA': 'L54.0',
    'ERITEMA MULTIFORME BOLHOSO': 'L51.1',
    'ERITEMA MULTIFORME NAO BOLHOSO': 'L51.0',
    'ERITEMA MULTIFORME, NAO ESPECIFICADO': 'L51.9',
    'ERITEMA NODOSO': 'L52',
    'ERITEMA POLIMORFO (ERITEMA MULTIFORME)': 'L51',
    'ERITEMA TOXICO': 'L53.0',
    'ERITEMA TOXICO NEONATAL': 'P83.1',
    'ERITRASMA': 'L08.1',
    'EROSAO DENTARIA': 'K03.2',
    'ERUPCAO CUTANEA GENERALIZADA DEVIDA A DROGAS E MEDICAMENTOS': 'L27.0',
    'ERUPCAO CUTANEA LOCALIZADA DEVIDA A DROGAS E MEDICAMENTOS': 'L27.1',
    'ESCABIOSE [SARNA]': 'B86',
    'ESCARLATINA': 'A38',
    'ESCARRO ANORMAL': 'R09.3',
    'ESCLERITE': 'H15.0',
    'ESCLERITE E EPISCLERITE EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H19.0',
    'ESCLEROSE DIFUSA': 'G37.0',
    'ESCLEROSE MULTIPLA': 'G35',
    'ESCLEROSE SISTEMICA': 'M34',
    'ESCLEROSE SISTEMICA NAO ESPECIFICADA': 'M34.9',
    'ESCOLIOSE': 'M41',
    'ESCOLIOSE CONGENITA DEVIDA A MALFORMACAO OSSEA CONGENITA': 'Q76.3',
    'ESCOLIOSE IDIOPATICA INFANTIL': 'M41.0',
    'ESCOLIOSE IDIOPATICA JUVENIL': 'M41.1',
    'ESCOLIOSE NAO ESPECIFICADA': 'M41.9',
    'ESCOLIOSE NEUROMUSCULAR': 'M41.4',
    'ESCOLIOSE TORACOGENICA': 'M41.3',
    'ESFEROCITOSE HEREDITARIA': 'D58.0',
    'ESFEROFAQUIA': 'Q12.4',
    'ESGOTAMENTO': 'Z73.0',
    'ESOFAGITE': 'K20',
    'ESPASMO ANAL': 'K59.4',
    'ESPASMO DO PILORO NAO CLASSIFICADO EM OUTRA PARTE': 'K31.3',
    'ESPASMO HEMIFACIAL CLONICO': 'G51.3',
    'ESPIRRO': 'R06.7',
    'ESPLENOMEGALIA NAO CLASSIFICADA EM OUTRA PARTE': 'R16.1',
    'ESPONDILITE ANCILOSANTE': 'M45',
    'ESPONDILITE POR ENTEROBACTERIAS': 'M49.2',
    'ESPONDILOLISE': 'M43.0',
    'ESPONDILOLISTESE': 'M43.1',
    'ESPONDILOPATIA INFLAMATORIA NAO ESPECIFICADA': 'M46.9',
    'ESPONDILOPATIA NEUROPATICA': 'M49.4',
    'ESPONDILOPATIAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M49',
    'ESPONDILOSE': 'M47',
    'ESPONDILOSE NAO ESPECIFICADA': 'M47.9',
    'ESPORAO DO CALCANEO': 'M77.3',
    'ESPOROTRICOSE': 'B42',
    'ESPOROTRICOSE LINFOCUTANEA': 'B42.1',
    'ESPOROTRICOSE NAO ESPECIFICADA': 'B42.9',
    'ESQUISTOSSOMOSE (BILHARZIOSE) (SCHISTOSOMIASE)': 'B65',
    'ESQUISTOSSOMOSE DEVIDA AO SCHISTOSOMA HAEMATOBIUM [ESQUISTOSSOMOSE URINARIA]': 'B65.0',
    'ESQUISTOSSOMOSE NAO ESPECIFICADA': 'B65.9',
    'ESQUIZOFRENIA': 'F20',
    'ESQUIZOFRENIA CATATONICA': 'F20.2',
    'ESQUIZOFRENIA HEBEFRENICA': 'F20.1',
    'ESQUIZOFRENIA INDIFERENCIADA': 'F20.3',
    'ESQUIZOFRENIA NAO ESPECIFICADA': 'F20.9',
    'ESQUIZOFRENIA PARANOIDE': 'F20.0',
    'ESQUIZOFRENIA SIMPLES': 'F20.6',
    'ESTADIA PROLONGADA EM AMBIENTE AGRAVITACIONAL - HABITACAO COLETIVA': 'X52.1',
    'ESTADO CATATONICO ORGANICO': 'F06.1',
    'ESTADO DA MENOPAUSA E DO CLIMATERIO FEMININO': 'N95.1',
    'ESTADO DE CHOQUE EMOCIONAL E TENSAO, NAO ESPECIFICADO': 'R45.7',
    'ESTADO DE INFECCAO ASSINTOMATICA PELO VIRUS DA IMUNODEFICIENCIA HUMANA [HIV]': 'Z21',
    'ESTADO DE MAL ASMATICO': 'J46',
    'ESTADO DE MAL ENXAQUECOSO': 'G43.2',
    'ESTADO DE MAL EPILEPTICO': 'G41',
    'ESTADO DE MAL EPILEPTICO, NAO ESPECIFICADO': 'G41.9',
    'ESTADO DE PEQUENO MAL EPILEPTICO': 'G41.1',
    'ESTADO DE STRESS POS-TRAUMATICO': 'F43.1',
    'ESTEATORREIA PANCREATICA': 'K90.3',
    'ESTENOSE (DA VALVA) AORTICA': 'I35.0',
    'ESTENOSE (ESTREITAMENTO) URETRAL NAO ESPECIFICADA(O)': 'N35.9',
    'ESTENOSE ADQUIRIDA DO CONDUTO AUDITIVO EXTERNO': 'H61.3',
    'ESTENOSE CONGENITA E ESTREITAMENTO CONGENITO DO ESOFAGO': 'Q39.3',
    'ESTENOSE DA COLUNA VERTEBRAL': 'M48.0',
    'ESTENOSE DA LARINGE': 'J38.6',
    'ESTENOSE DA URETRA': 'N35',
    'ESTENOSE DA VALVA PULMONAR': 'I37.0',
    'ESTENOSE DA VALVA PULMONAR COM INSUFICIENCIA': 'I37.2',
    'ESTENOSE DE DISCO INTERVERTEBRAL DO CANAL MEDULAR': 'M99.5',
    'ESTENOSE DO ANUS E DO RETO': 'K62.4',
    'ESTENOSE E INSUFICIENCIA DOS CANAIS LACRIMAIS': 'H04.5',
    'ESTENOSE MITRAL COM INSUFICIENCIA': 'I05.2',
    'ESTENOSE SUBGLOTICA POS-PROCEDIMENTO': 'J95.5',
    'ESTENOSE URETRAL POS-INFECCIOSA NAO CLASSIFICADA EM OUTRA PARTE': 'N35.1',
    'ESTERILIZACAO': 'Z30.2',
    'ESTOMATITE E LESOES CORRELATAS': 'K12',
    'ESTOMATITE POR CANDIDA': 'B37.0',
    'ESTOMATITE ULCERATIVA NECROTIZANTE': 'A69.0',
    'ESTOMATITE VESICULAR DEVIDA A ENTEROVIRUS COM EXANTEMA': 'B08.4',
    'ESTRABISMO CONVERGENTE CONCOMITANTE': 'H50.0',
    'ESTRABISMO DIVERGENTE CONCOMITANTE': 'H50.1',
    'ESTREPTOCOCOS E ESTAFILOCOCOS COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B95',
    'ESTRIDOR': 'R06.1',
    'ESTRONGILOIDIASE NAO ESPECIFICADA': 'B78.9',
    'ESTUPOR': 'R40.1',
    'EVENTOS QUE ORIGINAM A PERDA DE AUTO-ESTIMA NA INFANCIA': 'Z61.3',
    'EVIDENCIA DE ALCOOLISMO DETERMINADA PELO NIVEL DA INTOXICACAO': 'Y91',
    'EXAME DA PRESSAO ARTERIAL': 'Z01.3',
    'EXAME DE DOADOR POTENCIAL DE ORGAO E TECIDO': 'Z00.5',
    'EXAME DE LABORATORIO': 'Z01.7',
    'EXAME DE ROTINA DE SAUDE DA CRIANCA': 'Z00.1',
    'EXAME DE SAUDE OCUPACIONAL': 'Z10.0',
    'EXAME DE SEGUIMENTO APOS CIRURGIA POR OUTRAS AFECCOES': 'Z09.0',
    'EXAME DE SEGUIMENTO APOS OUTRO TRATAMENTO POR OUTRAS AFECCOES': 'Z09.8',
    'EXAME DE SEGUIMENTO APOS QUIMIOTERAPIA POR NEOPLASIA MALIGNA': 'Z08.2',
    'EXAME DE SEGUIMENTO APOS RADIOTERAPIA POR NEOPLASIA MALIGNA': 'Z08.1',
    'EXAME DE SEGUIMENTO APOS TRATAMENTO DE FRATURA': 'Z09.4',
    'EXAME DE SEGUIMENTO APOS TRATAMENTO DE OUTRAS AFECCOES QUE NAO NEOPLASIAS MALIGNAS': 'Z09',
    'EXAME DE SEGUIMENTO APOS TRATAMENTO NAO ESPECIFICADO POR OUTRAS AFECCOES': 'Z09.9',
    'EXAME DE SEGUIMENTO APOS TRATAMENTO POR NEOPLASIA MALIGNA': 'Z08',
    'EXAME DENTARIO': 'Z01.2',
    'EXAME DO ADOLESCENTE DURANTE O CRESCIMENTO NA PUBERDADE': 'Z00.3',
    'EXAME DOS OLHOS E DA VISAO': 'Z01.0',
    'EXAME DOS OUVIDOS E DA AUDICAO': 'Z01.1',
    'EXAME E OBSERVACAO APOS ACIDENTE DE TRABALHO': 'Z04.2',
    'EXAME E OBSERVACAO APOS ACIDENTE DE TRANSPORTE': 'Z04.1',
    'EXAME E OBSERVACAO APOS ALEGACAO DE ESTUPRO E SEDUCAO': 'Z04.4',
    'EXAME E OBSERVACAO APOS OUTRO ACIDENTE': 'Z04.3',
    'EXAME E OBSERVACAO APOS OUTROS FERIMENTOS INFLIGIDOS': 'Z04.5',
    'EXAME E OBSERVACAO POR OUTRAS RAZOES ESPECIFICADAS': 'Z04.8',
    'EXAME E OBSERVACAO POR RAZAO NAO ESPECIFICADA': 'Z04.9',
    'EXAME ESPECIAL DE RASTREAMENTO (SCREENING) DE OUTROS TRANSTORNOS E DOENCAS': 'Z13',
    'EXAME ESPECIAL DE RASTREAMENTO DE DIABETES MELLITUS': 'Z13.1',
    'EXAME ESPECIAL DE RASTREAMENTO DE DOENCAS CARDIOVASCULARES': 'Z13.6',
    'EXAME ESPECIAL DE RASTREAMENTO DE DOENCAS DOS OUVIDOS E DOS OLHOS': 'Z13.5',
    'EXAME ESPECIAL DE RASTREAMENTO DE DOENCAS INFECCIOSAS INTESTINAIS': 'Z11.0',
    'EXAME ESPECIAL DE RASTREAMENTO DE INFECCOES DE TRANSMISSAO PREDOMINANTEMENTE SEXUAL': 'Z11.3',
    'EXAME ESPECIAL DE RASTREAMENTO DE NEOPLASIA DA PROSTATA': 'Z12.5',
    'EXAME ESPECIAL DE RASTREAMENTO DE NEOPLASIA NAO ESPECIFICADA': 'Z12.9',
    'EXAME ESPECIAL DE RASTREAMENTO DE TUBERCULOSE PULMONAR': 'Z11.1',
    'EXAME ESPECIAL DE RASTREAMENTO DE VIRUS DA IMUNODEFICIENCIA HUMANA [HIV]': 'Z11.4',
    'EXAME ESPECIAL DE RASTREAMENTO NAO ESPECIFICADO': 'Z13.9',
    'EXAME ESPECIAL DE RASTREAMENTO PARA DOENCA INFECCIOSA E PARASITARIA NAO ESPECIFICADA': 'Z11.9',
    'EXAME ESPECIAL NAO ESPECIFICADO': 'Z01.9',
    'EXAME GERAL DE ROTINA (CHECK-UP) DE UMA SUBPOPULACAO DEFINIDA': 'Z10',
    'EXAME GERAL DE ROTINA DE EQUIPE ESPORTIVA': 'Z10.3',
    'EXAME GERAL DE ROTINA DE OUTRA SUBPOPULACAO DEFINIDA': 'Z10.8',
    'EXAME GERAL E INVESTIGACAO DE PESSOAS SEM QUEIXAS OU DIAGNOSTICO RELATADO': 'Z00',
    'EXAME GINECOLOGICO (GERAL) (DE ROTINA)': 'Z01.4',
    'EXAME MEDICO E CONSULTA COM FINALIDADES ADMINISTRATIVAS': 'Z02',
    'EXAME MEDICO GERAL': 'Z00.0',
    'EXAME NAO ESPECIFICADO COM FINALIDADES ADMINISTRATIVAS': 'Z02.9',
    'EXAME NO PERIODO DE CRESCIMENTO RAPIDO NA INFANCIA': 'Z00.2',
    'EXAME OU TESTE DE GRAVIDEZ': 'Z32',
    'EXAME PARA ADMISSAO A INSTITUICAO EDUCACIONAL': 'Z02.0',
    'EXAME PARA COMPARACAO OU DE CONTROLE DE NORMALIDADE NUM PROGRAMA DE INVESTIGACAO CLINICA': 'Z00.6',
    'EXAME PARA PARTICIPACAO EM ESPORTE': 'Z02.5',
    'EXAME PSIQUIATRICO GERAL NAO CLASSIFICADO EM OUTRA PARTE': 'Z00.4',
    'EXAME RADIOLOGICO NAO CLASSIFICADO EM OUTRA PARTE': 'Z01.6',
    'EXANTEMA SUBITO [SEXTA DOENCA]': 'B08.2',
    'EXAUSTAO DEVIDA AO CALOR E A PERDA HIDRICA': 'T67.3',
    'EXAUSTAO DEVIDA AO CALOR, SEM ESPECIFICACAO': 'T67.5',
    'EXAUSTAO DEVIDO A UM ESFORCO INTENSO': 'T73.3',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS': 'X50',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS - AREAS DE COMERCIO E DE SERVICOS': 'X50.5',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'X50.6',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS - HABITACAO COLETIVA': 'X50.1',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS - LOCAL NAO ESPECIFICADO': 'X50.9',
    'EXCESSO DE EXERCICIOS E MOVIMENTOS VIGOROSOS OU REPETITIVOS - RESIDENCIA': 'X50.0',
    'EXFOLIACAO DOS DENTES DEVIDA A CAUSAS SISTEMICAS': 'K08.0',
    'EXPLOSAO DE OUTROS MATERIAIS - RESIDENCIA': 'W40.0',
    'EXPOSICAO A CALOR NATURAL EXCESSIVO': 'X30',
    'EXPOSICAO A CORRENTE ELETRICA NAO ESPECIFICADA': 'W87',
    'EXPOSICAO A CORRENTE ELETRICA NAO ESPECIFICADA - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W87.6',
    'EXPOSICAO A CORRENTE ELETRICA NAO ESPECIFICADA - LOCAL NAO ESPECIFICADO': 'W87.9',
    'EXPOSICAO A CORRENTE ELETRICA NAO ESPECIFICADA - OUTROS LOCAIS ESPECIFICADOS': 'W87.8',
    'EXPOSICAO A FUMACA DE TABACO': 'Z58.7',
    'EXPOSICAO A LINHAS DE TRANSMISSAO DE CORRENTE ELETRICA': 'W85',
    'EXPOSICAO A LUZ SOLAR': 'X32',
    'EXPOSICAO A OUTRA CORRENTE ELETRICA ESPECIFICADA': 'W86',
    'EXPOSICAO A OUTRA CORRENTE ELETRICA ESPECIFICADA - OUTROS LOCAIS ESPECIFICADOS': 'W86.8',
    'EXPOSICAO A OUTRA CORRENTE ELETRICA ESPECIFICADA - RESIDENCIA': 'W86.0',
    'EXPOSICAO A OUTRAS FORCAS MECANICAS ANIMADAS E AS NAO ESPECIFICADAS': 'W64',
    'EXPOSICAO A POLUICAO DA AGUA': 'Z58.2',
    'EXPOSICAO A TIPO NAO ESPECIFICADO DE FUMACA, FOGO OU CHAMAS': 'X09',
    'EXPOSICAO A VIBRACAO': 'W43',
    'EXPOSICAO OCUPACIONAL A FATOR DE RISCO NAO ESPECIFICADO': 'Z57.9',
    'EXPOSICAO OCUPACIONAL A FATORES DE RISCO': 'Z57',
    'EXPOSICAO OCUPACIONAL A OUTROS CONTAMINANTES DO AR': 'Z57.3',
    'EXPOSICAO OCUPACIONAL A OUTROS FATORES DE RISCO': 'Z57.8',
    'EXTRAVASAMENTO DE URINA': 'R39.0',
    'EXTROFIA VESICAL': 'Q64.1',
    'FACILIDADES DE SAUDE NAO DISPONIVEIS OU NAO ACESSIVEIS': 'Z75.3',
    'FADIGA TRANSITORIA DEVIDA AO CALOR': 'T67.6',
    'FALENCIA OU REJEICAO DE TRANSPLANTE DE FIGADO': 'T86.4',
    'FALHA DE SUTURA OU DE LIGADURA DURANTE INTERVENCAO CIRURGICA': 'Y65.2',
    'FALSO TRABALHO DE PARTO ANTES DE SE COMPLETAREM 37 SEMANAS DE GESTACAO': 'O47.0',
    'FALTA DE ALIMENTO': 'X53',
    'FALTA DE ALIMENTO - ESCOLAS, OUTRAS INSTITUICOES E AREAS DE ADMINISTRACAO PUBLICA': 'X53.2',
    'FARINGITE AGUDA': 'J02',
    'FARINGITE AGUDA DEVIDA A OUTROS MICROORGANISMOS ESPECIFICADOS': 'J02.8',
    'FARINGITE AGUDA NAO ESPECIFICADA': 'J02.9',
    'FARINGITE CRONICA': 'J31.2',
    'FARINGITE ESTREPTOCOCICA': 'J02.0',
    'FARINGITE GONOCOCICA': 'A54.5',
    'FARINGITE VESICULAR DEVIDA A ENTEROVIRUS': 'B08.5',
    'FARINGOCONJUNTIVITE VIRAL': 'B30.2',
    'FASCICULACAO': 'R25.3',
    'FASCIITE NECROSANTE': 'M72.6',
    'FEBRE AMARELA NAO ESPECIFICADA': 'A95.9',
    'FEBRE DE CHIKUNGUNYA': 'A92.0',
    'FEBRE DE LASSA': 'A96.2',
    'FEBRE DE ORIGEM DESCONHECIDA E DE OUTRAS ORIGENS': 'R50',
    'FEBRE DE ORIGEM DESCONHECIDA SUBSEQUENTE AO PARTO': 'O86.4',
    'FEBRE EXANTEMATICA POR ENTEROVIRUS [EXANTEMA DE BOSTON]': 'A88.0',
    'FEBRE HEMORRAGICA DE JUNIN': 'A96.0',
    'FEBRE HEMORRAGICA DEVIDA AO VIRUS DO DENGUE': 'A91',
    'FEBRE INDUZIDA POR DROGAS': 'R50.2',
    'FEBRE MACULOSA (RICKETTSIOSES TRANSMITIDAS POR CARRAPATOS)': 'A77',
    'FEBRE MACULOSA NAO ESPECIFICADA': 'A77.9',
    'FEBRE MACULOSA POR RICKETTSIA CONORII': 'A77.1',
    'FEBRE MACULOSA POR RICKETTSIA RICHETTSII': 'A77.0',
    'FEBRE NAO ESPECIFICADA': 'R50.9',
    'FEBRE PARATIFOIDE A': 'A01.1',
    'FEBRE PARATIFOIDE B': 'A01.2',
    'FEBRE PARATIFOIDE C': 'A01.3',
    'FEBRE PARATIFOIDE NAO ESPECIFICADA': 'A01.4',
    'FEBRE Q': 'A78',
    'FEBRE RECORRENTE NAO ESPECIFICADA': 'A68.9',
    'FEBRE RECORRENTE TRANSMITIDA POR CARRAPATOS': 'A68.1',
    'FEBRE RECORRENTE TRANSMITIDA POR PIOLHOS': 'A68.0',
    'FEBRE REUMATICA COM COMPROMETIMENTO DO CORACAO': 'I01',
    'FEBRE REUMATICA SEM MENCAO DE COMPROMETIMENTO DO CORACAO': 'I00',
    'FEBRE TIFOIDE': 'A01.0',
    'FEBRE TRANSMITIDA POR MORDEDURA DE RATO, TIPO NAO ESPECIFICADO': 'A25.9',
    'FEBRE VIRAL TRANSMITIDA POR ARTROPODES, NAO ESPECIFICADA': 'A94',
    'FEBRE VIRAL TRANSMITIDA POR MOSQUITOS, NAO ESPECIFICADA': 'A92.9',
    'FEBRES RECORRENTES (BORRELIOSES)': 'A68',
    'FEBRES TIFOIDE E PARATIFOIDE': 'A01',
    'FEBRES TRANSMITIDAS POR MORDEDURA DE RATO': 'A25',
    'FENDA LABIAL': 'Q36',
    'FERIMENTO DA BOCHECHA E REGIAO TEMPORO-MANDIBULAR': 'S01.4',
    'FERIMENTO DA CABECA': 'S01',
    'FERIMENTO DA COXA': 'S71.1',
    'FERIMENTO DA MAMA': 'S21.0',
    'FERIMENTO DA PALPEBRA E DA REGIAO PERIOCULAR': 'S01.1',
    'FERIMENTO DA PAREDE ABDOMINAL': 'S31.1',
    'FERIMENTO DA PAREDE ANTERIOR DO TORAX': 'S21.1',
    'FERIMENTO DA PAREDE POSTERIOR DO TORAX': 'S21.2',
    'FERIMENTO DA PERNA': 'S81',
    'FERIMENTO DA PERNA, PARTE NAO ESPECIFICADA': 'S81.9',
    'FERIMENTO DA VAGINA E DA VULVA': 'S31.4',
    'FERIMENTO DE DEDO(S) COM LESAO DA UNHA': 'S61.1',
    'FERIMENTO DE DEDO(S) SEM LESAO DA UNHA': 'S61.0',
    'FERIMENTO DE MEMBRO INFERIOR, NIVEL NAO ESPECIFICADO': 'T13.1',
    'FERIMENTO DE OUTRAS PARTES DA PERNA': 'S81.8',
    'FERIMENTO DE OUTRAS PARTES DO ANTEBRACO': 'S51.8',
    'FERIMENTO DE OUTRAS PARTES DO PE': 'S91.3',
    'FERIMENTO DE OUTRAS PARTES DO PUNHO E DA MAO': 'S61.8',
    'FERIMENTO DE OUTRAS PARTES DO TORAX': 'S21.8',
    'FERIMENTO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA CINTURA ESCAPULAR': 'S41.8',
    'FERIMENTO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DO ABDOME': 'S31.8',
    'FERIMENTO DE OUTROS ORGAOS GENITAIS EXTERNOS E OS NAO ESPECIFICADOS': 'S31.5',
    'FERIMENTO DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.1',
    'FERIMENTO DO ABDOME DO DORSO E DA PELVE': 'S31',
    'FERIMENTO DO ANTEBRACO': 'S51',
    'FERIMENTO DO ANTEBRACO, PARTE NAO ESPECIFICADO': 'S51.9',
    'FERIMENTO DO BRACO': 'S41.1',
    'FERIMENTO DO COTOVELO': 'S51.0',
    'FERIMENTO DO COURO CABELUDO': 'S01.0',
    'FERIMENTO DO DORSO E DA PELVE': 'S31.0',
    'FERIMENTO DO ESCROTO E DO TESTICULO': 'S31.3',
    'FERIMENTO DO JOELHO': 'S81.0',
    'FERIMENTO DO LABIO E DA CAVIDADE ORAL': 'S01.5',
    'FERIMENTO DO MEMBRO SUPERIOR, NIVEL NAO ESPECIFICADO': 'T11.1',
    'FERIMENTO DO NARIZ': 'S01.2',
    'FERIMENTO DO OMBRO': 'S41.0',
    'FERIMENTO DO OMBRO E DO BRACO': 'S41',
    'FERIMENTO DO OUVIDO': 'S01.3',
    'FERIMENTO DO PENIS': 'S31.2',
    'FERIMENTO DO PESCOCO': 'S11',
    'FERIMENTO DO PUNHO E DA MAO': 'S61',
    'FERIMENTO DO PUNHO E DA MAO, PARTE NAO ESPECIFICADA': 'S61.9',
    'FERIMENTO DO QUADRIL': 'S71.0',
    'FERIMENTO DO QUADRIL E DA COXA': 'S71',
    'FERIMENTO DO TORAX': 'S21',
    'FERIMENTO DO TORAX PARTE NAO ESPECIFICADA': 'S21.9',
    'FERIMENTO DO TORNOZELO': 'S91.0',
    'FERIMENTO DO TRONCO, NIVEL NAO ESPECIFICADO': 'T09.1',
    'FERIMENTO DO(S) ARTELHO(S) COM LESAO DA UNHA': 'S91.2',
    'FERIMENTO DO(S) ARTELHO(S) SEM LESAO DA UNHA': 'S91.1',
    'FERIMENTO ENVOLVENDO A FARINGE E O ESOFAGO CERVICAL': 'S11.2',
    'FERIMENTO NA CABECA, DE OUTRAS LOCALIZACOES': 'S01.8',
    'FERIMENTO NA CABECA, PARTE NAO ESPECIFICADA': 'S01.9',
    'FERIMENTO PENETRANTE DA ORBITA COM OU SEM CORPO ESTRANHO': 'S05.4',
    'FERIMENTO PENETRANTE DO GLOBO OCULAR COM CORPO ESTRANHO': 'S05.5',
    'FERIMENTO PENETRANTE DO GLOBO OCULAR SEM CORPO ESTRANHO': 'S05.6',
    'FERIMENTOS DE OUTRAS PARTES DO PESCOCO': 'S11.8',
    'FERIMENTOS DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DA CINTURA PELVICA': 'S71.8',
    'FERIMENTOS DO PESCOCO, PARTE NAO ESPECIFICADA': 'S11.9',
    'FERIMENTOS DO TORNOZELO E DO PE': 'S91',
    'FERIMENTOS ENVOLVENDO A CABECA COM O PESCOCO': 'T01.0',
    'FERIMENTOS ENVOLVENDO MULTIPLAS REGIOES DO CORPO': 'T01',
    'FERIMENTOS ENVOLVENDO MULTIPLAS REGIOES DO(S) MEMBRO(S) INFERIOR(ES)': 'T01.3',
    'FERIMENTOS ENVOLVENDO OUTRAS COMBINACOES DE REGIOES DO CORPO': 'T01.8',
    'FERIMENTOS ENVOLVENDO REGIOES MULTIPLAS DO(S) MEMBRO(S) SUPERIOR(ES)': 'T01.2',
    'FERIMENTOS ENVOLVENDO REGIOES MULTIPLAS DO(S) MEMBRO(S) SUPERIOR(ES) COM MEMBRO(S) INFERIOR(ES)': 'T01.6',
    'FERIMENTOS MULTIPLOS DA CABECA': 'S01.7',
    'FERIMENTOS MULTIPLOS DA PERNA': 'S81.7',
    'FERIMENTOS MULTIPLOS DO ABDOME, DO DORSO E DA PELVE': 'S31.7',
    'FERIMENTOS MULTIPLOS DO ANTEBRACO': 'S51.7',
    'FERIMENTOS MULTIPLOS DO OMBRO E DO BRACO': 'S41.7',
    'FERIMENTOS MULTIPLOS DO PESCOCO': 'S11.7',
    'FERIMENTOS MULTIPLOS DO PUNHO E DA MAO': 'S61.7',
    'FERIMENTOS MULTIPLOS DO QUADRIL E DA COXA': 'S71.7',
    'FERIMENTOS MULTIPLOS DO TORNOZELO E DO PE': 'S91.7',
    'FERIMENTOS MULTIPLOS NAO ESPECIFICADOS': 'T01.9',
    'FETO E RECEM-NASCIDO AFETADOS POR COMPLICACOES DA PLACENTA DO CORDAO UMBILICAL E DAS MEMBRANAS': 'P02',
    'FETO E RECEM-NASCIDO AFETADOS POR OUTRAS AFECCOES DO CORDAO UMBILICAL E AS NAO ESPECIFICADAS': 'P02.6',
    'FIBROADENOSE DA MAMA': 'N60.2',
    'FIBROMATOSE DA FASCIA PLANTAR': 'M72.2',
    'FIBROMATOSE DE FASCIA PALMAR [DUPUYTREN]': 'M72.0',
    'FIBROMATOSE PSEUDOSSARCOMATOSA': 'M72.4',
    'FIBROMIALGIA': 'M79.7',
    'FIBROSE CISTICA': 'E84',
    'FIBROSE CISTICA COM MANIFESTACOES PULMONARES': 'E84.0',
    'FIBROSE E CIRROSE HEPATICAS': 'K74',
    'FIBROSE HEPATICA': 'K74.0',
    'FIGADO GORDUROSO ALCOOLICO': 'K70.0',
    'FILARIOSE': 'B74',
    'FILARIOSE NAO ESPECIFICADA': 'B74.9',
    'FILARIOSE POR WUCHERERIA BANCROFTI': 'B74.0',
    'FISSURA ANAL AGUDA': 'K60.0',
    'FISSURA ANAL CRONICA': 'K60.1',
    'FISSURA ANAL, NAO ESPECIFICADA': 'K60.2',
    'FISSURA E FISTULA DAS REGIOES ANAL E RETAL': 'K60',
    'FISSURA E FISTULA DO MAMILO': 'N64.0',
    'FISTULA ANAL': 'K60.3',
    'FISTULA ANORRETAL': 'K60.5',
    'FISTULA ARTERIOVENOSA ADQUIRIDA': 'I77.0',
    'FISTULA ARTICULAR': 'M25.1',
    'FISTULA DO INTESTINO': 'K63.2',
    'FISTULA DO LABIRINTO': 'H83.1',
    'FISTULA ENTERO-VESICAL': 'N32.1',
    'FISTULA RETAL': 'K60.4',
    'FISTULA VESICAL NAO CLASSIFICADA EM OUTRA PARTE': 'N32.2',
    'FISTULAS DO TRATO GENITAL FEMININO': 'N82',
    'FLATULENCIA E AFECCOES CORRELATAS': 'R14',
    'FLEBITE E TROMBOFLEBITE': 'I80',
    'FLEBITE E TROMBOFLEBITE DA VEIA FEMURAL': 'I80.1',
    'FLEBITE E TROMBOFLEBITE DE LOCALIZACAO NAO ESPECIFICADA': 'I80.9',
    'FLEBITE E TROMBOFLEBITE DE OUTRAS LOCALIZACOES': 'I80.8',
    'FLEBITE E TROMBOFLEBITE DE OUTROS VASOS PROFUNDOS DOS MEMBROS INFERIORES': 'I80.2',
    'FLEBITE E TROMBOFLEBITE DOS MEMBROS INFERIORES, NAO ESPECIFICADA': 'I80.3',
    'FLEBITE E TROMBOFLEBITE DOS VASOS SUPERFICIAIS DOS MEMBROS INFERIORES': 'I80.0',
    'FLEBITE E TROMBOFLEBITE INTRACRANIANAS E INTRA-RAQUIDIANAS': 'G08',
    'FLEBOTROMBOSE PROFUNDA NA GRAVIDEZ': 'O22.3',
    'FLUTTER E FIBRILACAO ATRIAL': 'I48',
    'FOBIAS SOCIAIS': 'F40.1',
    'FOLICULITE DESCALVANTE': 'L66.2',
    'FOLICULITE ULERITEMATOSA RETICULADA': 'L66.4',
    'FORMA AGUDA DA DOENCA DE CHAGAS, COM COMPROMETIMENTO CARDIACO': 'B57.0',
    'FRATURA AO NIVEL DO PUNHO E DA MAO': 'S62',
    'FRATURA DA ABOBADA DO CRANIO': 'S02.0',
    'FRATURA DA BASE DO CRANIO': 'S02.1',
    'FRATURA DA CINTURA ESCAPULAR, PARTE NAO ESPECIFICADA': 'S42.9',
    'FRATURA DA CLAVICULA': 'S42.0',
    'FRATURA DA COLUNA LOMBAR E DA PELVE': 'S32',
    'FRATURA DA DIAFISE DA TIBIA': 'S82.2',
    'FRATURA DA DIAFISE DO CUBITO [ULNA]': 'S52.2',
    'FRATURA DA DIAFISE DO FEMUR': 'S72.3',
    'FRATURA DA DIAFISE DO RADIO': 'S52.3',
    'FRATURA DA DIAFISE DO UMERO': 'S42.3',
    'FRATURA DA EXTREMIDADE DISTAL DA TIBIA': 'S82.3',
    'FRATURA DA EXTREMIDADE DISTAL DO RADIO': 'S52.5',
    'FRATURA DA EXTREMIDADE DISTAL DO RADIO E DO CUBITO [ULNA]': 'S52.6',
    'FRATURA DA EXTREMIDADE INFERIOR DO UMERO': 'S42.4',
    'FRATURA DA EXTREMIDADE PROXIMAL DA TIBIA': 'S82.1',
    'FRATURA DA EXTREMIDADE SUPERIOR DO CUBITO [ULNA]': 'S52.0',
    'FRATURA DA EXTREMIDADE SUPERIOR DO RADIO': 'S52.1',
    'FRATURA DA EXTREMIDADE SUPERIOR DO UMERO': 'S42.2',
    'FRATURA DA OMOPLATA [ESCAPULA]': 'S42.1',
    'FRATURA DA PERNA INCLUINDO TORNOZELO': 'S82',
    'FRATURA DA PERNA, PARTE NAO ESPECIFICADA': 'S82.9',
    'FRATURA DA ROTULA [PATELA]': 'S82.0',
    'FRATURA DA SEGUNDA VERTEBRA CERVICAL': 'S12.1',
    'FRATURA DAS DIAFISES DO RADIO E DO CUBITO [ULNA]': 'S52.4',
    'FRATURA DE COSTELA': 'S22.3',
    'FRATURA DE COSTELA(S) ESTERNO E COLUNA TORACICA': 'S22',
    'FRATURA DE DENTES': 'S02.5',
    'FRATURA DE FADIGA (STRESS) NAO CLASSIFICADA EM OUTRA PARTE': 'M84.3',
    'FRATURA DE FADIGA DE VERTEBRA': 'M48.4',
    'FRATURA DE MANDIBULA': 'S02.6',
    'FRATURA DE OSSO SUBSEQUENTE A IMPLANTE ORTOPEDICO, PROTESE ARTICULAR OU PLACA OSSEA': 'M96.6',
    'FRATURA DE OSSOS DO METATARSO': 'S92.3',
    'FRATURA DE OUTRAS PARTES DA COLUNA LOMBOSSACRA E DA PELVE E DE PARTES NAO ESPECIFICADAS': 'S32.8',
    'FRATURA DE OUTRAS PARTES DA PERNA': 'S82.8',
    'FRATURA DE OUTRAS PARTES DO ANTEBRACO': 'S52.8',
    'FRATURA DE OUTRAS PARTES DO OMBRO E DO BRACO': 'S42.8',
    'FRATURA DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DO PUNHO E DA MAO': 'S62.8',
    'FRATURA DE OUTRAS VERTEBRAS CERVICAIS ESPECIFICADAS': 'S12.2',
    'FRATURA DE OUTRO ARTELHO': 'S92.5',
    'FRATURA DE OUTRO(S) OSSO(S) DO CARPO': 'S62.1',
    'FRATURA DE OUTROS DEDOS': 'S62.6',
    'FRATURA DE OUTROS OSSOS DO METACARPO': 'S62.3',
    'FRATURA DE OUTROS OSSOS DO TARSO': 'S92.2',
    'FRATURA DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.2',
    'FRATURA DE VERTEBRA LOMBAR': 'S32.0',
    'FRATURA DE VERTEBRA TORACICA': 'S22.0',
    'FRATURA DO ACETABULO': 'S32.4',
    'FRATURA DO ANTEBRACO': 'S52',
    'FRATURA DO ANTEBRACO, PARTE NAO ESPECIFICADA': 'S52.9',
    'FRATURA DO ASSOALHO ORBITAL': 'S02.3',
    'FRATURA DO ASTRAGALO': 'S92.1',
    'FRATURA DO CALCANEO': 'S92.0',
    'FRATURA DO COCCIX': 'S32.2',
    'FRATURA DO COLO DO FEMUR': 'S72.0',
    'FRATURA DO CRANIO E DOS OSSOS DA FACE': 'S02',
    'FRATURA DO CRANIO OU DOS OSSOS DA FACE, PARTE NAO ESPECIFICADA': 'S02.9',
    'FRATURA DO ESTERNO': 'S22.2',
    'FRATURA DO FEMUR': 'S72',
    'FRATURA DO FEMUR, PARTE NAO ESPECIFICADA': 'S72.9',
    'FRATURA DO HALUX': 'S92.4',
    'FRATURA DO MALEOLO LATERAL': 'S82.6',
    'FRATURA DO MALEOLO MEDIAL': 'S82.5',
    'FRATURA DO MEMBRO INFERIOR, NIVEL NAO ESPECIFICADO': 'T12',
    'FRATURA DO MEMBRO SUPERIOR, NIVEL NAO ESPECIFICADO': 'T10',
    'FRATURA DO OMBRO E DO BRACO': 'S42',
    'FRATURA DO OSSO NAVICULAR [ESCAFOIDE] DA MAO': 'S62.0',
    'FRATURA DO PE (EXCETO DO TORNOZELO)': 'S92',
    'FRATURA DO PE NAO ESPECIFICADA': 'S92.9',
    'FRATURA DO PERONIO [FIBULA]': 'S82.4',
    'FRATURA DO POLEGAR': 'S62.5',
    'FRATURA DO PRIMEIRO METACARPIANO': 'S62.2',
    'FRATURA DO PUBIS': 'S32.5',
    'FRATURA DO SACRO': 'S32.1',
    'FRATURA DOS OSSOS DO TORAX, PARTE NAO ESPECIFICADA': 'S22.9',
    'FRATURA DOS OSSOS MALARES E MAXILARES': 'S02.4',
    'FRATURA DOS OSSOS NASAIS': 'S02.2',
    'FRATURA OSSEA EM DOENCAS NEOPLASICAS': 'M90.7',
    'FRATURA PATOLOGICA NAO CLASSIFICADA EM OUTRA PARTE': 'M84.4',
    'FRATURA PERTROCANTERICA': 'S72.1',
    'FRATURA SUBTROCANTERICA': 'S72.2',
    'FRATURAS DE OUTRAS PARTES DO FEMUR': 'S72.8',
    'FRATURAS DE OUTRAS PARTES DOS OSSOS DO TORAX': 'S22.8',
    'FRATURAS ENVOLVENDO MULTIPLAS REGIOES DO CORPO': 'T02',
    'FRATURAS ENVOLVENDO REGIOES MULTIPLAS DE UM MEMBRO SUPERIOR': 'T02.2',
    'FRATURAS MULTIPLAS DA PERNA': 'S82.7',
    'FRATURAS MULTIPLAS DE COLUNA LOMBAR E DA PELVE': 'S32.7',
    'FRATURAS MULTIPLAS DE COSTELAS': 'S22.4',
    'FRATURAS MULTIPLAS DE DEDO(S)': 'S62.7',
    'FRATURAS MULTIPLAS DO ANTEBRACO': 'S52.7',
    'FRATURAS MULTIPLAS DO FEMUR': 'S72.7',
    'FRATURAS MULTIPLAS DO PE': 'S92.7',
    'FRATURAS MULTIPLAS ENVOLVENDO OS OSSOS DO CRANIO E DA FACE': 'S02.7',
    'FRATURAS MULTIPLAS NAO ESPECIFICADAS': 'T02.9',
    'FUGA DISSOCIATIVA': 'F44.1',
    'GALACTORREIA': 'O92.6',
    'GALACTORREIA NAO-ASSOCIADA AO PARTO': 'N64.3',
    'GANGLIOS': 'M67.4',
    'GANGLIOSIDOSE GM2': 'E75.0',
    'GANGRENA NAO CLASSIFICADA EM OUTRA PARTE': 'R02',
    'GANHO DE PESO ANORMAL': 'R63.5',
    'GASTRITE ALCOOLICA': 'K29.2',
    'GASTRITE ATROFICA CRONICA': 'K29.4',
    'GASTRITE CRONICA, SEM OUTRA ESPECIFICACAO': 'K29.5',
    'GASTRITE E DUODENITE': 'K29',
    'GASTRITE HEMORRAGICA AGUDA': 'K29.0',
    'GASTRITE NAO ESPECIFICADA': 'K29.7',
    'GASTRITE SUPERFICIAL CRONICA': 'K29.3',
    'GASTRODUODENITE, SEM OUTRA ESPECIFICACAO': 'K29.9',
    'GASTROENTERITE E COLITE ALERGICAS OU LIGADAS A DIETA': 'K52.2',
    'GASTROENTERITE E COLITE DEVIDA A RADIACAO': 'K52.0',
    'GASTROENTERITE E COLITE NAO-INFECCIOSAS, NAO ESPECIFICADAS': 'K52.9',
    'GASTROENTERITE E COLITE TOXICAS': 'K52.1',
    'GASTROENTEROPATIA AGUDA PELO AGENTE DE NORWALK': 'A08.1',
    'GASTROSTOMIA': 'Z93.1',
    'GELADURA COM NECROSE DE TECIDOS': 'T34',
    'GELADURA SUPERFICIAL DO JOELHO E DA PERNA': 'T33.7',
    'GELADURA SUPERFICIAL DO PUNHO E DA MAO': 'T33.5',
    'GELADURA SUPERFICIAL DO TORNOZELO E DO PE': 'T33.8',
    'GELADURA, COM NECROSE DE TECIDOS, DE LOCALIZACAO NAO ESPECIFICADA': 'T34.9',
    'GELADURA, COM NECROSE DE TECIDOS, DO JOELHO E DA PERNA': 'T34.7',
    'GELADURA, COM NECROSE DE TECIDOS, DO TORNOZELO E DO PE': 'T34.8',
    'GELADURA, DE GRAU NAO ESPECIFICADO, DO MEMBRO SUPERIOR': 'T35.4',
    'GENGIVITE AGUDA': 'K05.0',
    'GENGIVITE CRONICA': 'K05.1',
    'GENGIVOESTOMATITE E FARINGOAMIGDALITE DEVIDA AO VIRUS DO HERPES': 'B00.2',
    'GESTACAO MULTIPLA, NAO ESPECIFICADA': 'O30.9',
    'GIARDIASE [LAMBLIASE]': 'A07.1',
    'GLAUCOMA': 'H40',
    'GLAUCOMA NAO ESPECIFICADO': 'H40.9',
    'GLAUCOMA PRIMARIO DE ANGULO FECHADO': 'H40.2',
    'GLAUCOMA SECUNDARIO A OUTROS TRANSTORNOS DO OLHO': 'H40.5',
    'GLAUCOMA SECUNDARIO A TRAUMATISMO OCULAR': 'H40.3',
    'GLOSSITE': 'K14.0',
    'GLOSSODINIA': 'K14.6',
    'GOLPE DE CALOR E INSOLACAO': 'T67.0',
    'GOLPE PANCADA PONTAPE MORDEDURA OU ESCORIACAO INFLIGIDOS POR OUTRA PESSOA': 'W50',
    'GOMAS E ULCERAS DEVIDAS A BOUBA': 'A66.4',
    'GONARTROSE (ARTROSE DO JOELHO)': 'M17',
    'GONARTROSE NAO ESPECIFICADA': 'M17.9',
    'GONARTROSE POS-TRAUMATICA BILATERAL': 'M17.2',
    'GONARTROSE PRIMARIA BILATERAL': 'M17.0',
    'GONORREIA COMPLICANDO A GRAVIDEZ, O PARTO E O PUERPERIO': 'O98.2',
    'GOTA': 'M10',
    'GOTA DEVIDA A DISFUNCAO RENAL': 'M10.3',
    'GOTA IDIOPATICA': 'M10.0',
    'GOTA, NAO ESPECIFICADA': 'M10.9',
    'GRANULOMA ANULAR': 'L92.0',
    'GRANULOMA DE CORPO ESTRANHO DA PELE E DO TECIDO SUBCUTANEO': 'L92.3',
    'GRANULOMA DE CORPO ESTRANHO NO TECIDO MOLE NAO CLASSIFICADO EM OUTRA PARTE': 'M60.2',
    'GRANULOMA INGUINAL': 'A58',
    'GRANULOMA PIOGENICO': 'L98.0',
    'GRAVIDEZ (AINDA) NAO CONFIRMADA': 'Z32.0',
    'GRAVIDEZ ABDOMINAL': 'O00.0',
    'GRAVIDEZ CONFIRMADA': 'Z32.1',
    'GRAVIDEZ ECTOPICA': 'O00',
    'GRAVIDEZ ECTOPICA, NAO ESPECIFICADA': 'O00.9',
    'GRAVIDEZ OVARIANA': 'O00.2',
    'HALITOSE': 'R19.6',
    'HALLUX VALGO (ADQUIRIDO)': 'M20.1',
    'HANSENIASE [LEPRA] DIMORFA': 'A30.3',
    'HANSENIASE [LEPRA] INDETERMINADA': 'A30.0',
    'HANSENIASE [LEPRA] NAO ESPECIFICADA': 'A30.9',
    'HELMINTIASE INTESTINAL NAO ESPECIFICADA': 'B82.0',
    'HELMINTIASE NAO ESPECIFICADA': 'B83.9',
    'HELMINTIASES INTESTINAIS MISTAS': 'B81.4',
    'HEMANGIOMA DE QUALQUER LOCALIZACAO': 'D18.0',
    'HEMANGIOMA E LINFANGIOMA DE QUALQUER LOCALIZACAO': 'D18',
    'HEMARTROSE': 'M25.0',
    'HEMATEMESE': 'K92.0',
    'HEMATOMA DO LIGAMENTO LARGO': 'N83.7',
    'HEMATURIA NAO ESPECIFICADA': 'R31',
    'HEMATURIA RECIDIVANTE E PERSISTENTE': 'N02',
    'HEMATURIA RECIDIVANTE E PERSISTENTE - ANORMALIDADE GLOMERULAR MINOR': 'N02.0',
    'HEMATURIA RECIDIVANTE E PERSISTENTE - NAO ESPECIFICADA': 'N02.9',
    'HEMATURIA RECIDIVANTE E PERSISTENTE - OUTRAS': 'N02.8',
    'HEMIPLEGIA': 'G81',
    'HEMIPLEGIA FLACIDA': 'G81.0',
    'HEMIPLEGIA NAO ESPECIFICADA': 'G81.9',
    'HEMOPTISE': 'R04.2',
    'HEMORRAGIA CONJUNTIVAL': 'H11.3',
    'HEMORRAGIA DAS VIAS RESPIRATORIAS': 'R04',
    'HEMORRAGIA DO ANUS E DO RETO': 'K62.5',
    'HEMORRAGIA DO HUMOR VITREO': 'H43.1',
    'HEMORRAGIA DO INICIO DA GRAVIDEZ': 'O20',
    'HEMORRAGIA DO INICIO DA GRAVIDEZ, NAO ESPECIFICADA': 'O20.9',
    'HEMORRAGIA E HEMATOMA COMPLICANDO PROCEDIMENTO NAO CLASSIFICADO EM OUTRA PARTE': 'T81.0',
    'HEMORRAGIA EPIDURAL': 'S06.4',
    'HEMORRAGIA GASTROINTESTINAL, SEM OUTRA ESPECIFICACAO': 'K92.2',
    'HEMORRAGIA INTRACEREBRAL': 'I61',
    'HEMORRAGIA INTRACEREBRAL CEREBELAR': 'I61.4',
    'HEMORRAGIA INTRACEREBRAL HEMISFERICA NAO ESPECIFICADA': 'I61.2',
    'HEMORRAGIA INTRACRANIANA (NAO-TRAUMATICA) NAO ESPECIFICADA': 'I62.9',
    'HEMORRAGIA NAO CLASSIFICADA EM OUTRA PARTE': 'R58',
    'HEMORRAGIA NAO ESPECIFICADA DAS VIAS RESPIRATORIAS': 'R04.9',
    'HEMORRAGIA RETINIANA': 'H35.6',
    'HEMORRAGIA SUBARACNOIDE': 'I60',
    'HEMORRAGIA SUBARACNOIDE NAO ESPECIFICADA': 'I60.9',
    'HEMORRAGIA SUBDURAL (AGUDA) (NAO-TRAUMATICA)': 'I62.0',
    'HEMORRAGIA SUBDURAL DEVIDA A TRAUMATISMO': 'S06.5',
    'HEMORRAGIA TARDIA OU EXCESSIVA CONSEQUENTE A ABORTO E A GRAVIDEZ ECTOPICA E MOLAR': 'O08.1',
    'HEMORRAGIA VAGINAL NEONATAL': 'P54.6',
    'HEMORROIDAS': 'I84',
    'HEMORROIDAS EXTERNAS COM OUTRAS COMPLICACOES': 'I84.4',
    'HEMORROIDAS EXTERNAS SEM COMPLICACAO': 'I84.5',
    'HEMORROIDAS EXTERNAS TROMBOSADAS': 'I84.3',
    'HEMORROIDAS INTERNAS COM OUTRAS COMPLICACOES': 'I84.1',
    'HEMORROIDAS INTERNAS SEM COMPLICACOES': 'I84.2',
    'HEMORROIDAS INTERNAS TROMBOSADAS': 'I84.0',
    'HEMORROIDAS NA GRAVIDEZ': 'O22.4',
    'HEMORROIDAS NAO ESPECIFICADAS COM OUTRAS COMPLICACOES': 'I84.8',
    'HEMORROIDAS SEM COMPLICACOES, NAO ESPECIFICADAS': 'I84.9',
    'HEMORROIDAS TROMBOSADAS, NAO ESPECIFICADAS': 'I84.7',
    'HEMOTORAX': 'J94.2',
    'HEMOTORAX TRAUMATICO': 'S27.1',
    'HEPATITE A COM COMA HEPATICO': 'B15.0',
    'HEPATITE A SEM COMA HEPATICO': 'B15.9',
    'HEPATITE AGUDA A': 'B15',
    'HEPATITE AGUDA B': 'B16',
    'HEPATITE AGUDA C': 'B17.1',
    'HEPATITE AGUDA E': 'B17.2',
    'HEPATITE ALCOOLICA': 'K70.1',
    'HEPATITE AUTOIMUNE': 'K75.4',
    'HEPATITE CRONICA ATIVA, NAO CLASSIFICADA EM OUTRA PARTE': 'K73.2',
    'HEPATITE CRONICA NAO CLASSIFICADA EM OUTRA PARTE': 'K73',
    'HEPATITE CRONICA VIRAL B SEM AGENTE DELTA': 'B18.1',
    'HEPATITE CRONICA, SEM OUTRA ESPECIFICACAO': 'K73.9',
    'HEPATITE REATIVA NAO-ESPECIFICA': 'K75.2',
    'HEPATITE VIRAL CRONICA': 'B18',
    'HEPATITE VIRAL CRONICA C': 'B18.2',
    'HEPATITE VIRAL CRONICA NAO ESPECIFICADA': 'B18.9',
    'HEPATITE VIRAL NAO ESPECIFICADA': 'B19',
    'HEPATITE VIRAL, NAO ESPECIFICADA, COM COMA': 'B19.0',
    'HEPATOMEGALIA COM ESPLENOMEGALIA NAO CLASSIFICADA EM OUTRA PARTE': 'R16.2',
    'HEPATOMEGALIA E ESPLENOMEGALIA NAO CLASSIFICADAS EM OUTRA PARTE': 'R16',
    'HEPATOMEGALIA NAO CLASSIFICADA EM OUTRA PARTE': 'R16.0',
    'HERNIA ABDOMINAL NAO ESPECIFICADA': 'K46',
    'HERNIA ABDOMINAL NAO ESPECIFICADA COM GANGRENA': 'K46.1',
    'HERNIA ABDOMINAL NAO ESPECIFICADA, COM OBSTRUCAO, SEM GANGRENA': 'K46.0',
    'HERNIA ABDOMINAL NAO ESPECIFICADA, SEM OBSTRUCAO OU GANGRENA': 'K46.9',
    'HERNIA CONGENITA DE HIATO': 'Q40.1',
    'HERNIA DIAFRAGMATICA': 'K44',
    'HERNIA DIAFRAGMATICA SEM OBSTRUCAO OU GANGRENA': 'K44.9',
    'HERNIA FEMORAL BILATERAL, COM OBSTRUCAO, SEM GANGRENA': 'K41.0',
    'HERNIA FEMORAL BILATERAL, SEM OBSTRUCAO OU GANGRENA': 'K41.2',
    'HERNIA FEMORAL UNILATERAL OU NAO ESPECIFICADA, COM OBSTRUCAO, SEM GANGRENA': 'K41.3',
    'HERNIA FEMORAL UNILATERAL OU NAO ESPECIFICADA, SEM OBSTRUCAO OU GANGRENA': 'K41.9',
    'HERNIA INGUINAL': 'K40',
    'HERNIA INGUINAL BILATERAL, COM OBSTRUCAO, SEM GANGRENA': 'K40.0',
    'HERNIA INGUINAL BILATERAL, SEM OBSTRUCAO OU GANGRENA': 'K40.2',
    'HERNIA INGUINAL UNILATERAL OU NAO ESPECIFICADA, COM GANGRENA': 'K40.4',
    'HERNIA INGUINAL UNILATERAL OU NAO ESPECIFICADA, COM OBSTRUCAO SEM GANGRENA': 'K40.3',
    'HERNIA INGUINAL UNILATERAL OU NAO ESPECIFICADA, SEM OBSTRUCAO OU GANGRENA': 'K40.9',
    'HERNIA UMBILICAL': 'K42',
    'HERNIA UMBILICAL COM OBSTRUCAO, SEM GANGRENA': 'K42.0',
    'HERNIA UMBILICAL SEM OBSTRUCAO OU GANGRENA': 'K42.9',
    'HERNIA VENTRAL': 'K43',
    'HERNIA VENTRAL SEM OBSTRUCAO OU GANGRENA': 'K43.9',
    'HERPES ZOSTER (ZONA)': 'B02',
    'HERPES ZOSTER ACOMPANHADO DE OUTRAS MANIFESTACOES NEUROLOGICAS': 'B02.2',
    'HERPES ZOSTER COM OUTRAS COMPLICACOES': 'B02.8',
    'HERPES ZOSTER DISSEMINADO': 'B02.7',
    'HERPES ZOSTER OFTALMICO': 'B02.3',
    'HERPES ZOSTER SEM COMPLICACAO': 'B02.9',
    'HIDRADENITE SUPURATIVA': 'L73.2',
    'HIDRARTROSE INTERMITENTE': 'M12.4',
    'HIDROCEFALIA': 'G91',
    'HIDROCEFALIA DE PRESSAO NORMAL': 'G91.2',
    'HIDROCEFALIA NAO ESPECIFICADA': 'G91.9',
    'HIDROCELE E ESPERMATOCELE': 'N43',
    'HIDROCELE ENCISTADA': 'N43.0',
    'HIDROCELE INFECTADA': 'N43.1',
    'HIDROCELE NAO ESPECIFICADA': 'N43.3',
    'HIDRONEFROSE COM ESTREITAMENTO DE URETER NAO CLASSIFICADA EM OUTRA PARTE': 'N13.1',
    'HIDRONEFROSE COM OBSTRUCAO DA JUNCAO URETERO-PELVICA': 'N13.0',
    'HIDRONEFROSE COM OBSTRUCAO POR CALCULOSE RENAL E URETERAL': 'N13.2',
    'HIDRONEFROSE CONGENITA': 'Q62.0',
    'HIFEMA': 'H21.0',
    'HIPERATIVIDADE': 'R46.3',
    'HIPERCAROTENEMIA': 'E67.1',
    'HIPERCERATOSE DEVIDA A BOUBA': 'A66.3',
    'HIPEREMESE GRAVIDICA COM DISTURBIO METABOLICO': 'O21.1',
    'HIPEREMESE GRAVIDICA LEVE': 'O21.0',
    'HIPERESPLENISMO': 'D73.1',
    'HIPERESTESIA': 'R20.3',
    'HIPERGAMAGLOBULINEMIA NAO ESPECIFICADA': 'D89.2',
    'HIPERGLICEMIA NAO ESPECIFICADA': 'R73.9',
    'HIPERGLICERIDEMIA PURA': 'E78.1',
    'HIPERIDROSE': 'R61',
    'HIPERIDROSE LOCALIZADA': 'R61.0',
    'HIPERLIPIDEMIA MISTA': 'E78.2',
    'HIPERMETROPIA': 'H52.0',
    'HIPERNASALIDADE E HIPONASALIDADE': 'R49.2',
    'HIPEROSMOLARIDADE E HIPERNATREMIA': 'E87.0',
    'HIPERPIGMENTACAO POS-INFLAMATORIA': 'L81.0',
    'HIPERPLASIA ADENOMATOSA ENDOMETRIAL': 'N85.1',
    'HIPERPLASIA DA PROSTATA': 'N40',
    'HIPERPOTASSEMIA': 'E87.5',
    'HIPERTENSAO ESSENCIAL (PRIMARIA)': 'I10',
    'HIPERTENSAO GESTACIONAL (INDUZIDA PELA GRAVIDEZ) SEM PROTEINURIA SIGNIFICATIVA': 'O13',
    'HIPERTENSAO INTRACRANIANA BENIGNA': 'G93.2',
    'HIPERTENSAO MATERNA NAO ESPECIFICADA': 'O16',
    'HIPERTENSAO PORTAL': 'K76.6',
    'HIPERTENSAO PRE-EXISTENTE COMPLICANDO A GRAVIDEZ O PARTO E O PUERPERIO': 'O10',
    'HIPERTENSAO PRE-EXISTENTE NAO ESPECIFICADA, COMPLICANDO A GRAVIDEZ, O PARTO E O PUERPERIO': 'O10.9',
    'HIPERTENSAO PULMONAR PRIMARIA': 'I27.0',
    'HIPERTENSAO RENOVASCULAR': 'I15.0',
    'HIPERTENSAO SECUNDARIA': 'I15',
    'HIPERTENSAO SECUNDARIA A OUTRAS AFECCOES RENAIS': 'I15.1',
    'HIPERTENSAO SECUNDARIA, NAO ESPECIFICADA': 'I15.9',
    'HIPERTROFIA DA MAMA': 'N62',
    'HIPERTROFIA DAS ADENOIDES': 'J35.2',
    'HIPERTROFIA DAS AMIGDALAS': 'J35.1',
    'HIPERTROFIA DAS AMIGDALAS COM HIPERTROFIA DAS ADENOIDES': 'J35.3',
    'HIPERTROFIA DAS PAPILAS LINGUAIS': 'K14.3',
    'HIPERTROFIA DE GLANDULA SALIVAR': 'K11.1',
    'HIPERTROFIA DO PREPUCIO, FIMOSE E PARAFIMOSE': 'N47',
    'HIPERTROFIA DOS CORNETOS NASAIS': 'J34.3',
    'HIPERURICEMIA SEM SINAIS DE ARTRITE INFLAMATORIA E DE DOENCA COM TOFOS': 'E79.0',
    'HIPOESTESIA CUTANEA': 'R20.1',
    'HIPOFUNCAO TESTICULAR': 'E29.1',
    'HIPOGLICEMIA INDUZIDA POR DROGA SEM COMA': 'E16.0',
    'HIPOGLICEMIA NAO ESPECIFICADA': 'E16.2',
    'HIPOPARATIREOIDISMO': 'E20',
    'HIPOPLASIA DO(S) TESTICULO(S) E DO ESCROTO': 'Q55.1',
    'HIPOPLASIA RENAL DE CAUSA DESCONHECIDA': 'N27',
    'HIPOPOTASSEMIA': 'E87.6',
    'HIPOSMOLARIDADE E HIPONATREMIA': 'E87.1',
    'HIPOSPADIA BALANICA': 'Q54.0',
    'HIPOTENSAO': 'I95',
    'HIPOTENSAO DEVIDA A DROGAS': 'I95.2',
    'HIPOTENSAO IDIOPATICA': 'I95.0',
    'HIPOTENSAO INTRACRANIANA CONSEQUENTE A DERIVACAO VENTRICULAR': 'G97.2',
    'HIPOTENSAO NAO ESPECIFICADA': 'I95.9',
    'HIPOTENSAO ORTOSTATICA': 'I95.1',
    'HIPOTERMIA': 'T68',
    'HIPOTIREOIDISMO CONGENITO COM BOCIO DIFUSO': 'E03.0',
    'HIPOTIREOIDISMO CONGENITO SEM BOCIO': 'E03.1',
    'HIPOTIREOIDISMO NAO ESPECIFICADO': 'E03.9',
    'HIPOTIREOIDISMO SUBCLINICO POR DEFICIENCIA DE IODO': 'E02',
    'HISTORIA FAMILIAR DE ABUSO DE ALCOOL': 'Z81.1',
    'HISTORIA FAMILIAR DE ABUSO DE OUTRA SUBSTANCIA PSICOATIVA': 'Z81.3',
    'HISTORIA FAMILIAR DE ACIDENTE VASCULAR CEREBRAL': 'Z82.3',
    'HISTORIA FAMILIAR DE ASMA E OUTRAS DOENCAS RESPIRATORIAS INFERIORES CRONICAS': 'Z82.5',
    'HISTORIA FAMILIAR DE NEOPLASIA DE MAMA': 'Z80.3',
    'HISTORIA FAMILIAR DE SURDEZ E PERDA DE AUDICAO': 'Z82.2',
    'HISTORIA PESSOAL DE ABUSO DE SUBSTANCIAS PSICOATIVAS': 'Z86.4',
    'HISTORIA PESSOAL DE ALERGIA A AGENTE ANALGESICO': 'Z88.6',
    'HISTORIA PESSOAL DE ALERGIA A AGENTE ANESTESICO': 'Z88.4',
    'HISTORIA PESSOAL DE ALERGIA A DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS NAO ESPECIFICADAS': 'Z88.9',
    'HISTORIA PESSOAL DE ALERGIA A OUTRO AGENTE ANTIBIOTICO': 'Z88.1',
    'HISTORIA PESSOAL DE ALERGIA A OUTROS AGENTES ANTIINFECCIOSOS': 'Z88.3',
    'HISTORIA PESSOAL DE ALERGIA A OUTROS DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS': 'Z88.8',
    'HISTORIA PESSOAL DE ALERGIA A PENICILINA': 'Z88.0',
    'HISTORIA PESSOAL DE ALERGIA A SORO E A VACINA': 'Z88.7',
    'HISTORIA PESSOAL DE ALERGIA AS SULFONAMIDAS': 'Z88.2',
    'HISTORIA PESSOAL DE ALGUMAS OUTRAS DOENCAS': 'Z86',
    'HISTORIA PESSOAL DE ANTICONCEPCAO': 'Z92.0',
    'HISTORIA PESSOAL DE AUTO AGRESSAO': 'Z91.5',
    'HISTORIA PESSOAL DE CIRURGIA DE GRANDE PORTE NAO CLASSIFICADA EM OUTRA PARTE': 'Z92.4',
    'HISTORIA PESSOAL DE FATORES DE RISCO NAO CLASSIFICADOS EM OUTRA PARTE': 'Z91',
    'HISTORIA PESSOAL DE LEUCEMIA': 'Z85.6',
    'HISTORIA PESSOAL DE MA HIGIENE PESSOAL': 'Z91.2',
    'HISTORIA PESSOAL DE NAO ADERENCIA A TRATAMENTO OU REGIME MEDICO': 'Z91.1',
    'HISTORIA PESSOAL DE NEOPLASIA MALIGNA DE MAMA': 'Z85.3',
    'HISTORIA PESSOAL DE OUTROS FATORES DE RISCO ESPECIFICADOS NAO CLASSIFICADOS EM OUTRA PARTE': 'Z91.8',
    'HISTORIA PESSOAL DE OUTROS TRANSTORNOS MENTAIS E COMPORTAMENTAIS': 'Z86.5',
    'HISTORIA PESSOAL DE QUIMIOTERAPIA PARA DOENCA NEOPLASICA': 'Z92.6',
    'HISTORIA PESSOAL DE TRATAMENTO MEDICO': 'Z92',
    'HISTORIA PESSOAL DE TRATAMENTO MEDICO NAO ESPECIFICADO': 'Z92.9',
    'HISTORIA PESSOAL DE TRAUMA PSICOLOGICO NAO CLASSIFICADO EM OUTRA PARTE': 'Z91.4',
    'HISTORIA PESSOAL DE USO DE LONGO PRAZO (ATUAL) DE ANTICOAGULANTES': 'Z92.1',
    'HISTORIA PESSOAL DE USO DE LONGO PRAZO (ATUAL) DE OUTROS MEDICAMENTOS': 'Z92.2',
    'HORDEOLO E CALAZIO': 'H00',
    'HORDEOLO E OUTRAS INFLAMACOES PROFUNDAS DAS PALPEBRAS': 'H00.0',
    'ICTERICIA NAO ESPECIFICADA': 'R17',
    'ICTERICIA NEONATAL ASSOCIADA AO PARTO PREMATURO': 'P59.0',
    'ICTERICIA NEONATAL DEVIDA A CONTUSOES': 'P58.0',
    'ICTERICIA NEONATAL DEVIDA A DEGLUTACAO DE SANGUE MATERNO': 'P58.5',
    'ICTERICIA NEONATAL DEVIDA A HEMOLISE EXCESSIVA NAO ESPECIFICADA': 'P58.9',
    'ICTERICIA NEONATAL DEVIDA A INIBIDORES DO LEITE MATERNO': 'P59.3',
    'ICTERICIA NEONATAL DEVIDA A OUTRAS CAUSAS E AS NAO ESPECIFICADAS': 'P59',
    'ICTERICIA NEONATAL DEVIDA A OUTRAS CAUSAS ESPECIFICADAS': 'P59.8',
    'ICTERICIA NEONATAL DEVIDA A OUTRAS HEMOLISES EXCESSIVAS': 'P58',
    'ICTERICIA NEONATAL DEVIDA A OUTRAS HEMOLISES EXCESSIVAS ESPECIFICADAS': 'P58.8',
    'ICTERICIA NEONATAL DEVIDA A POLICITEMIA': 'P58.3',
    'ICTERICIA NEONATAL NAO ESPECIFICADA': 'P59.9',
    'ILEO PARALITICO': 'K56.0',
    'ILEO PARALITICO E OBSTRUCAO INTESTINAL SEM HERNIA': 'K56',
    'ILEOCOLITE ULCERATIVA (CRONICA)': 'K51.1',
    'ILEOSTOMIA': 'Z93.2',
    'IMPACTO ACIDENTAL ATIVO OU PASSIVO CAUSADO POR EQUIPAMENTO ESPORTIVO': 'W21',
    'IMPACTO ACIDENTAL ATIVO OU PASSIVO CAUSADO POR OUTROS OBJETOS': 'W22',
    'IMPACTO CAUSADO POR OBJETO LANCADO PROJETADO OU EM QUEDA': 'W20',
    'IMPETIGINIZACAO DE OUTRAS DERMATOSES': 'L01.1',
    'IMPETIGO': 'L01',
    'IMPETIGO [QUALQUER LOCALIZACAO] [QUALQUER MICROORGANISMO]': 'L01.0',
    'IMPOTENCIA DE ORIGEM ORGANICA': 'N48.4',
    'IMUNODEFICIENCIA COM PREDOMINANCIA DE DEFEITOS DE ANTICORPOS': 'D80',
    'IMUNOTERAPIA PROFILATICA': 'Z29.1',
    'INALACAO E INGESTAO DE ALIMENTOS CAUSANDO OBSTRUCAO DO TRATO RESPIRATORIO': 'W79',
    'INALACAO E INGESTAO DE OUTROS OBJETOS CAUSANDO OBSTRUCAO DO TRATO RESPIRATORIO': 'W80',
    'INCONTINENCIA DE TENSAO (STRESS)': 'N39.3',
    'INCONTINENCIA FECAL': 'R15',
    'INCONTINENCIA URINARIA NAO ESPECIFICADA': 'R32',
    'INDURATIO PENIS PLASTICA': 'N48.6',
    'INFARTO AGUDO DO MIOCARDIO': 'I21',
    'INFARTO AGUDO DO MIOCARDIO NAO ESPECIFICADO': 'I21.9',
    'INFARTO AGUDO SUBENDOCARDICO DO MIOCARDIO': 'I21.4',
    'INFARTO AGUDO TRANSMURAL DA PAREDE ANTERIOR DO MIOCARDIO': 'I21.0',
    'INFARTO AGUDO TRANSMURAL DA PAREDE INFERIOR DO MIOCARDIO': 'I21.1',
    'INFARTO AGUDO TRANSMURAL DO MIOCARDIO DE OUTRAS LOCALIZACOES': 'I21.2',
    'INFARTO AGUDO TRANSMURAL DO MIOCARDIO, DE LOCALIZACAO NAO ESPECIFICADA': 'I21.3',
    'INFARTO CEREBRAL': 'I63',
    'INFARTO CEREBRAL DEVIDO A TROMBOSE DE ARTERIAS PRE-CEREBRAIS': 'I63.0',
    'INFARTO DO MIOCARDIO RECORRENTE': 'I22',
    'INFARTO DO MIOCARDIO RECORRENTE DE LOCALIZACAO NAO ESPECIFICADA': 'I22.9',
    'INFECCAO (PIOGENICA) DO DISCO INTERVERTEBRAL': 'M46.3',
    'INFECCAO AGUDA DAS VIAS AEREAS SUPERIORES NAO ESPECIFICADA': 'J06.9',
    'INFECCAO ANOGENITAL NAO ESPECIFICADA PELO VIRUS DO HERPES': 'A60.9',
    'INFECCAO BACTERIANA DE LOCALIZACAO NAO ESPECIFICADA': 'A49',
    'INFECCAO BACTERIANA NAO ESPECIFICADA': 'A49.9',
    'INFECCAO CAUSADA POR CLAMIDIAS NAO ESPECIFICADA': 'A74.9',
    'INFECCAO CONGENITA POR CITOMEGALOVIRUS': 'P35.1',
    'INFECCAO CONGENITA POR VIRUS DO HERPES [SIMPLES]': 'P35.2',
    'INFECCAO CUTANEA MICOBACTERIANA': 'A31.1',
    'INFECCAO DA FARINGE POR CLAMIDIAS': 'A56.4',
    'INFECCAO DA INCISAO CIRURGICA DE ORIGEM OBSTETRICA': 'O86.0',
    'INFECCAO DA MARGEM CUTANEA DO ANUS E DO RETO PELO VIRUS DO HERPES': 'A60.1',
    'INFECCAO DE COTO DA AMPUTACAO': 'T87.4',
    'INFECCAO DO TRATO URINARIO DE LOCALIZACAO NAO ESPECIFICADA': 'N39.0',
    'INFECCAO DOS ORGAOS GENITAIS E DO TRATO GENITURINARIO PELO VIRUS DO HERPES': 'A60.0',
    'INFECCAO E REACAO INFLAMATORIA DEVIDAS A DISPOSITIVO DE FIXACAO INTERNA [QUALQUER LOCAL]': 'T84.6',
    'INFECCAO E REACAO INFLAMATORIA DEVIDAS A PROTESE ARTICULAR INTERNA': 'T84.5',
    'INFECCAO ESTAFILOCOCICA DE LOCALIZACAO NAO ESPECIFICADA': 'A49.0',
    'INFECCAO ESTREPTOCOCICA DE LOCALIZACAO NAO ESPECIFICADA': 'A49.1',
    'INFECCAO GONOCOCICA': 'A54',
    'INFECCAO GONOCOCICA DO ANUS OU DO RETO': 'A54.6',
    'INFECCAO GONOCOCICA DO OLHO': 'A54.3',
    'INFECCAO GONOCOCICA NAO ESPECIFICADA': 'A54.9',
    'INFECCAO INTESTINAL BACTERIANA NAO ESPECIFICADA': 'A04.9',
    'INFECCAO INTESTINAL DEVIDA A VIRUS NAO ESPECIFICADO': 'A08.4',
    'INFECCAO LOCALIZADA DA PELE E DO TECIDO SUBCUTANEO, NAO ESPECIFICADA': 'L08.9',
    'INFECCAO MICOBACTERIANA NAO ESPECIFICADA': 'A31.9',
    'INFECCAO NAO ESPECIFICADA DEVIDA AO VIRUS DO HERPES': 'B00.9',
    'INFECCAO NAO ESPECIFICADA DO TRATO URINARIO NA GRAVIDEZ': 'O23.4',
    'INFECCAO NAO ESPECIFICADA POR SALMONELA': 'A02.9',
    'INFECCAO NEONATAL DO TRATO URINARIO': 'P39.3',
    'INFECCAO POR ADENOVIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.0',
    'INFECCAO POR CLAMIDIAS DO TRATO GENITURINARIO, LOCALIZACAO NAO ESPECIFICADA': 'A56.2',
    'INFECCAO POR CLAMIDIAS TRANSMITIDA POR VIA SEXUAL, DE OUTRAS LOCALIZACOES': 'A56.8',
    'INFECCAO POR CLAMIDIAS, PELVIPERITONIAL E DE OUTROS ORGAOS GENITURINARIOS': 'A56.1',
    'INFECCAO POR CORONAVIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.2',
    'INFECCAO POR ENTEROVIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.1',
    'INFECCAO POR ESCHERICHIA COLI ENTEROHEMORRAGICA': 'A04.3',
    'INFECCAO POR ESCHERICHIA COLI ENTEROPATOGENICA': 'A04.0',
    'INFECCAO POR ESCHERICHIA COLI ENTEROTOXIGENICA': 'A04.1',
    'INFECCAO POR HAEMOPHILUS INFLUENZAE DE LOCALIZACAO NAO ESPECIFICADA': 'A49.2',
    'INFECCAO POR MYCOPLASMA DE LOCALIZACAO NAO ESPECIFICADA': 'A49.3',
    'INFECCAO POR PAPOVAVIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.4',
    'INFECCAO POR PARVOVIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.3',
    'INFECCAO POR RETROVIRUS NAO CLASSIFICADA EM OUTRA PARTE': 'B33.3',
    'INFECCAO POS-TRAUMATICA DE FERIMENTO NAO CLASSIFICADA EM OUTRA PARTE': 'T79.3',
    'INFECCAO PULMONAR MICOBACTERIANA': 'A31.0',
    'INFECCAO SUBSEQUENTE A PROCEDIMENTO NAO CLASSIFICADA EM OUTRA PARTE': 'T81.4',
    'INFECCAO VIRAL NAO ESPECIFICADA': 'B34.9',
    'INFECCAO VIRAL NAO ESPECIFICADA CARACTERIZADA POR LESOES DA PELE E MEMBRANAS MUCOSAS': 'B09',
    'INFECCOES AGUDAS DAS VIAS AEREAS SUPERIORES DE LOCALIZACOES MULTIPLAS E NAO ESPECIFICADAS': 'J06',
    'INFECCOES AGUDAS NAO ESPECIFICADA DAS VIAS AEREAS INFERIORES': 'J22',
    'INFECCOES ANOGENITAIS PELO VIRUS DO HERPES (HERPES SIMPLES)': 'A60',
    'INFECCOES DA URETRA NA GRAVIDEZ': 'O23.2',
    'INFECCOES DE OUTRAS PARTES DO TRATO URINARIO NA GRAVIDEZ': 'O23.3',
    'INFECCOES DO TRATO GENITURINARIO NA GRAVIDEZ': 'O23',
    'INFECCOES INTESTINAIS VIRAIS OUTRAS E AS NAO ESPECIFICADAS': 'A08',
    'INFECCOES LOCALIZADAS POR SALMONELA': 'A02.2',
    'INFECCOES MAMARIAS ASSOCIADAS AO PARTO': 'O91',
    'INFECCOES PELO VIRUS DO HERPES (HERPES SIMPLES)': 'B00',
    'INFECCOES POR CLAMIDIAS DO TRATO GENITURINARIO INFERIOR': 'A56.0',
    'INFECCOES POR VIRUS ATIPICOS DO SISTEMA NERVOSO CENTRAL': 'A81',
    'INFECCOES VIRAIS NAO ESPECIFICADAS DO SISTEMA NERVOSO CENTRAL': 'A89',
    'INFERTILIDADE MASCULINA': 'N46',
    'INFESTACAO NAO ESPECIFICADA': 'B88.9',
    'INFLAMACAO AGUDA DA ORBITA': 'H05.0',
    'INFLAMACAO CORIORRETINIANA': 'H30',
    'INFLAMACAO CRONICA DOS CANAIS LACRIMAIS': 'H04.4',
    'INFLAMACAO NAO ESPECIFICADA DA COROIDE E DA RETINA': 'H30.9',
    'INFLAMACAO NAO ESPECIFICADA DA PALPEBRA': 'H01.9',
    'INFLAMACAO PELVICA FEMININA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N74.8',
    'INFLUENZA (GRIPE) DEVIDA A VIRUS NAO IDENTIFICADO': 'J11',
    'INFLUENZA COM OUTRAS MANIFESTACOES, DEVIDA A OUTRO VIRUS DA INFLUENZA [GRIPE] IDENTIFICADO': 'J10.8',
    'INFLUENZA COM PNEUMONIA DEVIDA A OUTRO VIRUS DA INFLUENZA [GRIPE] IDENTIFICADO': 'J10.0',
    'INFLUENZA DEVIDA A OUTRO VIRUS DA INFLUENZA [GRIPE] IDENTIFICADO': 'J10',
    'INFLUENZA [GRIPE] COM OUTRAS MANIFESTACOES RESPIRATORIAS, DEVIDA A VIRUS NAO IDENTIFICADO': 'J11.1',
    'INFLUENZA [GRIPE] COM OUTRAS MANIFESTACOES, DEVIDA A VIRUS NAO IDENTIFICADO': 'J11.8',
    'INFLUENZA [GRIPE] COM PNEUMONIA, DEVIDA A VIRUS NAO IDENTIFICADO': 'J11.0',
    'INFLUENZA [GRIPE] DEVIDA A VIRUS IDENTIFICADO DA GRIPE AVIARIA': 'J09',
    'INJECAO OU VACINACAO COM MEDICAMENTO OU SUBSTANCIA BIOLOGICA CONTAMINADOS': 'Y64.1',
    'INQUIETACAO E PREOCUPACAO EXAGERADAS COM ACONTECIMENTOS ESTRESSANTES': 'R46.6',
    'INSONIA NAO-ORGANICA': 'F51.0',
    'INSTABILIDADE CRONICA DO JOELHO': 'M23.5',
    'INSTABILIDADES DA COLUNA VERTEBRAL': 'M53.2',
    'INSUFICIENCIA (DA VALVA) AORTICA': 'I35.1',
    'INSUFICIENCIA (DA VALVA) MITRAL': 'I34.0',
    'INSUFICIENCIA ADRENOCORTICAL PRIMARIA': 'E27.1',
    'INSUFICIENCIA AORTICA REUMATICA': 'I06.1',
    'INSUFICIENCIA CARDIACA': 'I50',
    'INSUFICIENCIA CARDIACA CONGESTIVA': 'I50.0',
    'INSUFICIENCIA CARDIACA NAO ESPECIFICADA': 'I50.9',
    'INSUFICIENCIA HEPATICA AGUDA E SUBAGUDA': 'K72.0',
    'INSUFICIENCIA HEPATICA ALCOOLICA': 'K70.4',
    'INSUFICIENCIA HEPATICA CRONICA': 'K72.1',
    'INSUFICIENCIA HEPATICA NAO CLASSIFICADA EM OUTRA PARTE': 'K72',
    'INSUFICIENCIA HEPATICA, SEM OUTRAS ESPECIFICACOES': 'K72.9',
    'INSUFICIENCIA MITRAL REUMATICA': 'I05.1',
    'INSUFICIENCIA PULMONAR CRONICA POS-CIRURGICA': 'J95.3',
    'INSUFICIENCIA RENAL AGUDA': 'N17',
    'INSUFICIENCIA RENAL AGUDA COM NECROSE TUBULAR': 'N17.0',
    'INSUFICIENCIA RENAL AGUDA NAO ESPECIFICADA': 'N17.9',
    'INSUFICIENCIA RENAL CRONICA': 'N18',
    'INSUFICIENCIA RENAL CRONICA NAO ESPECIFICADA': 'N18.9',
    'INSUFICIENCIA RENAL NAO ESPECIFICADA': 'N19',
    'INSUFICIENCIA RENAL POS-PROCEDIMENTOS': 'N99.0',
    'INSUFICIENCIA RESPIRATORIA AGUDA': 'J96.0',
    'INSUFICIENCIA RESPIRATORIA CRONICA': 'J96.1',
    'INSUFICIENCIA RESPIRATORIA NAO CLASSIFICADA DE OUTRA PARTE': 'J96',
    'INSUFICIENCIA RESPIRATORIA NAO ESPECIFICADA': 'J96.9',
    'INSUFICIENCIA VENOSA (CRONICA) (PERIFERICA)': 'I87.2',
    'INSUFICIENCIA VENTRICULAR ESQUERDA': 'I50.1',
    'INTERTRIGO ERITEMATOSO': 'L30.4',
    'INTOLERANCIA A LACTOSE': 'E73',
    'INTOLERANCIA A LACTOSE, NAO ESPECIFICADA': 'E73.9',
    'INTOXICACAO ACIDENTAL POR E EXPOSICAO A OUTROS GASES E VAPORES': 'X47',
    'INTOXICACAO ACIDENTAL POR E EXPOSICAO A OUTROS GASES E VAPORES - AREAS DE COMERCIO E DE SERVICOS': 'X47.5',
    'INTOXICACAO ACIDENTAL POR E EXPOSICAO A OUTROS GASES E VAPORES - LOCAL NAO ESPECIFICADO': 'X47.9',
    'INTOXICACAO ACIDENTAL POR E EXPOSICAO A OUTROS GASES E VAPORES - RESIDENCIA': 'X47.0',
    'INTOXICACAO ALCOOLICA GRAVE': 'Y91.2',
    'INTOXICACAO ALCOOLICA LEVE': 'Y91.0',
    'INTOXICACAO ALCOOLICA MODERADA': 'Y91.1',
    'INTOXICACAO ALIMENTAR BACTERIANA NAO ESPECIFICADA': 'A05.9',
    'INTOXICACAO ALIMENTAR DEVIDA A CLOSTRIDIUM PERFRINGENS [CLOSTRIDIUM WELCHII]': 'A05.2',
    'INTOXICACAO ALIMENTAR DEVIDA A VIBRIO PARAHEMOLYTICUS': 'A05.3',
    'INTOXICACAO ALIMENTAR ESTAFILOCOCICA': 'A05.0',
    'INTOXICACAO PELO GRUPO DO CLORANFENICOL': 'T36.2',
    'INTOXICACAO POR ADSTRINGENTES E DETERGENTES DE USO LOCAL': 'T49.2',
    'INTOXICACAO POR AGENTES DE DIAGNOSTICO': 'T50.8',
    'INTOXICACAO POR ANALGESICO NAO-OPIACEO, ANTIPIRETICO E ANTI-REUMATICO, NAO ESPECIFICADOS': 'T39.9',
    'INTOXICACAO POR ANALGESICOS ANTIPIRETICOS E ANTI-REUMATICOS NAO-OPIACEOS': 'T39',
    'INTOXICACAO POR ANDROGENOS E ANABOLIZANTES CONGENERES': 'T38.7',
    'INTOXICACAO POR ANTI-HELMINTICOS': 'T37.4',
    'INTOXICACAO POR ANTIBIOTICOS SISTEMICOS': 'T36',
    'INTOXICACAO POR ANTIBIOTICOS SISTEMICOS NAO ESPECIFICADOS': 'T36.9',
    'INTOXICACAO POR ANTICOAGULANTES': 'T45.5',
    'INTOXICACAO POR ANTIDEPRESSIVOS TRICICLICOS E TETRACICLICOS': 'T43.0',
    'INTOXICACAO POR BENZODIAZEPINAS': 'T42.4',
    'INTOXICACAO POR BLOQUEADORES GANGLIONARES NAO CLASSIFICADOS EM OUTRA PARTE': 'T44.2',
    'INTOXICACAO POR CANNABIS (DERIVADOS)': 'T40.7',
    'INTOXICACAO POR COCAINA': 'T40.5',
    'INTOXICACAO POR DERIVADOS DO 4-AMINOFENOL': 'T39.1',
    'INTOXICACAO POR DIGESTIVOS': 'T47.5',
    'INTOXICACAO POR DROGAS ANTI-RESFRIADO': 'T48.5',
    'INTOXICACAO POR DROGAS ANTIALERGICAS E ANTIEMETICAS': 'T45.0',
    'INTOXICACAO POR DROGAS ANTIVIRAIS': 'T37.5',
    'INTOXICACAO POR DROGAS E PREPARACOES DE USO OTORRINOLARINGOLOGICO': 'T49.6',
    'INTOXICACAO POR DROGAS PSICOTROPICAS NAO CLASSIFICADAS EM OUTRA PARTE': 'T43',
    'INTOXICACAO POR DROGAS QUE AFETAM PRINCIPALMENTE O SISTEMA NERVOSO AUTONOMO': 'T44',
    'INTOXICACAO POR ENZIMAS, NAO CLASSIFICADAS EM OUTRA PARTE': 'T45.3',
    'INTOXICACAO POR GASES TERAPEUTICOS': 'T41.5',
    'INTOXICACAO POR GLICOCORTICOIDES E ANALOGOS SINTETICOS': 'T38.0',
    'INTOXICACAO POR INIBIDORES DA ENZIMA DE CONVERSAO DA ANGIOTENSINA': 'T46.4',
    'INTOXICACAO POR LISERGIDA [LSD]': 'T40.8',
    'INTOXICACAO POR MINERALOCORTICOIDES E SEUS ANTAGONISTAS': 'T50.0',
    'INTOXICACAO POR NARCOTICOS E PSICODISLEPTICOS (ALUCINOGENOS)': 'T40',
    'INTOXICACAO POR OPIO': 'T40.0',
    'INTOXICACAO POR OUTRAS DROGAS ANTIEPILEPTICAS E SEDATIVOS-HIPNOTICOS': 'T42.6',
    'INTOXICACAO POR OUTRAS DROGAS MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS E AS NAO ESPECIFICADAS': 'T50.9',
    'INTOXICACAO POR OUTRAS DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS E AS NAO ESPECIFICADAS': 'T50.9',
    'INTOXICACAO POR OUTRAS SUBSTANCIAS QUE ATUAM PRIMARIAMENTE SOBRE O APARELHO GASTRINTESTINAL': 'T47.8',
    'INTOXICACAO POR OUTROS ANTAGONISTAS HORMONAIS, E OS NAO ESPECIFICADOS': 'T38.9',
    'INTOXICACAO POR OUTROS ANTIBIOTICOS SISTEMICOS': 'T36.8',
    'INTOXICACAO POR OUTROS ANTIDEPRESSIVOS E OS NAO ESPECIFICADOS': 'T43.2',
    'INTOXICACAO POR OUTROS ANTIINFLAMATORIOS NAO ESTEROIDES': 'T39.3',
    'INTOXICACAO POR OUTROS ANTIPSICOTICOS E NEUROLEPTICOS E OS NAO ESPECIFICADOS': 'T43.5',
    'INTOXICACAO POR OUTROS MEDICAMENTOS ANTIPROTOZOARIOS': 'T37.3',
    'INTOXICACAO POR OUTROS OPIACEOS': 'T40.2',
    'INTOXICACAO POR PENICILINAS': 'T36.0',
    'INTOXICACAO POR PREPARADO DE USO TOPICO, NAO ESPECIFICADO': 'T49.9',
    'INTOXICACAO POR PRODUTOS QUE AGEM SOBRE O EQUILIBRIO ELETROLITICO, CALORICO E HIDRICO': 'T50.3',
    'INTOXICACAO POR PSICOESTIMULANTES QUE POTENCIALMENTE PODEM PROVOCAR DEPENDENCIA': 'T43.6',
    'INTOXICACAO POR RIFAMICINAS': 'T36.6',
    'INTOXICACAO POR SALICILATOS': 'T39.0',
    'INTOXICACAO POR SUBSTANCIA NAO ESPECIFICADA QUE ATUA PRIMARIAMENTE SOBRE O APARELHO GASTROINTESTINAL': 'T47.9',
    'INTOXICACAO POR SULFONAMIDAS': 'T37.0',
    'INTUMESCIMENTO MAMARIO DO RECEM-NASCIDO': 'P83.4',
    'INTUSSUSCEPCAO': 'K56.1',
    'IRIDOCICLITE': 'H20',
    'IRIDOCICLITE EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H22.1',
    'IRRITABILIDADE E MAU HUMOR': 'R45.4',
    'ISOIMUNIZACAO RH DO FETO E DO RECEM-NASCIDO': 'P55.0',
    'ISOLAMENTO': 'Z29.0',
    'ISOMERISMO DOS APENDICES ATRIAIS': 'Q20.6',
    'ISQUEMIA CEREBRAL TRANSITORIA NAO ESPECIFICADA': 'G45.9',
    'KLEBSIELLA PNEUMONIAE [M. PNEUMONIAE], COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B96.1',
    'KWASHIORKOR': 'E40',
    'KWASHIORKOR MARASMATICO': 'E42',
    'LABIRINTITE': 'H83.0',
    'LACERACAO E RUPTURA OCULAR COM PROLAPSO OU PERDA DE TECIDO INTRA-OCULAR': 'S05.2',
    'LACERACAO OCULAR SEM PROLAPSO OU PERDA DE TECIDO INTRA-OCULAR': 'S05.3',
    'LARINGITE AGUDA': 'J04.0',
    'LARINGITE CRONICA': 'J37.0',
    'LARINGITE E LARINGOTRAQUEITE CRONICAS': 'J37',
    'LARINGITE E TRAQUEITE AGUDAS': 'J04',
    'LARINGITE OBSTRUTIVA AGUDA (CRUPE) E EPIGLOTITE': 'J05',
    'LARINGITE OBSTRUTIVA AGUDA [CRUPE]': 'J05.0',
    'LARINGOFARINGITE AGUDA': 'J06.0',
    'LARINGOMALACIA CONGENITA': 'Q31.5',
    'LARINGOTRAQUEITE AGUDA': 'J04.2',
    'LARINGOTRAQUEITE CRONICA': 'J37.1',
    'LARVA MIGRANS VISCERAL': 'B83.0',
    'LEIOMIOMA DO UTERO': 'D25',
    'LEIOMIOMA DO UTERO, NAO ESPECIFICADO': 'D25.9',
    'LEIOMIOMA SUBMUCOSO DO UTERO': 'D25.0',
    'LEIOMIOMA SUBSEROSO DO UTERO': 'D25.2',
    'LEISHMANIOSE CUTANEA': 'B55.1',
    'LEPTOSPIROSE': 'A27',
    'LEPTOSPIROSE NAO ESPECIFICADA': 'A27.9',
    'LESAO AUTOPROVOCADA INTENCIONALMENTE POR ENFORCAMENTO ESTRANGULAMENTO E SUFOCACAO': 'X70',
    'LESAO AUTOPROVOCADA INTENCIONALMENTE POR MEIOS NAO ESPECIFICADOS': 'X84',
    'LESAO AUTOPROVOCADA INTENCIONALMENTE POR OBJETO CONTUNDENTE': 'X79',
    'LESAO AUTOPROVOCADA INTENCIONALMENTE POR OBJETO CORTANTE OU PENETRANTE': 'X78',
    'LESAO AUTOPROVOCADA INTENCIONALMENTE POR PRECIPITACAO DE UM LUGAR ELEVADO': 'X80',
    'LESAO DO COURO CABELUDO DEVIDA A TRAUMATISMO DE PARTO': 'P12',
    'LESAO DO NERVO CIATICO': 'G57.0',
    'LESAO DO NERVO PLANTAR': 'G57.6',
    'LESAO DO NERVO RADIAL': 'G56.3',
    'LESAO ENCEFALICA ANOXICA, NAO CLASSIFICADA EM OUTRA PARTE': 'G93.1',
    'LESAO NAO ESPECIFICADA DO OMBRO': 'M75.9',
    'LESAO POR ESMAGAMENTO DA FACE': 'S07.0',
    'LESAO POR ESMAGAMENTO DE OUTRAS PARTES DO TORNOZELO E DO PE': 'S97.8',
    'LESAO POR ESMAGAMENTO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DO PUNHO E DA MAO': 'S67.8',
    'LESAO POR ESMAGAMENTO DO ANTEBRACO': 'S57',
    'LESAO POR ESMAGAMENTO DO CRANIO': 'S07.1',
    'LESAO POR ESMAGAMENTO DO POLEGAR E DE OUTRO(S) DEDO(S)': 'S67.0',
    'LESAO POR ESMAGAMENTO DO PUNHO E DA MAO': 'S67',
    'LESAO POR ESMAGAMENTO DO TORNOZELO E DO PE': 'S97',
    'LESOES BIOMECANICAS NAO CLASSIFICADAS EM OUTRA PARTE': 'M99',
    'LESOES DA GENGIVA E DO REBORDO ALVEOLAR SEM DENTES, ASSOCIADAS A TRAUMATISMOS': 'K06.2',
    'LESOES DO NERVO CUBITAL [ULNAR]': 'G56.2',
    'LESOES DO OMBRO': 'M75',
    'LESOES GRANULOMATOSAS E GRANULOMATOIDES DA MUCOSA ORAL': 'K13.4',
    'LESOES INICIAIS DA BOUBA': 'A66.0',
    'LESOES INTERMEDIARIAS DA PINTA': 'A67.1',
    'LESOES MISTAS DA PINTA': 'A67.3',
    'LESOES OSTEOARTICULARES DEVIDAS A BOUBA': 'A66.6',
    'LESOES POR ESMAGAMENTO DA CABECA': 'S07',
    'LESOES PRIMARIAS DA PINTA': 'A67.0',
    'LEUCEMIA LINFOBLASTICA AGUDA': 'C91.0',
    'LEUCEMIA LINFOCITICA CRONICA': 'C91.1',
    'LEUCEMIA LINFOIDE': 'C91',
    'LEUCEMIA MIELOIDE': 'C92',
    'LEUCEMIA MIELOIDE AGUDA': 'C92.0',
    'LEUCEMIA MIELOIDE CRONICA': 'C92.1',
    'LEUCEMIA MIELOIDE, NAO ESPECIFICADA': 'C92.9',
    'LEUCEMIA MONOCITICA': 'C93',
    'LEUCEMIA NAO ESPECIFICADA': 'C95.9',
    'LEUCOPLASIA E OUTRAS AFECCOES DO EPITELIO ORAL, INCLUSIVE DA LINGUA': 'K13.2',
    'LINFADENITE AGUDA': 'L04',
    'LINFADENITE AGUDA DE FACE, CABECA E PESCOCO': 'L04.0',
    'LINFADENITE AGUDA DE LOCALIZACAO NAO ESPECIFICADA': 'L04.9',
    'LINFADENITE AGUDA DE MEMBRO INFERIOR': 'L04.3',
    'LINFADENITE AGUDA DE OUTRAS LOCALIZACOES': 'L04.8',
    'LINFADENITE INESPECIFICA': 'I88',
    'LINFADENITE MESENTERICA NAO ESPECIFICA': 'I88.0',
    'LINFADENITE NAO ESPECIFICADA': 'I88.9',
    'LINFADENOPATIA TUBERCULOSA PERIFERICA': 'A18.2',
    'LINFANGITE': 'I89.1',
    'LINFEDEMA HEREDITARIO': 'Q82.0',
    'LINFEDEMA NAO CLASSIFICADO EM OUTRA PARTE': 'I89.0',
    'LINFOGRANULOMA (VENEREO) POR CLAMIDIA': 'A55',
    'LINFOHISTIOCITOSE HEMOFAGOCITICA': 'D76.1',
    'LINFOMA NAO-HODGKIN DIFUSO': 'C83',
    'LINFOMA NAO-HODGKIN DIFUSO, NAO ESPECIFICADO': 'C83.9',
    'LINFOMA NAO-HODGKIN DIFUSO, PEQUENAS CELULAS (DIFUSO)': 'C83.0',
    'LINGUA GEOGRAFICA': 'K14.1',
    'LIPOMATOSE NAO CLASSIFICADA EM OUTRA PARTE': 'E88.2',
    'LIQUEN PLANO': 'L43',
    'LIQUEN PLANO, NAO ESPECIFICADO': 'L43.9',
    'LIQUEN RUBRO MONILIFORME': 'L44.3',
    'LIQUEN SIMPLES CRONICO E PRURIGO': 'L28',
    'LORDOSE NAO ESPECIFICADA': 'M40.5',
    'LUMBAGO COM CIATICA': 'M54.4',
    'LUPUS ERITEMATOSO': 'L93',
    'LUPUS ERITEMATOSO CUTANEO SUBAGUDO': 'L93.1',
    'LUPUS ERITEMATOSO DISCOIDE': 'L93.0',
    'LUPUS ERITEMATOSO DISSEMINADO (SISTEMICO)': 'M32',
    'LUPUS ERITEMATOSO DISSEMINADO [SISTEMICO] COM COMPROMETIMENTO DE OUTROS ORGAOS E SISTEMAS': 'M32.1',
    'LUPUS ERITEMATOSO DISSEMINADO [SISTEMICO] NAO ESPECIFICADO': 'M32.9',
    'LUXACAO (NAO-TRAUMATICA) DA EPIFISE SUPERIOR DO FEMUR': 'M93.0',
    'LUXACAO CONGENITA UNILATERAL DO QUADRIL': 'Q65.0',
    'LUXACAO DA ARTICULACAO ACROMIOCLAVICULAR': 'S43.1',
    'LUXACAO DA ARTICULACAO DO OMBRO': 'S43.0',
    'LUXACAO DA ARTICULACAO DO TORNOZELO': 'S93.0',
    'LUXACAO DA ARTICULACAO ESTERNOCLAVICULAR': 'S43.2',
    'LUXACAO DA CABECA DO RADIO': 'S53.0',
    'LUXACAO DA CARTILAGEM DO SEPTO NASAL': 'S03.1',
    'LUXACAO DA ROTULA [PATELA]': 'S83.0',
    'LUXACAO DAS ARTICULACOES SACROILIACA E SACROCOCCIGEA': 'S33.2',
    'LUXACAO DE OUTRAS PARTES DA CABECA E DAS NAO ESPECIFICADAS': 'S03.3',
    'LUXACAO DE OUTRAS PARTES DO PESCOCO E DAS NAO ESPECIFICADAS': 'S13.2',
    'LUXACAO DE OUTRAS PARTES DO TORAX E DAS NAO ESPECIFICADAS': 'S23.2',
    'LUXACAO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DA CINTURA ESCAPULAR': 'S43.3',
    'LUXACAO DE OUTRAS PARTES E DAS NAO ESPECIFICADAS DO PE': 'S93.3',
    'LUXACAO DE VERTEBRA CERVICAL': 'S13.1',
    'LUXACAO DENTARIA': 'S03.2',
    'LUXACAO DO COTOVELO, NAO ESPECIFICADA': 'S53.1',
    'LUXACAO DO DEDO': 'S63.1',
    'LUXACAO DO JOELHO': 'S83.1',
    'LUXACAO DO MAXILAR': 'S03.0',
    'LUXACAO DO PUNHO': 'S63.0',
    'LUXACAO DO(S) ARTELHO(S)': 'S93.1',
    'LUXACAO ENTORSE E DISTENSAO DA ARTICULACAO E DOS LIGAMENTOS DO QUADRIL': 'S73',
    'LUXACAO ENTORSE E DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS AO NIVEL DO PUNHO E DA MAO': 'S63',
    'LUXACAO ENTORSE E DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS AO NIVEL DO TORNOZELO E DO PE': 'S93',
    'LUXACAO ENTORSE E DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS DA CINTURA ESCAPULAR': 'S43',
    'LUXACAO ENTORSE E DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS DO COTOVELO': 'S53',
    'LUXACAO ENTORSE E DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS DO JOELHO': 'S83',
    'LUXACAO ENTORSE E DISTENSAO DE ARTICULACOES E DOS LIGAMENTOS DO TORAX': 'S23',
    'LUXACAO ENTORSE OU DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS DA CABECA': 'S03',
    'LUXACAO ENTORSE OU DISTENSAO DAS ARTICULACOES E DOS LIGAMENTOS DO PESCOCO': 'S13',
    'LUXACAO, ENTORSE E DISTENSAO DE ARTICULACOES E LIGAMENTOS NAO ESPECIFICADOS DO TRONCO': 'T09.2',
    'LUXACAO, ENTORSE E DISTENSAO DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.3',
    'LUXACOES ENTORSES E DISTENSOES ENVOLVENDO REGIOES MULTIPLAS DO CORPO': 'T03',
    'LUXACOES MULTIPLAS DOS DEDOS': 'S63.2',
    'LUXACOES, ENTORSES E DISTENSOES ENVOLVENDO OUTRAS COMBINACOES DE REGIOES DO CORPO': 'T03.8',
    'LUXACOES, ENTORSES E DISTENSOES ENVOLVENDO REGIOES MULTIPLAS DE MEMBRO(S) SUPERIOR(ES)': 'T03.2',
    'LUXACOES, ENTORSES E DISTENSOES ENVOLVENDO REGIOES MULTIPLAS DO(S) MEMBRO(S) INFERIOR(ES)': 'T03.3',
    'LUXACOES, ENTORSES E DISTENSOES ENVOLVENDO REGIOES MULTIPLAS DOS MEMBROS SUPERIORES COM INFERIORES': 'T03.4',
    'LUXACOES, ENTORSES E DISTENSOES MULTIPLAS, NAO ESPECIFICADAS': 'T03.9',
    'MA ADAPTACAO AO TRABALHO': 'Z56.5',
    'MA-ABSORCAO DEVIDA A INTOLERANCIA NAO CLASSIFICADA EM OUTRA PARTE': 'K90.4',
    'MA-ABSORCAO INTESTINAL': 'K90',
    'MA-ABSORCAO INTESTINAL, SEM OUTRA ESPECIFICACAO': 'K90.9',
    'MACROGLOBULINEMIA DE WALDENSTROM': 'C88.0',
    'MACROGLOSSIA': 'Q38.2',
    'MAL ESTAR, FADIGA': 'R53',
    'MALARIA NAO ESPECIFICADA': 'B54',
    'MALARIA POR PLASMODIUM MALARIAE SEM COMPLICACOES': 'B52.9',
    'MALFORMACAO CONGENITA NAO ESPECIFICADA DO INTESTINO': 'Q43.9',
    'MALFORMACAO DO URACO': 'Q64.4',
    'MALFORMACOES CONGENITAS DA COLUNA VERTEBRAL E DOS OSSOS DO TORAX': 'Q76',
    'MALFORMACOES CONGENITAS DAS GRANDES ARTERIAS': 'Q25',
    'MALFORMACOES CONGENITAS DAS VALVAS AORTICA E MITRAL': 'Q23',
    'MALFORMACOES CONGENITAS DE OUTRAS GLANDULAS ENDOCRINAS': 'Q89.2',
    'MALFORMACOES CONGENITAS DO QUADRIL': 'Q65',
    'MALFORMACOES DOS VASOS CORONARIOS': 'Q24.5',
    'MALOCLUSAO, NAO ESPECIFICADA': 'K07.4',
    'MANCHAS CAFE-COM-LEITE': 'L81.3',
    'MANIA COM SINTOMAS PSICOTICOS': 'F30.2',
    'MANIA SEM SINTOMAS PSICOTICOS': 'F30.1',
    'MAO (PULSO) OU PE PENDENTE (ADQUIRIDO)': 'M21.3',
    'MAO E PE DE IMERSAO': 'T69.0',
    'MAO E PE EM GARRA E MAO E PE TORTOS ADQUIRIDOS': 'M21.5',
    'MARCHA ATAXICA': 'R26.0',
    'MASSA, TUMORACAO OU TUMEFACAO INTRA-ABDOMINAL E PELVICA': 'R19.0',
    'MASTITE INFECCIOSA NEONATAL': 'P39.0',
    'MASTITE NAO PURULENTA ASSOCIADA AO PARTO': 'O91.2',
    'MASTODINIA': 'N64.4',
    'MASTOIDITE AGUDA': 'H70.0',
    'MASTOIDITE CRONICA': 'H70.1',
    'MASTOIDITE E AFECCOES CORRELATAS': 'H70',
    'MASTOIDITE NAO ESPECIFICADA': 'H70.9',
    'MAU FUNCIONAMENTO DE ABERTURA EXTERNA (ESTOMA) DO TRATO URINARIO': 'N99.5',
    'MAU FUNCIONAMENTO DE COLOSTOMIA E ENTEROSTOMIA': 'K91.4',
    'MAU FUNCIONAMENTO DE TRAQUEOSTOMIA': 'J95.0',
    'MEDICAMENTO E SUBSTANCIA BIOLOGICA CONTAMINADOS, ADMINISTRADA POR MEIOS NAO ESPECIFICADOS': 'Y64.9',
    'MEDIDA PROFILATICA NAO ESPECIFICADA': 'Z29.9',
    'MEGACOLON NAO CLASSIFICADO EM OUTRA PARTE': 'K59.3',
    'MEGAESOFAGO NA DOENCA DE CHAGAS': 'K23.1',
    'MELANOMA MALIGNO DA ORELHA E DO CONDUTO AUDITIVO EXTERNO': 'C43.2',
    'MELANOMA MALIGNO DA PELE': 'C43',
    'MELANOMA MALIGNO DE OUTRAS PARTES E PARTES NAO ESPECIFICADAS DA FACE': 'C43.3',
    'MELENA': 'K92.1',
    'MENINGISMO': 'R29.1',
    'MENINGITE BACTERIANA NAO CLASSIFICADA EM OUTRA PARTE': 'G00',
    'MENINGITE BACTERIANA NAO ESPECIFICADA': 'G00.9',
    'MENINGITE E MENINGOENCEFALITE POR LISTERIA': 'A32.1',
    'MENINGITE EM DOENCAS VIRAIS CLASSIFICADAS EM OUTRA PARTE': 'G02.0',
    'MENINGITE EM OUTRAS DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'G02.8',
    'MENINGITE MENINGOCOCICA': 'A39.0',
    'MENINGITE NAO ESPECIFICADA': 'G03.9',
    'MENINGITE POR CAXUMBA [PAROTIDITE EPIDEMICA]': 'B26.1',
    'MENINGITE POR VARICELA': 'B01.0',
    'MENINGITE TUBERCULOSA': 'A17.0',
    'MENINGITE VIRAL': 'A87',
    'MENINGITE VIRAL NAO ESPECIFICADA': 'A87.9',
    'MENISCO CISTICO': 'M23.0',
    'MENSTRUACAO AUSENTE ESCASSA E POUCO FREQUENTE': 'N91',
    'MENSTRUACAO EXCESSIVA E FREQUENTE COM CICLO IRREGULAR': 'N92.1',
    'MENSTRUACAO EXCESSIVA E FREQUENTE COM CICLO REGULAR': 'N92.0',
    'MENSTRUACAO EXCESSIVA FREQUENTE E IRREGULAR': 'N92',
    'MENSTRUACAO EXCESSIVA NA PUBERDADE': 'N92.2',
    'MENSTRUACAO IRREGULAR, NAO ESPECIFICADA': 'N92.6',
    'MERALGIA PARESTESICA': 'G57.1',
    'MERGULHO OU PULO NA AGUA CAUSANDO OUTRO TRAUMATISMO QUE NAO AFOGAMENTO OU SUBMERSAO': 'W16',
    'MESOTELIOMA DO PERITONIO': 'C45.1',
    'METATARSALGIA': 'M77.4',
    'MIALGIA': 'M79.1',
    'MIALGIA EPIDEMICA': 'B33.0',
    'MIASTENIA GRAVIS': 'G70.0',
    'MICCAO DOLOROSA, NAO ESPECIFICADA': 'R30.9',
    'MICOSE FUNGOIDE': 'C84.0',
    'MICOSE NAO ESPECIFICADA': 'B49',
    'MICOSE SUPERFICIAL NAO ESPECIFICADA': 'B36.9',
    'MICOSES OPORTUNISTAS': 'B48.7',
    'MICROANGIOPATIA TROMBOTICA': 'M31.1',
    'MICROCEFALIA': 'Q02',
    'MIELOFIBROSE AGUDA': 'C94.5',
    'MIELOMA MULTIPLO': 'C90.0',
    'MIELOMA MULTIPLO E NEOPLASIAS MALIGNAS DE PLASMOCITOS': 'C90',
    'MIELOPATIA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G99.2',
    'MIIASE': 'B87',
    'MIIASE AURICULAR': 'B87.4',
    'MIIASE CUTANEA': 'B87.0',
    'MIIASE DAS FERIDAS': 'B87.1',
    'MIIASE DE OUTRAS LOCALIZACOES': 'B87.8',
    'MIIASE NAO ESPECIFICADA': 'B87.9',
    'MIIASE OCULAR': 'B87.2',
    'MILIARIA CRISTALINA': 'L74.1',
    'MILIARIA RUBRA': 'L74.0',
    'MILIARIA, NAO ESPECIFICADA': 'L74.3',
    'MIOCARDIOPATIA ISQUEMICA': 'I25.5',
    'MIOCLONIA': 'G25.3',
    'MIOPATIA ALCOOLICA': 'G72.1',
    'MIOPATIA INFLAMATORIA NAO CLASSIFICADA EM OUTRA PARTE': 'G72.4',
    'MIOPIA': 'H52.1',
    'MIOSITE': 'M60',
    'MIOSITE INFECCIOSA': 'M60.0',
    'MIOSITE NAO ESPECIFICADA': 'M60.9',
    'MIOSITE OSSIFICANTE PROGRESSIVA': 'M61.1',
    'MIRINGITE AGUDA': 'H73.0',
    'MODIFICACAO DURADOURA DA PERSONALIDADE APOS DOENCA PSIQUIATRICA': 'F62.1',
    'MOLUSCO CONTAGIOSO': 'B08.1',
    'MONOARTRITES NAO CLASSIFICADAS EM OUTRA PARTE': 'M13.1',
    'MONONEURITE MULTIPLA': 'G58.7',
    'MONONEUROPATIA DIABETICA': 'G59.0',
    'MONONEUROPATIA DOS MEMBROS INFERIORES, NAO ESPECIFICADA': 'G57.9',
    'MONONEUROPATIA DOS MEMBROS SUPERIORES, NAO ESPECIFICADA': 'G56.9',
    'MONONEUROPATIA NAO ESPECIFICADA': 'G58.9',
    'MONONEUROPATIAS DOS MEMBROS INFERIORES': 'G57',
    'MONONEUROPATIAS DOS MEMBROS SUPERIORES': 'G56',
    'MONONUCLEOSE INFECCIOSA': 'B27',
    'MONONUCLEOSE INFECCIOSA NAO ESPECIFICADA': 'B27.9',
    'MONONUCLEOSE POR CITOMEGALOVIRUS': 'B27.1',
    'MONOPLEGIA DO MEMBRO SUPERIOR': 'G83.2',
    'MORDEDURA DA MUCOSA DAS BOCHECHAS E DOS LABIOS': 'K13.1',
    'MORDEDURA DE RATO': 'W53',
    'MORDEDURA DE RATO - HABITACAO COLETIVA': 'W53.1',
    'MORDEDURA DE RATO - LOCAL NAO ESPECIFICADO': 'W53.9',
    'MORDEDURA DE RATO - RESIDENCIA': 'W53.0',
    'MORDEDURA OU ESMAGAMENTO PROVOCADO POR OUTROS REPTEIS': 'W59',
    'MORDEDURA OU ESMAGAMENTO PROVOCADO POR OUTROS REPTEIS - FAZENDA': 'W59.7',
    'MORDEDURA OU ESMAGAMENTO PROVOCADO POR OUTROS REPTEIS - LOCAL NAO ESPECIFICADO': 'W59.9',
    'MORDEDURA OU ESMAGAMENTO PROVOCADO POR OUTROS REPTEIS - RESIDENCIA': 'W59.0',
    'MORDEDURA OU GOLPE PROVOCADO POR CAO': 'W54',
    'MORDEDURA OU GOLPE PROVOCADO POR CROCODILO OU ALIGATOR': 'W58',
    'MORDEDURA OU GOLPE PROVOCADO POR OUTROS ANIMAIS MAMIFEROS': 'W55',
    'MORDEDURAS E PICADAS DE INSETO E DE OUTROS ARTROPODES NAO-VENENOSOS': 'W57',
    'MORMO': 'A24.0',
    'MORTE FETAL DE CAUSA NAO ESPECIFICADA': 'P95',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM OUTRO VEICULO NAO-MOTORIZADO': 'V26',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM AUTOMOVEL(CARRO), PICK-UP OU CAMINHONETE': 'V23',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM OBJETO FIXO OU PARADO': 'V27',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM PEDESTRE OU UM ANIMAL': 'V20',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM VEICULO A MOTOR DE DUAS OU TRES RODAS': 'V22',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM VEICULO A PEDAL': 'V21',
    'MOTOCICLISTA TRAUMATIZADO EM COLISAO COM UM VEICULO DE TRANSPORTE PESADO OU UM ONIBUS': 'V24',
    'MOTOCICLISTA TRAUMATIZADO EM UM ACIDENTE DE TRANSPORTE SEM COLISAO': 'V28',
    'MOTOCICLISTA [QUALQUER] TRAUMATIZADO EM OUTROS ACIDENTES DE TRANSPORTE ESPECIFICADOS': 'V29.8',
    'MOTOCICLISTA [QUALQUER] TRAUMATIZADO EM UM ACIDENTE DE TRANSITO NAO ESPECIFICADO': 'V29.9',
    'MOVIMENTOS ANORMAIS DA CABECA': 'R25.0',
    'MOVIMENTOS INVOLUNTARIOS ANORMAIS': 'R25',
    'MUCOCELE DE GLANDULA SALIVAR': 'K11.6',
    'MUCORMICOSE CUTANEA': 'B46.3',
    'MYCOPLASMA PNEUMONIAE [M. PNEUMONIAE], COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B96.0',
    'NAO ADMINISTRACAO DE DROGA, MEDICAMENTO OU SUBSTANCIA BIOLOGICA NECESSARIA': 'Y63.6',
    'NASOFARINGITE AGUDA [RESFRIADO COMUM]': 'J00',
    'NASOFARINGITE CRONICA': 'J31.1',
    'NAUSEA E VOMITOS': 'R11',
    'NECATORIASE': 'B76.1',
    'NECESSIDADE DE ASSISTENCIA COM CUIDADOS PESSOAIS': 'Z74.1',
    'NECESSIDADE DE IMUNIZACAO CONTRA A DIFTERIA-PERTUSSIS-TETANO COM POLIOMIELITE [DPT + POLIO]': 'Z27.3',
    'NECESSIDADE DE IMUNIZACAO CONTRA A RAIVA': 'Z24.2',
    'NECESSIDADE DE IMUNIZACAO SOMENTE CONTRA A INFLUENZA [GRIPE]': 'Z25.1',
    'NECESSIDADE DE IMUNIZACAO SOMENTE CONTRA O TETANO': 'Z23.5',
    'NECROBIOSE LIPOIDICA NAO CLASSIFICADA EM OUTRA PARTE': 'L92.1',
    'NECROLISE EPIDERMICA TOXICA [SINDROME DE LYELL]': 'L51.2',
    'NECROSE DA POLPA': 'K04.1',
    'NECROSE DO COTO DA AMPUTACAO': 'T87.5',
    'NEFRITE TUBULO-INTERSTICIAL AGUDA': 'N10',
    'NEFRITE TUBULO-INTERSTICIAL CRONICA': 'N11',
    'NEFRITE TUBULO-INTERSTICIAL CRONICA NAO ESPECIFICADA': 'N11.9',
    'NEFRITE TUBULO-INTERSTICIAL NAO ESPECIFICADA SE AGUDA OU CRONICA': 'N12',
    'NEFROPATIA HEREDITARIA NAO CLASSIFICADA EM OUTRA PARTE': 'N07',
    'NEFROPATIA HEREDITARIA NAO CLASSIFICADA EM OUTRA PARTE - ANORMALIDADE GLOMERULAR MINOR': 'N07.0',
    'NEFROPATIA HEREDITARIA NAO CLASSIFICADA EM OUTRA PARTE - LESOES GLOMERULARES FOCAIS E SEGMENTARES': 'N07.1',
    'NEFROPATIA HEREDITARIA NAO CLASSIFICADA EM OUTRA PARTE - NAO ESPECIFICADA': 'N07.9',
    'NEFROPATIA INDUZIDA POR DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS NAO ESPECIFICADAS': 'N14.2',
    'NEGLIGENCIA E ABANDONO': 'Y06',
    'NEGLIGENCIA E ABANDONO PELO ESPOSO OU COMPANHEIRO': 'Y06.0',
    'NEGLIGENCIA E ABANDONO POR OUTRA PESSOA ESPECIFICADA': 'Y06.8',
    'NEGLIGENCIA E ABANDONO POR PESSOA NAO ESPECIFICADA': 'Y06.9',
    'NEOPLASIA BENIGNA DA AMIGDALA': 'D10.4',
    'NEOPLASIA BENIGNA DA BEXIGA': 'D30.3',
    'NEOPLASIA BENIGNA DA GLANDULA PAROTIDA': 'D11.0',
    'NEOPLASIA BENIGNA DA GLANDULA SUPRA-RENAL (ADRENAL)': 'D35.0',
    'NEOPLASIA BENIGNA DA LARINGE': 'D14.1',
    'NEOPLASIA BENIGNA DA MAMA': 'D24',
    'NEOPLASIA BENIGNA DA MEDULA ESPINHAL': 'D33.4',
    'NEOPLASIA BENIGNA DA PELE DA ORELHA E DO CONDUTO AUDITIVO EXTERNO': 'D23.2',
    'NEOPLASIA BENIGNA DA PELE DA PALPEBRA, INCLUINDO O CANTO': 'D23.1',
    'NEOPLASIA BENIGNA DA PELE DO COURO CABELUDO E DO PESCOCO': 'D23.4',
    'NEOPLASIA BENIGNA DA PELE, NAO ESPECIFICADA': 'D23.9',
    'NEOPLASIA BENIGNA DA PROSTATA': 'D29.1',
    'NEOPLASIA BENIGNA DAS MENINGES': 'D32',
    'NEOPLASIA BENIGNA DAS MENINGES, NAO ESPECIFICADA': 'D32.9',
    'NEOPLASIA BENIGNA DE OUTRAS PARTES DA OROFARINGE': 'D10.5',
    'NEOPLASIA BENIGNA DE OUTROS ORGAOS INTRATORACICOS E DOS NAO ESPECIFICADOS': 'D15',
    'NEOPLASIA BENIGNA DO COLO DO UTERO': 'D26.0',
    'NEOPLASIA BENIGNA DO COLON RETO CANAL ANAL E ANUS': 'D12',
    'NEOPLASIA BENIGNA DO ENCEFALO E DE OUTRAS PARTES DO SISTEMA NERVOSO CENTRAL': 'D33',
    'NEOPLASIA BENIGNA DO ESOFAGO': 'D13.0',
    'NEOPLASIA BENIGNA DO FIGADO': 'D13.4',
    'NEOPLASIA BENIGNA DO OUVIDO MEDIO E DO APARELHO RESPIRATORIO': 'D14',
    'NEOPLASIA BENIGNA DO OVARIO': 'D27',
    'NEOPLASIA BENIGNA DO PANCREAS': 'D13.6',
    'NEOPLASIA BENIGNA DO PENIS': 'D29.0',
    'NEOPLASIA BENIGNA DO TECIDO CONJUNTIVO E OUTROS TECIDOS MOLES DA CABECA, FACE E PESCOCO': 'D21.0',
    'NEOPLASIA BENIGNA DOS BRONQUIOS E PULMAO': 'D14.3',
    'NEOPLASIA BENIGNA DOS GANGLIOS LINFATICOS (LINFONODOS)': 'D36.0',
    'NEOPLASIA BENIGNA DOS LABIOS': 'D10.0',
    'NEOPLASIA BENIGNA DOS ORGAOS GENITAIS MASCULINOS': 'D29',
    'NEOPLASIA BENIGNA DOS ORGAOS URINARIOS': 'D30',
    'NEOPLASIA BENIGNA DOS OSSOS LONGOS DOS MEMBROS INFERIORES': 'D16.2',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA BEXIGA': 'D41.4',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA GLANDULA SUPRA-RENAL (ADRENAL)': 'D44.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA LARINGE': 'D38.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA MAMA': 'D48.6',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA PELVE RENAL': 'D41.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA PROSTATA': 'D40.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DA TRAQUEIA, BRONQUIOS E PULMAO': 'D38.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DAS MENINGES CEREBRAIS': 'D42.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DE ORGAO DIGESTIVO, NAO ESPECIFICADO': 'D37.9',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DE ORGAO GENITAL MASCULINO, NAO ESPECIFICADO': 'D40.9',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DE ORGAO RESPIRATORIO, NAO ESPECIFICADO': 'D38.6',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DE ORGAO URINARIO, NAO ESPECIFICADO': 'D41.9',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO ENCEFALO E DO SISTEMA NERVOSO CENTRAL': 'D43',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO ESTOMAGO': 'D37.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO LABIO, CAVIDADE ORAL E FARINGE': 'D37.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO OVARIO': 'D39.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO RETO': 'D37.5',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO RIM': 'D41.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO SISTEMA NERVOSO CENTRAL, NAO ESPECIFICADO': 'D43.9',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO TESTICULO': 'D40.1',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DO UTERO': 'D39.0',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DOS COLONS': 'D37.4',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DOS ORGAOS GENITAIS MASCULINOS': 'D40',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO DOS ORGAOS URINARIOS': 'D41',
    'NEOPLASIA DE COMPORTAMENTO INCERTO OU DESCONHECIDO SEM OUTRA ESPECIFICACAO': 'D48.9',
    'NEOPLASIA LIPOMATOSA BENIGNA': 'D17',
    'NEOPLASIA LIPOMATOSA BENIGNA DA PELE E DO TECIDO SUBCUTANEO DA CABECA, FACE E PESCOCO': 'D17.0',
    'NEOPLASIA LIPOMATOSA BENIGNA DA PELE E TECIDO SUBCUTANEO DO TRONCO': 'D17.1',
    'NEOPLASIA LIPOMATOSA BENIGNA DA PELE E TECIDO SUBCUTANEO DOS MEMBROS': 'D17.2',
    'NEOPLASIA LIPOMATOSA BENIGNA DE LOCALIZACAO NAO ESPECIFICADA': 'D17.9',
    'NEOPLASIA MALIGNA DA AMIGDALA COM LESAO INVASIVA': 'C09.8',
    'NEOPLASIA MALIGNA DA BASE DA LINGUA': 'C01',
    'NEOPLASIA MALIGNA DA BEXIGA': 'C67',
    'NEOPLASIA MALIGNA DA BEXIGA COM LESAO INVASIVA': 'C67.8',
    'NEOPLASIA MALIGNA DA BEXIGA, SEM OUTRA ESPECIFICACOES': 'C67.9',
    'NEOPLASIA MALIGNA DA BOCA, NAO ESPECIFICADA': 'C06.9',
    'NEOPLASIA MALIGNA DA CABECA DO PANCREAS': 'C25.0',
    'NEOPLASIA MALIGNA DA CABECA, FACE E PESCOCO': 'C76.0',
    'NEOPLASIA MALIGNA DA CARDIA': 'C16.0',
    'NEOPLASIA MALIGNA DA CAUDA DO PANCREAS': 'C25.2',
    'NEOPLASIA MALIGNA DA CAVIDADE NASAL E DO OUVIDO MEDIO': 'C30',
    'NEOPLASIA MALIGNA DA COLUNA VERTEBRAL': 'C41.2',
    'NEOPLASIA MALIGNA DA FACE DORSAL DA LINGUA': 'C02.0',
    'NEOPLASIA MALIGNA DA FARINGE, NAO ESPECIFICADA': 'C14.0',
    'NEOPLASIA MALIGNA DA FLEXURA (ANGULO) HEPATICA(O)': 'C18.3',
    'NEOPLASIA MALIGNA DA FOSSA AMIGDALIANA': 'C09.0',
    'NEOPLASIA MALIGNA DA GENGIVA, NAO ESPECIFICADA': 'C03.9',
    'NEOPLASIA MALIGNA DA GLANDULA PARATIREOIDE': 'C75.0',
    'NEOPLASIA MALIGNA DA GLANDULA PAROTIDA': 'C07',
    'NEOPLASIA MALIGNA DA GLANDULA SUBMANDIBULAR': 'C08.0',
    'NEOPLASIA MALIGNA DA GLANDULA SUPRA-RENAL, NAO ESPECIFICADA': 'C74.9',
    'NEOPLASIA MALIGNA DA GLANDULA TIREOIDE': 'C73',
    'NEOPLASIA MALIGNA DA GLOTE': 'C32.0',
    'NEOPLASIA MALIGNA DA HIPOFARINGE, NAO ESPECIFICADA': 'C13.9',
    'NEOPLASIA MALIGNA DA JUNCAO RETOSSIGMOIDE': 'C19',
    'NEOPLASIA MALIGNA DA LARINGE': 'C32',
    'NEOPLASIA MALIGNA DA LARINGE COM LESAO INVASIVA': 'C32.8',
    'NEOPLASIA MALIGNA DA LARINGE, NAO ESPECIFICADA': 'C32.9',
    'NEOPLASIA MALIGNA DA LINGUA COM LESAO INVASIVA': 'C02.8',
    'NEOPLASIA MALIGNA DA LINGUA, NAO ESPECIFICADA': 'C02.9',
    'NEOPLASIA MALIGNA DA MAMA': 'C50',
    'NEOPLASIA MALIGNA DA MAMA COM LESAO INVASIVA': 'C50.8',
    'NEOPLASIA MALIGNA DA MAMA, NAO ESPECIFICADA': 'C50.9',
    'NEOPLASIA MALIGNA DA MANDIBULA': 'C41.1',
    'NEOPLASIA MALIGNA DA MEDULA ESPINHAL': 'C72.0',
    'NEOPLASIA MALIGNA DA NASOFARINGE': 'C11',
    'NEOPLASIA MALIGNA DA NASOFARINGE COM LESAO INVASIVA': 'C11.8',
    'NEOPLASIA MALIGNA DA OROFARINGE': 'C10',
    'NEOPLASIA MALIGNA DA OROFARINGE COM LESAO INVASIVA': 'C10.8',
    'NEOPLASIA MALIGNA DA OROFARINGE, NAO ESPECIFICADA': 'C10.9',
    'NEOPLASIA MALIGNA DA PAREDE LATERAL DA OROFARINGE': 'C10.2',
    'NEOPLASIA MALIGNA DA PELE COM LESAO INVASIVA': 'C44.8',
    'NEOPLASIA MALIGNA DA PELE DA ORELHA E DO CONDUTO AUDITIVO EXTERNO': 'C44.2',
    'NEOPLASIA MALIGNA DA PELE DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA FACE': 'C44.3',
    'NEOPLASIA MALIGNA DA PELE DO MEMBRO INFERIOR, INCLUINDO QUADRIL': 'C44.7',
    'NEOPLASIA MALIGNA DA PELE DO MEMBRO SUPERIOR, INCLUINDO OMBRO': 'C44.6',
    'NEOPLASIA MALIGNA DA PELE DO TRONCO': 'C44.5',
    'NEOPLASIA MALIGNA DA PELE, NAO ESPECIFICADA': 'C44.9',
    'NEOPLASIA MALIGNA DA PELVE': 'C76.3',
    'NEOPLASIA MALIGNA DA PELVE RENAL': 'C65',
    'NEOPLASIA MALIGNA DA PORCAO CERVICAL DO ESOFAGO (ESOFAGO CERVICAL)': 'C15.0',
    'NEOPLASIA MALIGNA DA PORCAO TORACICA DO ESOFAGO (ESOFAGO TORACICO)': 'C15.1',
    'NEOPLASIA MALIGNA DA PROSTATA': 'C61',
    'NEOPLASIA MALIGNA DA REGIAO POS-CRICOIDEA': 'C13.0',
    'NEOPLASIA MALIGNA DA URETRA': 'C68.0',
    'NEOPLASIA MALIGNA DA VESICULA BILIAR': 'C23',
    'NEOPLASIA MALIGNA DA VIA BILIAR, NAO ESPECIFICADA': 'C24.9',
    'NEOPLASIA MALIGNA DA VULVA': 'C51',
    'NEOPLASIA MALIGNA DAS CARTILAGENS DA LARINGE': 'C32.3',
    'NEOPLASIA MALIGNA DAS VIAS BILIARES EXTRA-HEPATICAS': 'C24.0',
    'NEOPLASIA MALIGNA DE LOCALIZACOES MAL DEFINIDAS DENTRO DO APARELHO DIGESTIVO': 'C26.9',
    'NEOPLASIA MALIGNA DE ORGAO URINARIO, NAO ESPECIFICADO': 'C68.9',
    'NEOPLASIA MALIGNA DE OUTRAS LOCALIZACOES E DAS MAL DEFINIDAS COM LESAO INVASIVA': 'C76.8',
    'NEOPLASIA MALIGNA DE OUTRAS LOCALIZACOES E DE LOCALIZACOES MAL DEFINIDAS': 'C76',
    'NEOPLASIA MALIGNA DE OUTRAS LOCALIZACOES MAL DEFINIDAS': 'C76.7',
    'NEOPLASIA MALIGNA DE OUTRAS PARTES DO PANCREAS': 'C25.7',
    'NEOPLASIA MALIGNA DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA BOCA': 'C06',
    'NEOPLASIA MALIGNA DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA BOCA COM LESAO INVASIVA': 'C06.8',
    'NEOPLASIA MALIGNA DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA LINGUA': 'C02',
    'NEOPLASIA MALIGNA DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DAS VIAS BILIARES': 'C24',
    'NEOPLASIA MALIGNA DE OUTROS ORGAOS DIGESTIVOS E DE LOCALIZACOES MAL DEFINIDAS NO APARELHO DIGESTIVO': 'C26',
    'NEOPLASIA MALIGNA DE OUTROS ORGAOS URINARIOS E DOS NAO ESPECIFICADOS': 'C68',
    'NEOPLASIA MALIGNA DO ABDOME': 'C76.2',
    'NEOPLASIA MALIGNA DO ANUS E DO CANAL ANAL': 'C21',
    'NEOPLASIA MALIGNA DO ANUS, NAO ESPECIFICADO': 'C21.0',
    'NEOPLASIA MALIGNA DO ASSOALHO ANTERIOR DA BOCA': 'C04.0',
    'NEOPLASIA MALIGNA DO ASSOALHO DA BOCA': 'C04',
    'NEOPLASIA MALIGNA DO ASSOALHO DA BOCA, NAO ESPECIFICADO': 'C04.9',
    'NEOPLASIA MALIGNA DO ASSOALHO LATERAL DA BOCA': 'C04.1',
    'NEOPLASIA MALIGNA DO BRONQUIO PRINCIPAL': 'C34.0',
    'NEOPLASIA MALIGNA DO CANAL ANAL': 'C21.1',
    'NEOPLASIA MALIGNA DO CECO': 'C18.0',
    'NEOPLASIA MALIGNA DO CEREBELO': 'C71.6',
    'NEOPLASIA MALIGNA DO CEREBRO, EXCETO LOBOS E VENTRICULOS': 'C71.0',
    'NEOPLASIA MALIGNA DO COLO DO UTERO': 'C53',
    'NEOPLASIA MALIGNA DO COLO DO UTERO COM LESAO INVASIVA': 'C53.8',
    'NEOPLASIA MALIGNA DO COLO DO UTERO, NAO ESPECIFICADO': 'C53.9',
    'NEOPLASIA MALIGNA DO COLON': 'C18',
    'NEOPLASIA MALIGNA DO COLON DESCENDENTE': 'C18.6',
    'NEOPLASIA MALIGNA DO COLON SIGMOIDE': 'C18.7',
    'NEOPLASIA MALIGNA DO COLON TRANSVERSO': 'C18.4',
    'NEOPLASIA MALIGNA DO COLON, NAO ESPECIFICADO': 'C18.9',
    'NEOPLASIA MALIGNA DO CORACAO MEDIASTINO E PLEURA': 'C38',
    'NEOPLASIA MALIGNA DO CORPO DO ESTOMAGO': 'C16.2',
    'NEOPLASIA MALIGNA DO CORPO DO UTERO': 'C54',
    'NEOPLASIA MALIGNA DO CORPO DO UTERO COM LESAO INVASIVA': 'C54.8',
    'NEOPLASIA MALIGNA DO CORPO DO UTERO, NAO ESPECIFICADO': 'C54.9',
    'NEOPLASIA MALIGNA DO ENCEFALO': 'C71',
    'NEOPLASIA MALIGNA DO ENCEFALO COM LESAO INVASIVA': 'C71.8',
    'NEOPLASIA MALIGNA DO ENCEFALO, NAO ESPECIFICADO': 'C71.9',
    'NEOPLASIA MALIGNA DO ENDOCERVIX': 'C53.0',
    'NEOPLASIA MALIGNA DO ENDOMETRIO': 'C54.1',
    'NEOPLASIA MALIGNA DO ESOFAGO': 'C15',
    'NEOPLASIA MALIGNA DO ESOFAGO, NAO ESPECIFICADO': 'C15.9',
    'NEOPLASIA MALIGNA DO ESTOMAGO': 'C16',
    'NEOPLASIA MALIGNA DO ESTOMAGO COM LESAO INVASIVA': 'C16.8',
    'NEOPLASIA MALIGNA DO ESTOMAGO, NAO ESPECIFICADO': 'C16.9',
    'NEOPLASIA MALIGNA DO FIGADO E DAS VIAS BILIARES INTRA-HEPATICAS': 'C22',
    'NEOPLASIA MALIGNA DO FIGADO, NAO ESPECIFICADA': 'C22.9',
    'NEOPLASIA MALIGNA DO FUNDO DO ESTOMAGO': 'C16.1',
    'NEOPLASIA MALIGNA DO INTESTINO DELGADO': 'C17',
    'NEOPLASIA MALIGNA DO INTESTINO DELGADO COM LESAO INVASIVA': 'C17.8',
    'NEOPLASIA MALIGNA DO INTESTINO DELGADO, NAO ESPECIFICADO': 'C17.9',
    'NEOPLASIA MALIGNA DO ISTMO DO UTERO': 'C54.0',
    'NEOPLASIA MALIGNA DO JEJUNO': 'C17.1',
    'NEOPLASIA MALIGNA DO LABIO': 'C00',
    'NEOPLASIA MALIGNA DO LABIO EXTERNO, NAO ESPECIFICADO': 'C00.2',
    'NEOPLASIA MALIGNA DO LABIO INFERIOR, FACE INTERNA': 'C00.4',
    'NEOPLASIA MALIGNA DO LABIO, CAVIDADE ORAL E FARINGE COM LESAO INVASIVA': 'C14.8',
    'NEOPLASIA MALIGNA DO LABIO, SEM ESPECIFICACAO, FACE INTERNA': 'C00.5',
    'NEOPLASIA MALIGNA DO LOBO INFERIOR, BRONQUIO OU PULMAO': 'C34.3',
    'NEOPLASIA MALIGNA DO LOBO MEDIO, BRONQUIO OU PULMAO': 'C34.2',
    'NEOPLASIA MALIGNA DO LOBO SUPERIOR, BRONQUIO OU PULMAO': 'C34.1',
    'NEOPLASIA MALIGNA DO MEDIASTINO ANTERIOR': 'C38.1',
    'NEOPLASIA MALIGNA DO MEMBRO SUPERIOR': 'C76.4',
    'NEOPLASIA MALIGNA DO OLHO E ANEXOS': 'C69',
    'NEOPLASIA MALIGNA DO OLHO E ANEXOS COM LESAO INVASIVA': 'C69.8',
    'NEOPLASIA MALIGNA DO OVARIO': 'C56',
    'NEOPLASIA MALIGNA DO PALATO': 'C05',
    'NEOPLASIA MALIGNA DO PALATO MOLE': 'C05.1',
    'NEOPLASIA MALIGNA DO PANCREAS': 'C25',
    'NEOPLASIA MALIGNA DO PANCREAS COM LESAO INVASIVA': 'C25.8',
    'NEOPLASIA MALIGNA DO PANCREAS ENDOCRINO': 'C25.4',
    'NEOPLASIA MALIGNA DO PANCREAS, NAO ESPECIFICADO': 'C25.9',
    'NEOPLASIA MALIGNA DO PENIS': 'C60',
    'NEOPLASIA MALIGNA DO PILORO': 'C16.4',
    'NEOPLASIA MALIGNA DO QUADRANTE SUPERIOR EXTERNO DA MAMA': 'C50.4',
    'NEOPLASIA MALIGNA DO RETO': 'C20',
    'NEOPLASIA MALIGNA DO RETO, ANUS E DO CANAL ANAL COM LESAO INVASIVA': 'C21.8',
    'NEOPLASIA MALIGNA DO RETROPERITONIO': 'C48.0',
    'NEOPLASIA MALIGNA DO RIM, EXCETO PELVE RENAL': 'C64',
    'NEOPLASIA MALIGNA DO TECIDO CONJUNTIVO E DE OUTROS TECIDOS MOLES': 'C49',
    'NEOPLASIA MALIGNA DO TECIDO CONJUNTIVO E TECIDOS MOLES DA CABECA, FACE E PESCOCO': 'C49.0',
    'NEOPLASIA MALIGNA DO TERCO SUPERIOR DO ESOFAGO': 'C15.3',
    'NEOPLASIA MALIGNA DO TESTICULO CRIPTORQUIDICO': 'C62.0',
    'NEOPLASIA MALIGNA DO TESTICULO, SEM OUTRAS ESPECIFICACOES': 'C62.9',
    'NEOPLASIA MALIGNA DO TIMO': 'C37',
    'NEOPLASIA MALIGNA DO TRATO INTESTINAL, PARTE NAO ESPECIFICADA': 'C26.0',
    'NEOPLASIA MALIGNA DO TRIGONO DA BEXIGA': 'C67.0',
    'NEOPLASIA MALIGNA DO UTERO, PORCAO NAO ESPECIFICADA': 'C55',
    'NEOPLASIA MALIGNA DOS BRONQUIOS E DOS PULMOES': 'C34',
    'NEOPLASIA MALIGNA DOS BRONQUIOS E DOS PULMOES COM LESAO INVASIVA': 'C34.8',
    'NEOPLASIA MALIGNA DOS BRONQUIOS OU PULMOES, NAO ESPECIFICADO': 'C34.9',
    'NEOPLASIA MALIGNA DOS NERVOS PERIFERICOS DA CABECA, FACE E PESCOCO': 'C47.0',
    'NEOPLASIA MALIGNA DOS OSSOS CURTOS DOS MEMBROS SUPERIORES': 'C40.1',
    'NEOPLASIA MALIGNA DOS OSSOS DO CRANIO E DA FACE': 'C41.0',
    'NEOPLASIA MALIGNA DOS OSSOS E CARTILAGENS ARTICULARES DOS MEMBROS': 'C40',
    'NEOPLASIA MALIGNA DOS OSSOS E CARTILAGENS ARTICULARES, NAO ESPECIFICADOS': 'C41.9',
    'NEOPLASIA MALIGNA DOS OSSOS LONGOS DOS MEMBROS INFERIORES': 'C40.2',
    'NEOPLASIA MALIGNA DOS SEIOS DA FACE': 'C31',
    'NEOPLASIA MALIGNA DOS TECIDOS MOLES DO RETROPERITONIO E DO PERITONIO': 'C48',
    'NEOPLASIA MALIGNA SECUNDARIA DE OUTRA LOCALIZACAO ESPECIFICADA': 'C79.8',
    'NEOPLASIA MALIGNA SECUNDARIA DE OUTRAS LOCALIZACOES': 'C79',
    'NEOPLASIA MALIGNA SECUNDARIA DO ENCEFALO E DAS MENINGES CEREBRAIS': 'C79.3',
    'NEOPLASIA MALIGNA SECUNDARIA DO FIGADO': 'C78.7',
    'NEOPLASIA MALIGNA SECUNDARIA DO INTESTINO DELGADO': 'C78.4',
    'NEOPLASIA MALIGNA SECUNDARIA DO INTESTINO GROSSO E DO RETO': 'C78.5',
    'NEOPLASIA MALIGNA SECUNDARIA DO MEDIASTINO': 'C78.1',
    'NEOPLASIA MALIGNA SECUNDARIA DO OVARIO': 'C79.6',
    'NEOPLASIA MALIGNA SECUNDARIA DO RETROPERITONIO E DO PERITONIO': 'C78.6',
    'NEOPLASIA MALIGNA SECUNDARIA DOS ORGAOS RESPIRATORIOS E DIGESTIVOS': 'C78',
    'NEOPLASIA MALIGNA SECUNDARIA DOS OSSOS E DA MEDULA OSSEA': 'C79.5',
    'NEOPLASIA MALIGNA SECUNDARIA DOS PULMOES': 'C78.0',
    'NEOPLASIA MALIGNA SECUNDARIA E NAO ESPECIFICADA DOS GANGLIOS LINFATICOS': 'C77',
    'NEOPLASIA MALIGNA SECUNDARIA E NAO ESPECIFICADA DOS GANGLIOS LINFATICOS INTRA-ABDOMINAIS': 'C77.2',
    'NEOPLASIA MALIGNA, SEM ESPECIFICACAO DE LOCALIZACAO': 'C80',
    'NEOPLASIAS MALIGNAS DE LOCALIZACOES MULTIPLAS INDEPENDENTES (PRIMARIAS)': 'C97',
    'NERVOSISMO': 'R45.0',
    'NEURASTENIA': 'F48.0',
    'NEURITE OPTICA': 'H46',
    'NEUROMIELITE OPTICA [DOENCA DE DEVIC]': 'G36.0',
    'NEUROMIOPATIA E NEUROPATIA PARANEOPLASICAS': 'G13.0',
    'NEURONITE VESTIBULAR': 'H81.2',
    'NEUROPATIA AUTONOMICA EM DOENCAS ENDOCRINAS E METABOLICAS': 'G99.0',
    'NEUROPATIA AUTONOMICA PERIFERICA IDIOPATICA': 'G90.0',
    'NEUROPATIA HEREDITARIA E IDIOPATICA': 'G60',
    'NEUROPATIA HEREDITARIA E IDIOPATICA NAO ESPECIFICADA': 'G60.9',
    'NEUROPATIA HEREDITARIA MOTORA E SENSORIAL': 'G60.0',
    'NEUROPATIA INTERCOSTAL': 'G58.0',
    'NEUROPATIA PROGRESSIVA IDIOPATICA': 'G60.3',
    'NEUROPATIA SERICA': 'G61.1',
    'NEUROSSIFILIS CONGENITA TARDIA [NEUROSSIFILIS JUVENIL]': 'A50.4',
    'NEUROSSIFILIS NAO ESPECIFICADA': 'A52.3',
    'NEVO MELANOCITICO DE OUTRAS PARTES E DE PARTES NAO ESPECIFICADAS DA FACE': 'D22.3',
    'NEVO MELANOCITICO DO TRONCO': 'D22.5',
    'NEVO NAO-NEOPLASICO': 'I78.1',
    'NEVOS MELANOCITICOS': 'D22',
    'NEVRALGIA DO TRIGEMEO': 'G50.0',
    'NEVRALGIA E NEURITE NAO ESPECIFICADAS': 'M79.2',
    'NEVRALGIA POS-ZOSTER': 'G53.0',
    'NODULO MAMARIO NAO ESPECIFICADO': 'N63',
    'OBESIDADE': 'E66',
    'OBESIDADE DEVIDA A EXCESSO DE CALORIAS': 'E66.0',
    'OBESIDADE NAO ESPECIFICADA': 'E66.9',
    'OBJETO DE DISCRIMINACAO E PERSEGUICAO PERCEBIDAS': 'Z60.5',
    'OBJETO ESTRANHO DEIXADO ACIDENTALMENTE NO CORPO DURANTE A PRESTACAO DE CUIDADOS CIRURGICOS E MEDICOS': 'Y61',
    'OBJETO ESTRANHO DEIXADO ACIDENTALMENTE NO CORPO DURANTE INJECAO OU VACINACAO (IMUNIZACAO)': 'Y61.3',
    'OBSERVACAO POR SUSPEITA DE DOENCA OU AFECCAO NAO ESPECIFICADA': 'Z03.9',
    'OBSERVACAO POR SUSPEITA DE EFEITO TOXICO DE SUBSTANCIA INGERIDA': 'Z03.6',
    'OBSERVACAO POR SUSPEITA DE INFARTO DO MIOCARDIO': 'Z03.4',
    'OBSERVACAO POR SUSPEITA DE NEOPLASIA MALIGNA': 'Z03.1',
    'OBSERVACAO POR SUSPEITA DE OUTRAS DOENCAS E AFECCOES': 'Z03.8',
    'OBSERVACAO POR SUSPEITA DE TRANSTORNOS MENTAIS E DO COMPORTAMENTO': 'Z03.2',
    'OBSERVACAO POR SUSPEITA DE TUBERCULOSE': 'Z03.0',
    'OBSTRUCAO DA TROMPA DE EUSTAQUIO': 'H68.1',
    'OBSTRUCAO DA VESICULA BILIAR': 'K82.0',
    'OBSTRUCAO DE VIA BILIAR': 'K83.1',
    'OBSTRUCAO DO COLO DA BEXIGA': 'N32.0',
    'OBSTRUCAO DO DUODENO': 'K31.5',
    'OBSTRUCAO DO ESOFAGO': 'K22.2',
    'OBSTRUCAO INTESTINAL POS-OPERATORIA': 'K91.3',
    'OBTENCAO DE ATESTADO MEDICO': 'Z02.7',
    'OCLUSAO ARTERIAL RETINIANA TRANSITORIA': 'H34.0',
    'OCLUSAO E ESTENOSE DA ARTERIA CAROTIDA': 'I65.2',
    'OCLUSAO VASCULAR RETINIANA NAO ESPECIFICADA': 'H34.9',
    'OCUPANTE DE UM AUTOMOVEL (CARRO) TRAUMATIZADO EM COLISAO COM OUTRO VEICULO NAO-MOTORIZADO': 'V46',
    'OCUPANTE DE UM AUTOMOVEL (CARRO) TRAUMATIZADO EM COLISAO COM UM OBJETO FIXO OU PARADO': 'V47',
    'OCUPANTE DE UM AUTOMOVEL (CARRO) TRAUMATIZADO EM COLISAO COM UM PEDESTRE OU UM ANIMAL': 'V40',
    'OCUPANTE DE UM AUTOMOVEL (CARRO) TRAUMATIZADO EM COLISAO COM UM VEICULO A PEDAL': 'V41',
    'OCUPANTE DE UM AUTOMOVEL (CARRO) TRAUMATIZADO EM UM ACIDENTE DE TRANSPORTE SEM COLISAO': 'V48',
    'OCUPANTE DE UM ONIBUS TRAUMATIZADO EM COLISAO COM UM VEICULO DE TRANSPORTE PESADO OU UM ONIBUS': 'V74',
    'OCUPANTE DE UM ONIBUS TRAUMATIZADO EM UM ACIDENTE DE TRANSPORTE SEM COLISAO': 'V78',
    'OCUPANTE DE UM TRICICLO MOTORIZADO TRAUMATIZADO EM COLISAO COM UM PEDESTRE OU UM ANIMAL': 'V30',
    'OCUPANTE DE UM VEICULO DE TRANSPORTE PESADO TRAUMATIZADO EM UM ACIDENTE DE TRANSPORTE SEM COLISAO': 'V68',
    'ODONTOCLASIA': 'K02.4',
    'OLIGOMENORREIA, NAO ESPECIFICADA': 'N91.5',
    'ONFALITE DO RECEM-NASCIDO COM OU SEM HEMORRAGIA LEVE': 'P38',
    'ONICOGRIFOSE': 'L60.2',
    'ONICOLISE': 'L60.1',
    'ORGAOS E TECIDOS TRANSPLANTADOS': 'Z94',
    'ORIENTACAO SEXUAL EGODISTONICA': 'F66.1',
    'ORIFICIO ARTIFICIAL NAO ESPECIFICADO': 'Z93.9',
    'ORIFICIOS ARTIFICIAIS': 'Z93',
    'ORQUITE E EPIDIDIMITE': 'N45',
    'ORQUITE POR CAXUMBA [PAROTIDITE EPIDEMICA]': 'B26.0',
    'ORQUITE, EPIDIDIMITE E EPIDIDIMO-ORQUITE, COM MENCAO DE ABSCESSO': 'N45.0',
    'ORQUITE, EPIDIDIMITE E EPIDIDIMO-ORQUITE, SEM MENCAO DE ABSCESSO': 'N45.9',
    'OSTEOCONDRITE DISSECANTE': 'M93.2',
    'OSTEOCONDRODISPLASIA NAO ESPECIFICADA': 'Q78.9',
    'OSTEOCONDROPATIAS, NAO ESPECIFICADA': 'M93.9',
    'OSTEOCONDROSE DA COLUNA VERTEBRAL': 'M42',
    'OSTEOCONDROSE JUVENIL, NAO ESPECIFICADA': 'M92.9',
    'OSTEOCONDROSE VERTEBRAL JUVENIL': 'M42.0',
    'OSTEOCONDROSE VERTEBRAL, NAO ESPECIFICADA': 'M42.9',
    'OSTEOFITO': 'M25.7',
    'OSTEOGENESE IMPERFEITA': 'Q78.0',
    'OSTEOMALACIA NAO ESPECIFICADA DO ADULTO': 'M83.9',
    'OSTEOMIELITE': 'M86',
    'OSTEOMIELITE CRONICA COM SEIO DRENANTE': 'M86.4',
    'OSTEOMIELITE CRONICA MULTIFOCAL': 'M86.3',
    'OSTEOMIELITE DAS VERTEBRAS': 'M46.2',
    'OSTEOMIELITE NAO ESPECIFICADA': 'M86.9',
    'OSTEOMIELITE SUBAGUDA': 'M86.2',
    'OSTEONECROSE': 'M87',
    'OSTEONECROSE EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M90.5',
    'OSTEONECROSE NAO ESPECIFICADA': 'M87.9',
    'OSTEOPATIA POS-POLIOMIELITE': 'M89.6',
    'OSTEOPATIAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M90',
    'OSTEOPOROSE COM FRATURA PATOLOGICA': 'M80',
    'OSTEOPOROSE DE DESUSO': 'M81.2',
    'OSTEOPOROSE DE DESUSO COM FRATURA PATOLOGICA': 'M80.2',
    'OSTEOPOROSE IDIOPATICA': 'M81.5',
    'OSTEOPOROSE NAO ESPECIFICADA': 'M81.9',
    'OSTEOPOROSE POR MA-ABSORCAO POS-CIRURGICA COM FRATURA PATOLOGICA': 'M80.3',
    'OSTEOPOROSE POS-MENOPAUSICA': 'M81.0',
    'OSTEOPOROSE SEM FRATURA PATOLOGICA': 'M81',
    'OTALGIA': 'H92.0',
    'OTALGIA E SECRECAO AUDITIVA': 'H92',
    'OTITE BAROTRAUMATICA': 'T70.0',
    'OTITE EXTERNA': 'H60',
    'OTITE EXTERNA AGUDA NAO-INFECCIOSA': 'H60.5',
    'OTITE EXTERNA EM DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'H62.0',
    'OTITE EXTERNA EM DOENCAS VIRAIS CLASSIFICADAS EM OUTRA PARTE': 'H62.1',
    'OTITE EXTERNA EM MICOSES': 'H62.2',
    'OTITE EXTERNA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H62.4',
    'OTITE EXTERNA EM OUTRAS DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'H62.3',
    'OTITE EXTERNA MALIGNA': 'H60.2',
    'OTITE EXTERNA NAO ESPECIFICADA': 'H60.9',
    'OTITE MEDIA AGUDA SEROSA': 'H65.0',
    'OTITE MEDIA AGUDA SUPURATIVA': 'H66.0',
    'OTITE MEDIA EM DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'H67.0',
    'OTITE MEDIA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H67',
    'OTITE MEDIA EM DOENCAS VIRAIS CLASSIFICADAS EM OUTRA PARTE': 'H67.1',
    'OTITE MEDIA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H67.8',
    'OTITE MEDIA MUCOIDE CRONICA': 'H65.3',
    'OTITE MEDIA NAO ESPECIFICADA': 'H66.9',
    'OTITE MEDIA NAO-SUPURATIVA': 'H65',
    'OTITE MEDIA NAO-SUPURATIVA, NAO ESPECIFICADA': 'H65.9',
    'OTITE MEDIA SEROSA CRONICA': 'H65.2',
    'OTITE MEDIA SUPURATIVA E AS NAO ESPECIFICADAS': 'H66',
    'OTITE MEDIA SUPURATIVA NAO ESPECIFICADA': 'H66.4',
    'OTITE MEDIA TUBOTIMPANICA SUPURATIVA CRONICA': 'H66.1',
    'OTORRAGIA': 'H92.2',
    'OTORREIA': 'H92.1',
    'OUTRA AMNESIA': 'R41.3',
    'OUTRA CONTRATURA DE TENDAO (BAINHA)': 'M67.1',
    'OUTRA DEFORMIDADE DO HALLUX (ADQUIRIDA)': 'M20.3',
    'OUTRA DEGENERACAO DE DISCO CERVICAL': 'M50.3',
    'OUTRA DEGENERACAO ESPECIFICADA DE DISCO INTERVERTEBRAL': 'M51.3',
    'OUTRA DEMENCIA VASCULAR': 'F01.8',
    'OUTRA DISFUNCAO NEUROMUSCULAR DA BEXIGA': 'N31.8',
    'OUTRA DISFUNCAO TESTICULAR': 'E29.8',
    'OUTRA DOR CRONICA': 'R52.2',
    'OUTRA DOR TORACICA': 'R07.3',
    'OUTRA DORSALGIA': 'M54.8',
    'OUTRA EMBOLIA E TROMBOSE VENOSAS': 'I82',
    'OUTRA ENDOMETRIOSE': 'N80.8',
    'OUTRA ENTESOPATIA DO PE': 'M77.5',
    'OUTRA FEBRE ESPECIFICADA': 'R50.8',
    'OUTRA FORMA DE DOENCA DE CROHN': 'K50.8',
    'OUTRA GOTA SECUNDARIA': 'M10.4',
    'OUTRA HIDROCELE': 'N43.2',
    'OUTRA HIPERTENSAO PULMONAR SECUNDARIA': 'I27.2',
    'OUTRA HIPOGLICEMIA': 'E16.1',
    'OUTRA INSUFICIENCIA RENAL CRONICA': 'N18.8',
    'OUTRA OBESIDADE': 'E66.8',
    'OUTRA OSTEOMALACIA DO ADULTO': 'M83.8',
    'OUTRA OSTEOMIELITE': 'M86.8',
    'OUTRA OSTEOMIELITE AGUDA': 'M86.1',
    'OUTRA OSTEOMIELITE CRONICA': 'M86.6',
    'OUTRA OSTEOMIELITE CRONICA HEMATOGENICA': 'M86.5',
    'OUTRA QUIMIOTERAPIA': 'Z51.2',
    'OUTRA REACAO A PUNCAO ESPINAL E LOMBAR': 'G97.1',
    'OUTRA TROMBOCITOPENIA PRIMARIA': 'D69.4',
    'OUTRAS (TENO)SINOVITES INFECCIOSAS': 'M65.1',
    'OUTRAS ACARIASES': 'B88.0',
    'OUTRAS AFECCOES ATROFICAS DA PELE': 'L90.8',
    'OUTRAS AFECCOES BOLHOSAS': 'L13',
    'OUTRAS AFECCOES BOLHOSAS ESPECIFICADAS': 'L13.8',
    'OUTRAS AFECCOES CARDIACAS EM DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'I52.0',
    'OUTRAS AFECCOES CARDIACAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'I52',
    'OUTRAS AFECCOES DA MAMA E DA LACTACAO ASSOCIADAS AO PARTO': 'O92',
    'OUTRAS AFECCOES DA MAMA, E AS NAO ESPECIFICADAS, ASSOCIADAS AO PARTO': 'O92.2',
    'OUTRAS AFECCOES DA PELE E DO TECIDO SUBCUTANEO NAO CLASSIFICADAS EM OUTRA PARTE': 'L98',
    'OUTRAS AFECCOES DA PELE E DO TECIDO SUBCUTANEO RELACIONADAS COM A RADIACAO': 'L59',
    'OUTRAS AFECCOES DA PROSTATA': 'N42',
    'OUTRAS AFECCOES DAS ARTERIAS E ARTERIOLAS': 'I77',
    'OUTRAS AFECCOES DAS UNHAS': 'L60.8',
    'OUTRAS AFECCOES ERITEMATOSAS': 'L53',
    'OUTRAS AFECCOES ERITEMATOSAS ESPECIFICADAS': 'L53.8',
    'OUTRAS AFECCOES ESPECIFICADAS ASSOCIADAS COM OS ORGAOS GENITAIS FEMININOS E COM O CICLO MENSTRUAL': 'N94.8',
    'OUTRAS AFECCOES ESPECIFICADAS DA PELE E DO TECIDO SUBCUTANEO': 'L98.8',
    'OUTRAS AFECCOES ESPECIFICADAS DA PELE E DO TECIDO SUBCUTANEO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'L99.8',
    'OUTRAS AFECCOES ESPECIFICADAS DA PROSTATA': 'N42.8',
    'OUTRAS AFECCOES ESPECIFICADAS DO TEGUMENTO PROPRIAS DO FETO E DO RECEM-NASCIDO': 'P83.8',
    'OUTRAS AFECCOES FOLICULARES': 'L73',
    'OUTRAS AFECCOES FOLICULARES ESPECIFICADAS': 'L73.8',
    'OUTRAS AFECCOES GRANULOMATOSAS DA PELE E DO TECIDO SUBCUTANEO': 'L92.8',
    'OUTRAS AFECCOES HEMORRAGICAS ESPECIFICADAS': 'D69.8',
    'OUTRAS AFECCOES HIPERTROFICAS DA PELE': 'L91.8',
    'OUTRAS AFECCOES INFILTRATIVAS DA PELE E DO TECIDO SUBCUTANEO': 'L98.6',
    'OUTRAS AFECCOES INFLAMATORIAS DA VAGINA E DA VULVA': 'N76',
    'OUTRAS AFECCOES LOCALIZADAS DO TECIDO CONJUNTIVO': 'L94',
    'OUTRAS AFECCOES ORIGINADAS NO PERIODO PERINATAL': 'P96',
    'OUTRAS AFECCOES PAPULO-DESCAMATIVAS': 'L44',
    'OUTRAS AFECCOES PLEURAIS': 'J94',
    'OUTRAS AFECCOES RESPIRATORIAS DEVIDA A PRODUTOS QUIMICOS, GASES, FUMACAS E VAPORES': 'J68.8',
    'OUTRAS AFECCOES RESPIRATORIAS ORIGINADAS NO PERIODO PERINATAL': 'P28',
    'OUTRAS AFECCOES SISTEMICAS DO TECIDO CONJUNTIVO': 'M35',
    'OUTRAS ALTERACOES AGUDAS DA PELE DEVIDAS A RADIACAO ULTRAVIOLETA': 'L56',
    'OUTRAS ALTERACOES AGUDAS ESPECIFICADAS DA PELE DEVIDAS A RADIACAO ULTRAVIOLETA': 'L56.8',
    'OUTRAS ALTERACOES CUTANEAS': 'R23',
    'OUTRAS ALTERACOES DA PELE E AS NAO ESPECIFICADAS': 'R23.8',
    'OUTRAS ALUCINACOES': 'R44.2',
    'OUTRAS ANCILOSTOMIASES': 'B76.8',
    'OUTRAS ANEMIAS': 'D64',
    'OUTRAS ANEMIAS APLASTICAS': 'D61',
    'OUTRAS ANEMIAS APLASTICAS ESPECIFICADAS': 'D61.8',
    'OUTRAS ANEMIAS ESPECIFICADAS': 'D64.8',
    'OUTRAS ANEMIAS HEMOLITICAS ADQUIRIDAS': 'D59.8',
    'OUTRAS ANEMIAS HEMOLITICAS AUTO-IMUNES': 'D59.1',
    'OUTRAS ANEMIAS HEMOLITICAS HEREDITARIAS': 'D58',
    'OUTRAS ANEMIAS HEMOLITICAS NAO-AUTOIMUNES': 'D59.4',
    'OUTRAS ANEMIAS MEGALOBLASTICAS NAO CLASSIFICADAS EM OUTRAS PARTES': 'D53.1',
    'OUTRAS ANEMIAS NUTRICIONAIS': 'D53',
    'OUTRAS ANEMIAS NUTRICIONAIS ESPECIFICADAS': 'D53.8',
    'OUTRAS ANEMIAS POR DEFICIENCIA DE FERRO': 'D50.8',
    'OUTRAS ANEMIAS POR DEFICIENCIA DE VITAMINA B12': 'D51.8',
    'OUTRAS ANOMALIAS DENTOFACIAIS': 'K07.8',
    'OUTRAS ANOMALIAS OBSTRUTIVAS DA PELVE RENAL E DO URETER': 'Q62.3',
    'OUTRAS ANORMALIDADES ADQUIRIDAS DOS OSSICULOS DO OUVIDO': 'H74.3',
    'OUTRAS ANORMALIDADES DA MARCHA E DA MOBILIDADE E AS NAO ESPECIFICADAS': 'R26.8',
    'OUTRAS ANORMALIDADES E AS NAO ESPECIFICADAS DA RESPIRACAO': 'R06.8',
    'OUTRAS ANORMALIDADES E AS NAO ESPECIFICADAS DO BATIMENTO CARDIACO': 'R00.8',
    'OUTRAS ANORMALIDADES FECAIS': 'R19.5',
    'OUTRAS ARRITMIAS CARDIACAS': 'I49',
    'OUTRAS ARRITMIAS CARDIACAS ESPECIFICADAS': 'I49.8',
    'OUTRAS ARTRITES': 'M13',
    'OUTRAS ARTRITES E POLIARTRITES ESTREPTOCOCICAS': 'M00.2',
    'OUTRAS ARTRITES ESPECIFICADAS': 'M13.8',
    'OUTRAS ARTRITES JUVENIS': 'M08.8',
    'OUTRAS ARTRITES REUMATOIDES': 'M06',
    'OUTRAS ARTRITES REUMATOIDES ESPECIFICADAS': 'M06.8',
    'OUTRAS ARTRITES REUMATOIDES SORO-POSITIVAS': 'M05.8',
    'OUTRAS ARTROPATIAS ESPECIFICAS NAO CLASSIFICADAS EM OUTRA PARTE': 'M12.8',
    'OUTRAS ARTROPATIAS POS-INFECCIOSAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M03.2',
    'OUTRAS ARTROPATIAS PSORIASICAS': 'M07.3',
    'OUTRAS ARTROPATIAS REACIONAIS': 'M02.8',
    'OUTRAS ARTROSES': 'M19',
    'OUTRAS ARTROSES ESPECIFICADAS': 'M19.8',
    'OUTRAS ARTROSES PRIMARIAS DA PRIMEIRA ARTICULACAO CARPOMETACARPIANA': 'M18.1',
    'OUTRAS ARTROSES SECUNDARIAS': 'M19.2',
    'OUTRAS ATAXIAS HEREDITARIAS': 'G11.8',
    'OUTRAS BURSITES DO COTOVELO': 'M70.3',
    'OUTRAS BURSITES DO JOELHO': 'M70.5',
    'OUTRAS BURSITES DO QUADRIL': 'M70.7',
    'OUTRAS BURSITES NAO CLASSIFICADAS EM OUTRA PARTE': 'M71.5',
    'OUTRAS BURSOPATIAS': 'M71',
    'OUTRAS BURSOPATIAS ESPECIFICADAS': 'M71.8',
    'OUTRAS CARDIOMIOPATIAS': 'I42.8',
    'OUTRAS CARIES DENTARIAS': 'K02.8',
    'OUTRAS CATARATAS': 'H26',
    'OUTRAS CATARATAS SENIS': 'H25.8',
    'OUTRAS CAUSAS MAL DEFINIDAS E AS NAO ESPECIFICADAS DE MORTALIDADE': 'R99',
    'OUTRAS CERATITES': 'H16.8',
    'OUTRAS CERATITES SUPERFICIAIS SEM CONJUNTIVITE': 'H16.1',
    'OUTRAS CICATRIZES E OPACIDADES DA CORNEA': 'H17.8',
    'OUTRAS CISTITES': 'N30.8',
    'OUTRAS CISTITES CRONICAS': 'N30.2',
    'OUTRAS COLECISTITES': 'K81.8',
    'OUTRAS COLELITIASES': 'K80.8',
    'OUTRAS COLITES ULCERATIVAS': 'K51.8',
    'OUTRAS COMPLICACOES ATUAIS SUBSEQUENTES AO INFARTO AGUDO DO MIOCARDIO': 'I23.8',
    'OUTRAS COMPLICACOES DE ANESTESIA': 'T88.5',
    'OUTRAS COMPLICACOES DE CUIDADOS MEDICOS E CIRURGICOS ESPECIFICADOS NAO CLASSIFICADOS EM OUTRA PARTE': 'T88.8',
    'OUTRAS COMPLICACOES DE CUIDADOS MEDICOS E CIRURGICOS NAO CLASSIFICADAS EM OUTRA PARTE': 'T88',
    'OUTRAS COMPLICACOES DE PROCEDIMENTOS NAO CLASSIFICADAS EM OUTRA PARTE': 'T81.8',
    'OUTRAS COMPLICACOES DO PUERPERIO, NAO CLASSIFICADAS EM OUTRA PARTE': 'O90.8',
    'OUTRAS COMPLICACOES E AS NAO ESPECIFICADAS DO COTO DE AMPUTACAO': 'T87.6',
    'OUTRAS COMPLICACOES ESPECIFICADAS DO TRABALHO DE PARTO E DO PARTO': 'O75.8',
    'OUTRAS COMPLICACOES ESPECIFICAS DE GESTACAO MULTIPLA': 'O31.8',
    'OUTRAS COMPLICACOES SUBSEQUENTES A IMUNIZACAO NAO CLASSIFICADAS EM OUTRA PARTE': 'T88.1',
    'OUTRAS CONJUNTIVITES': 'H10.8',
    'OUTRAS CONJUNTIVITES AGUDAS': 'H10.2',
    'OUTRAS CONJUNTIVITES VIRAIS': 'B30.8',
    'OUTRAS CONVULSOES E AS NAO ESPECIFICADAS': 'R56.8',
    'OUTRAS COXARTROSES DISPLASICAS': 'M16.3',
    'OUTRAS COXARTROSES POS-TRAUMATICAS': 'M16.5',
    'OUTRAS COXARTROSES PRIMARIAS': 'M16.1',
    'OUTRAS COXARTROSES SECUNDARIAS': 'M16.7',
    'OUTRAS DEFICIENCIAS NUTRICIONAIS': 'E63',
    'OUTRAS DEFORMIDADES (ADQUIRIDAS) DO(S) DEDO(S) DOS PES': 'M20.5',
    'OUTRAS DEFORMIDADES ADQUIRIDAS DO SISTEMA OSTEOMUSCULAR E DO TECIDO CONJUNTIVO': 'M95',
    'OUTRAS DEFORMIDADES ADQUIRIDAS DO TORNOZELO E DO PE': 'M21.6',
    'OUTRAS DEFORMIDADES ADQUIRIDAS DOS MEMBROS': 'M21',
    'OUTRAS DEFORMIDADES CONGENITAS DOS PES EM VALGO': 'Q66.6',
    'OUTRAS DEFORMIDADES DA ORELHA': 'Q17.3',
    'OUTRAS DEFORMIDADES OSTEOMUSCULARES CONGENITAS': 'Q68.8',
    'OUTRAS DEFORMIDADES POR REDUCAO DO ENCEFALO': 'Q04.3',
    'OUTRAS DERMATITES': 'L30',
    'OUTRAS DERMATITES ATOPICAS': 'L20.8',
    'OUTRAS DERMATITES ESPECIFICADAS': 'L30.8',
    'OUTRAS DERMATITES SEBORREICAS': 'L21.8',
    'OUTRAS DERMATOFITOSES': 'B35.8',
    'OUTRAS DERMATOMIOSITES': 'M33.1',
    'OUTRAS DIFICULDADES A MICCAO': 'R39.1',
    'OUTRAS DIFICULDADES FISICAS E MENTAIS RELACIONADAS AO TRABALHO': 'Z56.6',
    'OUTRAS DISPLASIAS MAMARIAS BENIGNAS': 'N60.8',
    'OUTRAS DISTONIAS': 'G24.8',
    'OUTRAS DOENCAS BACTERIANAS ESPECIFICADAS': 'A48.8',
    'OUTRAS DOENCAS BACTERIANAS ZOONOTICAS ESPECIFICADAS NAO CLASSIFICADAS EM OUTRA PARTE': 'A28.8',
    'OUTRAS DOENCAS CEREBROVASCULARES ESPECIFICADAS': 'I67.8',
    'OUTRAS DOENCAS CRONICAS DAS AMIGDALAS E DAS ADENOIDES': 'J35.8',
    'OUTRAS DOENCAS DA FARINGE': 'J39.2',
    'OUTRAS DOENCAS DA GLANDULA DE BARTHOLIN': 'N75.8',
    'OUTRAS DOENCAS DA LARINGE': 'J38.7',
    'OUTRAS DOENCAS DA LINGUA': 'K14.8',
    'OUTRAS DOENCAS DA POLPA E DOS TECIDOS PERIAPICAIS E AS NAO ESPECIFICADAS': 'K04.9',
    'OUTRAS DOENCAS DA VALVA MITRAL': 'I05.8',
    'OUTRAS DOENCAS DAS CORDAS VOCAIS': 'J38.3',
    'OUTRAS DOENCAS DAS GLANDULAS SALIVARES': 'K11.8',
    'OUTRAS DOENCAS DESMIELINIZANTES DO SISTEMA NERVOSO CENTRAL': 'G37',
    'OUTRAS DOENCAS DOS BRONQUIOS NAO CLASSIFICADAS EM OUTRA PARTE': 'J98.0',
    'OUTRAS DOENCAS E AFECCOES ESPECIFICADAS COMPLICANDO A GRAVIDEZ, O PARTO E O PUERPERIO': 'O99.8',
    'OUTRAS DOENCAS ESPECIFICADAS DA MEDULA ESPINAL': 'G95.8',
    'OUTRAS DOENCAS ESPECIFICADAS DA VESICULA BILIAR': 'K82.8',
    'OUTRAS DOENCAS ESPECIFICADAS DAS VIAS AEREAS SUPERIORES': 'J39.8',
    'OUTRAS DOENCAS ESPECIFICADAS DAS VIAS BILIARES': 'K83.8',
    'OUTRAS DOENCAS ESPECIFICADAS DE TRANSMISSAO PREDOMINANTEMENTE SEXUAL': 'A63.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO ANUS E DO RETO': 'K62.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO APARELHO DIGESTIVO': 'K92.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO APENDICE': 'K38.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO ESOFAGO': 'K22.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO ESTOMAGO E DO DUODENO': 'K31.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO FIGADO': 'K76.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO INTESTINO': 'K63.8',
    'OUTRAS DOENCAS ESPECIFICADAS DO SANGUE E DOS ORGAOS HEMATOPOETICOS': 'D75.8',
    'OUTRAS DOENCAS ESPECIFICADAS DOS MAXILARES': 'K10.8',
    'OUTRAS DOENCAS ESPECIFICADAS DOS TECIDOS DUROS DOS DENTES': 'K03.8',
    'OUTRAS DOENCAS ESPECIFICADAS DOS VASOS PULMONARES': 'I28.8',
    'OUTRAS DOENCAS ESPECIFICADAS POR VIRUS': 'B33.8',
    'OUTRAS DOENCAS EXTRAPIRAMIDAIS E TRANSTORNOS DOS MOVIMENTOS': 'G25',
    'OUTRAS DOENCAS EXTRAPIRAMIDAIS E TRANSTORNOS DOS MOVIMENTOS, ESPECIFICADOS': 'G25.8',
    'OUTRAS DOENCAS INFLAMATORIAS DA PROSTATA': 'N41.8',
    'OUTRAS DOENCAS PERIODONTAIS': 'K05.5',
    'OUTRAS DOENCAS POR VIRUS NAO CLASSIFICADA EM OUTRA PARTE': 'B33',
    'OUTRAS DOENCAS PULMONARES DO CORACAO ESPECIFICADAS': 'I27.8',
    'OUTRAS DOENCAS PULMONARES INTERSTICIAIS COM FIBROSE': 'J84.1',
    'OUTRAS DOENCAS PULMONARES INTERSTICIAIS ESPECIFICADAS': 'J84.8',
    'OUTRAS DOENCAS REUMATICAS DA VALVA AORTICA': 'I06.8',
    'OUTRAS DOENCAS VASCULARES PERIFERICAS ESPECIFICADAS': 'I73.8',
    'OUTRAS DORES ABDOMINAIS E AS NAO ESPECIFICADAS': 'R10.4',
    'OUTRAS DORSOPATIAS DEFORMANTES': 'M43',
    'OUTRAS DORSOPATIAS ESPECIFICADAS': 'M53.8',
    'OUTRAS DORSOPATIAS NAO CLASSIFICADAS EM OUTRA PARTE': 'M53',
    'OUTRAS ENCEFALITES VIRAIS NAO CLASSIFICADAS EM OUTRA PARTE': 'A85',
    'OUTRAS ENCEFALITES, MIELITES E ENCEFALOMIELITES': 'G04.8',
    'OUTRAS ENDOFTALMITES': 'H44.1',
    'OUTRAS ENTERITES VIRAIS': 'A08.3',
    'OUTRAS ENTESOPATIAS': 'M77',
    'OUTRAS ENTESOPATIAS DO MEMBRO INFERIOR, EXCLUINDO O PE': 'M76.8',
    'OUTRAS ENTESOPATIAS NAO CLASSIFICADAS EM OUTRA PARTE': 'M77.8',
    'OUTRAS EPILEPSIAS': 'G40.8',
    'OUTRAS EPILEPSIAS E SINDROMES EPILEPTICAS GENERALIZADAS': 'G40.4',
    'OUTRAS ESCOLIOSES IDIOPATICAS': 'M41.2',
    'OUTRAS ESCOLIOSES SECUNDARIAS': 'M41.5',
    'OUTRAS ESPONDILOPATIAS INFLAMATORIAS ESPECIFICADAS': 'M46.8',
    'OUTRAS ESPONDILOSES': 'M47.8',
    'OUTRAS ESPONDILOSES COM MIELOPATIA': 'M47.1',
    'OUTRAS ESPONDILOSES COM RADICULOPATIAS': 'M47.2',
    'OUTRAS ESQUIZOFRENIAS': 'F20.8',
    'OUTRAS FEBRES HEMORRAGICAS ESPECIFICADAS POR VIRUS': 'A98.8',
    'OUTRAS FEBRES HEMORRAGICAS POR ARENAVIRUS': 'A96.8',
    'OUTRAS FEBRES HEMORRAGICAS POR VIRUS NAO CLASSIFICADAS EM OUTRA PARTE': 'A98',
    'OUTRAS FEBRES MACULOSAS': 'A77.8',
    'OUTRAS FEBRES POR VIRUS TRANSMITIDAS POR ARTROPODES NAO CLASSIFICADAS EM OUTRA PARTE': 'A93',
    'OUTRAS FEBRES VIRAIS ESPECIFICADAS TRANSMITIDAS POR ARTROPODES': 'A93.8',
    'OUTRAS FEBRES VIRAIS ESPECIFICADAS TRANSMITIDAS POR MOSQUITOS': 'A92.8',
    'OUTRAS FEBRES VIRAIS TRANSMITIDAS POR MOSQUITOS': 'A92',
    'OUTRAS FISTULAS DO TRATO GENITURINARIO FEMININO': 'N82.1',
    'OUTRAS FORMAS DE ACNE': 'L70.8',
    'OUTRAS FORMAS DE ACTINOMICOSE': 'A42.8',
    'OUTRAS FORMAS DE ALOPECIA AREATA': 'L63.8',
    'OUTRAS FORMAS DE ALOPECIA CICATRICIAL': 'L66.8',
    'OUTRAS FORMAS DE ANGINA PECTORIS': 'I20.8',
    'OUTRAS FORMAS DE APENDICITE': 'K36',
    'OUTRAS FORMAS DE BARTONELOSE': 'A44.8',
    'OUTRAS FORMAS DE BLOQUEIO ATRIOVENTRICULAR E AS NAO ESPECIFICADAS': 'I44.3',
    'OUTRAS FORMAS DE BLOQUEIO DE RAMO DIREITO E AS NAO ESPECIFICADAS': 'I45.1',
    'OUTRAS FORMAS DE CHOQUE': 'R57.8',
    'OUTRAS FORMAS DE CIRROSE HEPATICA E AS NAO ESPECIFICADAS': 'K74.6',
    'OUTRAS FORMAS DE CISTOS FOLICULARES DA PELE E DO TECIDO SUBCUTANEO': 'L72.8',
    'OUTRAS FORMAS DE DOENCA ISQUEMICA AGUDA DO CORACAO': 'I24.8',
    'OUTRAS FORMAS DE ENFISEMA': 'J43.8',
    'OUTRAS FORMAS DE ENXAQUECA': 'G43.8',
    'OUTRAS FORMAS DE ERISIPELOIDE': 'A26.8',
    'OUTRAS FORMAS DE ERITEMA MULTIFORME': 'L51.8',
    'OUTRAS FORMAS DE ESTOMATITE': 'K12.1',
    'OUTRAS FORMAS DE GRAVIDEZ ECTOPICA': 'O00.8',
    'OUTRAS FORMAS DE HIDROCEFALIA': 'G91.8',
    'OUTRAS FORMAS DE HIPERPIGMENTACAO PELA MELANINA': 'L81.4',
    'OUTRAS FORMAS DE HIPERTENSAO SECUNDARIA': 'I15.8',
    'OUTRAS FORMAS DE INFECCAO DEVIDA AO VIRUS DO HERPES': 'B00.8',
    'OUTRAS FORMAS DE LEPTOSPIROSE': 'A27.8',
    'OUTRAS FORMAS DE LUPUS ERITEMATOSO DISSEMINADO [SISTEMICO]': 'M32.8',
    'OUTRAS FORMAS DE LUPUS ERITEMATOSO LOCALIZADO': 'L93.2',
    'OUTRAS FORMAS DE MA-ABSORCAO INTESTINAL': 'K90.8',
    'OUTRAS FORMAS DE OBSTRUCAO INTESTINAL, E AS NAO ESPECIFICADAS': 'K56.6',
    'OUTRAS FORMAS DE PARACOCCIDIOIDOMICOSE': 'B41.8',
    'OUTRAS FORMAS DE PARALISIA CEREBRAL': 'G80.8',
    'OUTRAS FORMAS DE PARKINSONISMO SECUNDARIO': 'G21.8',
    'OUTRAS FORMAS DE PENFIGOIDE': 'L12.8',
    'OUTRAS FORMAS DE PNEUMOTORAX ESPONTANEO': 'J93.1',
    'OUTRAS FORMAS DE PRURIDO': 'L29.8',
    'OUTRAS FORMAS DE PRURIGO': 'L28.2',
    'OUTRAS FORMAS DE PSORIASE': 'L40.8',
    'OUTRAS FORMAS DE SIFILIS SECUNDARIA': 'A51.4',
    'OUTRAS FORMAS DE SIFILIS TARDIA SINTOMATICA': 'A52.7',
    'OUTRAS FORMAS DE VOMITOS COMPLICANDO A GRAVIDEZ': 'O21.8',
    'OUTRAS FORMAS E AS NAO ESPECIFICADAS DA SIFILIS': 'A53',
    'OUTRAS FORMAS ESPECIFICADAS DE BLOQUEIO CARDIACO': 'I45.5',
    'OUTRAS FORMAS ESPECIFICADAS DE DOENCA PULMONAR OBSTRUTIVA CRONICA': 'J44.8',
    'OUTRAS FORMAS ESPECIFICADAS DE TREMOR': 'G25.2',
    'OUTRAS FORMAS NAO CICATRICIAIS DA PERDA DE CABELOS OU PELOS': 'L65',
    'OUTRAS FORMAS, ESPECIFICADAS, NAO CICATRICIAIS, DE PERDA DE CABELOS OU PELOS': 'L65.8',
    'OUTRAS FRATURAS DO CRANIO E DOS OSSOS DA FACE': 'S02.8',
    'OUTRAS GASTRITES': 'K29.6',
    'OUTRAS GASTRITES AGUDAS': 'K29.1',
    'OUTRAS GASTROENTERITES E COLITES ESPECIFICADAS, NAO-INFECCIOSAS': 'K52.8',
    'OUTRAS GASTROENTERITES E COLITES NAO-INFECCIOSAS': 'K52',
    'OUTRAS GONARTROSES POS-TRAUMATICA': 'M17.3',
    'OUTRAS GONARTROSES PRIMARIAS': 'M17.1',
    'OUTRAS GONARTROSES SECUNDARIAS': 'M17.5',
    'OUTRAS GONARTROSES SECUNDARIAS BILATERAIS': 'M17.4',
    'OUTRAS HELMINTIASES': 'B83',
    'OUTRAS HELMINTIASES ESPECIFICADAS': 'B83.8',
    'OUTRAS HEMORRAGIAS DO INICIO DA GRAVIDEZ': 'O20.8',
    'OUTRAS HEMORRAGIAS INTRACRANIANAS NAO-TRAUMATICAS': 'I62',
    'OUTRAS HEMORRAGIAS SUBARACNOIDES': 'I60.8',
    'OUTRAS HEPATITES CRONICAS NAO CLASSIFICADA EM OUTRA PARTE': 'K73.8',
    'OUTRAS HEPATITES CRONICAS VIRAIS': 'B18.8',
    'OUTRAS HEPATITES VIRAIS AGUDAS': 'B17',
    'OUTRAS HEPATITES VIRAIS AGUDAS ESPECIFICADAS': 'B17.8',
    'OUTRAS HERNIAS ABDOMINAIS': 'K45',
    'OUTRAS HERNIAS ABDOMINAIS ESPECIFICADAS, COM OBSTRUCAO, SEM GANGRENA': 'K45.0',
    'OUTRAS HERNIAS ABDOMINAIS ESPECIFICADAS, SEM OBSTRUCAO OU GANGRENA': 'K45.8',
    'OUTRAS HIDRONEFROSES E AS NAO ESPECIFICADAS': 'N13.3',
    'OUTRAS HIPOCALCEMIAS NEONATAIS': 'P71.1',
    'OUTRAS IMUNODEFICIENCIAS': 'D84',
    'OUTRAS IMUNODEFICIENCIAS ESPECIFICADAS': 'D84.8',
    'OUTRAS INCONTINENCIAS URINARIAS ESPECIFICADAS': 'N39.4',
    'OUTRAS INFECCOES AGUDAS DAS VIAS AEREAS SUPERIORES DE LOCALIZACOES MULTIPLAS': 'J06.8',
    'OUTRAS INFECCOES BACTERIANAS DE LOCALIZACAO NAO ESPECIFICADA': 'A49.8',
    'OUTRAS INFECCOES BACTERIANAS INTESTINAIS ESPECIFICADAS': 'A04.8',
    'OUTRAS INFECCOES CAUSADAS POR CLAMIDIAS TRANSMITIDAS POR VIA SEXUAL': 'A56',
    'OUTRAS INFECCOES E AS NAO ESPECIFICADAS DO TRATO URINARIO NA GRAVIDEZ': 'O23.9',
    'OUTRAS INFECCOES ESPECIFICADAS POR SALMONELA': 'A02.8',
    'OUTRAS INFECCOES ESPECIFICAS DO PERIODO PERINATAL': 'P39',
    'OUTRAS INFECCOES GONOCOCICAS': 'A54.8',
    'OUTRAS INFECCOES INTESTINAIS BACTERIANAS': 'A04',
    'OUTRAS INFECCOES INTESTINAIS ESPECIFICADAS': 'A08.5',
    'OUTRAS INFECCOES INTESTINAIS POR ESCHERICHIA COLI': 'A04.4',
    'OUTRAS INFECCOES LOCALIZADAS DA PELE E DO TECIDO SUBCUTANEO': 'L08',
    'OUTRAS INFECCOES LOCALIZADAS, ESPECIFICADAS, DA PELE E DO TECIDO SUBCUTANEO': 'L08.8',
    'OUTRAS INFECCOES MICOBACTERIANAS': 'A31.8',
    'OUTRAS INFECCOES POR ORTOPOXVIRUS': 'B08.0',
    'OUTRAS INFECCOES POR SALMONELLA': 'A02',
    'OUTRAS INFECCOES POR VIRUS DE LOCALIZACAO NAO ESPECIFICADA': 'B34.8',
    'OUTRAS INFECCOES VIRAIS DO SISTEMA NERVOSO CENTRAL NAO CLASSIFICADAS EM OUTRA PARTE': 'A88',
    'OUTRAS INFECCOES VIRAIS ESPECIFICADAS CARACTERIZADAS POR LESOES DE PELE E DAS MEMBRANAS MUCOSAS': 'B08.8',
    'OUTRAS INFESTACOES': 'B88',
    'OUTRAS INFESTACOES POR ARTROPODOS': 'B88.2',
    'OUTRAS INFESTACOES POR CESTOIDES': 'B71',
    'OUTRAS INFLAMACOES CORIORRETINIANAS': 'H30.8',
    'OUTRAS INFLAMACOES ESPECIFICADAS DA PALPEBRA': 'H01.8',
    'OUTRAS INFLAMACOES ESPECIFICADAS DA VAGINA E DA VULVA': 'N76.8',
    'OUTRAS INSTABILIDADES ARTICULARES': 'M25.3',
    'OUTRAS INTOLERANCIAS A LACTOSE': 'E73.8',
    'OUTRAS INTOXICACOES ALIMENTARES BACTERIANAS ESPECIFICADAS': 'A05.8',
    'OUTRAS INTOXICACOES ALIMENTARES BACTERIANAS, NAO CLASSIFICADAS EM OUTRA PARTE': 'A05',
    'OUTRAS INTOXICACOES POR PEIXES E MARISCOS': 'T61.2',
    'OUTRAS IRIDOCICLITES': 'H20.8',
    'OUTRAS LESOES CUTANEAS PRECOCES DA BOUBA': 'A66.2',
    'OUTRAS LESOES DO NERVO MEDIANO': 'G56.1',
    'OUTRAS LESOES DO OMBRO': 'M75.8',
    'OUTRAS LESOES E AS NAO ESPECIFICADAS DA MUCOSA ORAL': 'K13.7',
    'OUTRAS LEUCEMIAS DE TIPO CELULAR NAO ESPECIFICADO': 'C95.7',
    'OUTRAS LEUCEMIAS ESPECIFICADAS': 'C94.7',
    'OUTRAS LEUCEMIAS LINFOIDES': 'C91.7',
    'OUTRAS LEUCEMIAS MIELOIDES': 'C92.7',
    'OUTRAS LEUCEMIAS MONOCITICAS': 'C93.7',
    'OUTRAS LINFADENITES INESPECIFICAS': 'I88.8',
    'OUTRAS LORDOSES': 'M40.4',
    'OUTRAS MALFORMACOES CONGENITAS DA PELE': 'Q82',
    'OUTRAS MALFORMACOES CONGENITAS DO OLHO': 'Q15',
    'OUTRAS MALFORMACOES CONGENITAS DOS OSSOS DO CRANIO E DA FACE': 'Q75',
    'OUTRAS MALFORMACOES CONGENITAS ESPECIFICADAS DA PELE': 'Q82.8',
    'OUTRAS MALFORMACOES CONGENITAS ESPECIFICADAS DO ESTOMAGO': 'Q40.2',
    'OUTRAS MALFORMACOES CONGENITAS ESPECIFICADAS DO INTESTINO': 'Q43.8',
    'OUTRAS MALFORMACOES CONGENITAS ESPECIFICADAS DOS OSSOS DO CRANIO E DA FACE': 'Q75.8',
    'OUTRAS MANIFESTACOES DA DEFICIENCIA DE TIAMINA': 'E51.8',
    'OUTRAS MANIFESTACOES OCULARES DEVIDAS A DEFICIENCIA DE VITAMINA A': 'E50.7',
    'OUTRAS MASTOIDITES E AFECCOES RELACIONADAS COM A MASTOIDITE': 'H70.8',
    'OUTRAS MEDIDAS PROFILATICAS ESPECIFICADAS': 'Z29.8',
    'OUTRAS MENINGITES BACTERIANAS': 'G00.8',
    'OUTRAS MICOSES ESPECIFICADAS': 'B48.8',
    'OUTRAS MICOSES NAO CLASSIFICADAS EM OUTRA PARTE': 'B48',
    'OUTRAS MICOSES SUPERFICIAIS': 'B36',
    'OUTRAS MICOSES SUPERFICIAIS ESPECIFICADAS': 'B36.8',
    'OUTRAS MIOPATIAS': 'G72',
    'OUTRAS MIOSITES': 'M60.8',
    'OUTRAS MONONEUROPATIAS': 'G58',
    'OUTRAS MONONEUROPATIAS DOS MEMBROS INFERIORES': 'G57.8',
    'OUTRAS MONONEUROPATIAS DOS MEMBROS SUPERIORES': 'G56.8',
    'OUTRAS MONONEUROPATIAS ESPECIFICADAS': 'G58.8',
    'OUTRAS MONONUCLEOSES INFECCIOSAS': 'B27.8',
    'OUTRAS MORTES SUBITAS DE CAUSA DESCONHECIDA': 'R96',
    'OUTRAS NEFRITES TUBULO-INTERSTICIAIS CRONICAS': 'N11.8',
    'OUTRAS NEOPLASIAS MALIGNAS DA PELE': 'C44',
    'OUTRAS NEUROPATIAS HEREDITARIAS E IDIOPATICAS': 'G60.8',
    'OUTRAS OBSTRUCOES DO INTESTINO': 'K56.4',
    'OUTRAS OBSTRUCOES INTESTINAIS DO RECEM-NASCIDO': 'P76',
    'OUTRAS OPACIDADES DO VITREO': 'H43.3',
    'OUTRAS OSTEOARTROPATIAS HIPERTROFICAS': 'M89.4',
    'OUTRAS OSTEOCONDRODISPLASIAS': 'Q78',
    'OUTRAS OSTEOCONDROPATIAS': 'M93',
    'OUTRAS OSTEOCONDROSES JUVENIS': 'M92',
    'OUTRAS OSTEONECROSES': 'M87.8',
    'OUTRAS OSTEOPOROSES': 'M81.8',
    'OUTRAS OTITES EXTERNAS': 'H60.8',
    'OUTRAS OTITES EXTERNAS INFECCIOSAS': 'H60.3',
    'OUTRAS OTITES MEDIAS AGUDAS NAO-SUPURATIVAS': 'H65.1',
    'OUTRAS OTITES MEDIAS CRONICAS NAO-SUPURATIVAS': 'H65.4',
    'OUTRAS OTITES MEDIAS SUPURATIVAS CRONICAS': 'H66.3',
    'OUTRAS PANCREATITES AGUDAS': 'K85.8',
    'OUTRAS PANCREATITES CRONICAS': 'K86.1',
    'OUTRAS PERCEPCOES AUDITIVAS ANORMAIS': 'H93.2',
    'OUTRAS PERDAS DE AUDICAO': 'H91',
    'OUTRAS PERDAS DE AUDICAO ESPECIFICADAS': 'H91.8',
    'OUTRAS PERFURACOES DA MEMBRANA DO TIMPANO': 'H72.8',
    'OUTRAS PNEUMONIAS BACTERIANAS': 'J15.8',
    'OUTRAS PNEUMONIAS DEVIDAS A MICROORGANISMOS NAO ESPECIFICADOS': 'J18.8',
    'OUTRAS PNEUMONIAS VIRAIS': 'J12.8',
    'OUTRAS POLIARTROSES': 'M15.8',
    'OUTRAS POLINEUROPATIAS': 'G62',
    'OUTRAS POLINEUROPATIAS ESPECIFICADAS': 'G62.8',
    'OUTRAS POLINEUROPATIAS INFLAMATORIAS': 'G61.8',
    'OUTRAS PURPURAS NAO-TROMBOCITOPENICAS': 'D69.2',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO': 'W17',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W17.3',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - AREAS DE COMERCIO E DE SERVICOS': 'W17.5',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W17.6',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - HABITACAO COLETIVA': 'W17.1',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - LOCAL NAO ESPECIFICADO': 'W17.9',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - OUTROS LOCAIS ESPECIFICADOS': 'W17.8',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - RESIDENCIA': 'W17.0',
    'OUTRAS QUEDAS DE UM NIVEL A OUTRO - RUA E ESTRADA': 'W17.4',
    'OUTRAS QUEDAS NO MESMO NIVEL': 'W18',
    'OUTRAS QUEDAS NO MESMO NIVEL - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W18.3',
    'OUTRAS QUEDAS NO MESMO NIVEL - AREAS DE COMERCIO E DE SERVICOS': 'W18.5',
    'OUTRAS QUEDAS NO MESMO NIVEL - HABITACAO COLETIVA': 'W18.1',
    'OUTRAS QUEDAS NO MESMO NIVEL - LOCAL NAO ESPECIFICADO': 'W18.9',
    'OUTRAS QUEDAS NO MESMO NIVEL - OUTROS LOCAIS ESPECIFICADOS': 'W18.8',
    'OUTRAS QUEDAS NO MESMO NIVEL - RESIDENCIA': 'W18.0',
    'OUTRAS QUEDAS NO MESMO NIVEL - RUA E ESTRADA': 'W18.4',
    'OUTRAS QUEDAS NO MESMO NIVEL POR COLISAO COM OU EMPURRAO POR OUTRA PESSOA': 'W03',
    'OUTRAS QUEIMADURAS SOLARES': 'L55.8',
    'OUTRAS REACOES DE INTOLERANCIA ALIMENTAR NAO CLASSIFICADAS EM OUTRA PARTE': 'T78.1',
    'OUTRAS RINITES ALERGICAS': 'J30.3',
    'OUTRAS RINITES ALERGICAS SAZONAIS': 'J30.2',
    'OUTRAS RUPTURAS ESPONTANEAS DE LIGAMENTO(S) DO JOELHO': 'M23.6',
    'OUTRAS RUPTURAS MUSCULARES (NAO-TRAUMATICAS)': 'M62.1',
    'OUTRAS SEPTICEMIAS': 'A41',
    'OUTRAS SEPTICEMIAS ESPECIFICADAS': 'A41.8',
    'OUTRAS SHIGUELOSES': 'A03.8',
    'OUTRAS SINDROMES COM MALFORMACOES CONGENITAS QUE ACOMETEM MULTIPLOS SISTEMAS': 'Q87',
    'OUTRAS SINDROMES DE ALGIAS CEFALICAS': 'G44',
    'OUTRAS SINDROMES DE CEFALEIA ESPECIFICADAS': 'G44.8',
    'OUTRAS SINDROMES DE MAUS TRATOS': 'Y07',
    'OUTRAS SINDROMES DE MAUS TRATOS PELO ESPOSO OU COMPANHEIRO': 'Y07.0',
    'OUTRAS SINDROMES ESPECIFICADAS DE MAUS TRATOS': 'T74.8',
    'OUTRAS SINDROMES MIELODISPLASICAS': 'D46.7',
    'OUTRAS SINDROMES PARALITICAS': 'G83',
    'OUTRAS SINDROMES VASCULARES CEREBRAIS EM DOENCAS CEREBROVASCULARES': 'G46.8',
    'OUTRAS SINOVITES E TENOSSINOVITES': 'M65.8',
    'OUTRAS SINUSITES AGUDAS': 'J01.8',
    'OUTRAS SINUSITES CRONICAS': 'J32.8',
    'OUTRAS TIREOTOXICOSES': 'E05.8',
    'OUTRAS URETRITES': 'N34.2',
    'OUTRAS UROPATIAS OBSTRUTIVAS E POR REFLUXO': 'N13.8',
    'OUTRAS URTICARIAS': 'L50.8',
    'OUTRAS VASCULOPATIAS NECROTIZANTES': 'M31',
    'OUTRAS VASCULOPATIAS NECROTIZANTES ESPECIFICADAS': 'M31.8',
    'OUTRAS VERTIGENS PERIFERICAS': 'H81.3',
    'OUTRO CISTO OSSEO': 'M85.6',
    'OUTRO GLAUCOMA': 'H40.8',
    'OUTRO PERIODO DE ESPERA PARA INVESTIGACAO E TRATAMENTO': 'Z75.2',
    'OUTRO PROLAPSO GENITAL FEMININO': 'N81.8',
    'OUTRO SEGUIMENTO CIRURGICO': 'Z48',
    'OUTRO SEGUIMENTO CIRURGICO ESPECIFICADO': 'Z48.8',
    'OUTRO TIPO DE INSUFICIENCIA RENAL AGUDA': 'N17.8',
    'OUTRO TRAUMATISMO DA MEDULA LOMBAR': 'S34.1',
    'OUTROS ABSCESSOS DA FARINGE': 'J39.1',
    'OUTROS ACHADOS ANORMAIS DE EXAMES QUIMICOS DO SANGUE': 'R79',
    'OUTROS ACHADOS ANORMAIS ESPECIFICADOS DE EXAMES QUIMICOS DO SANGUE': 'R79.8',
    'OUTROS ACHADOS ANORMAIS NA URINA': 'R82',
    'OUTROS ACHADOS ANORMAIS NA URINA E OS NAO ESPECIFICADOS': 'R82.9',
    'OUTROS ACIDENTES DE TRANSPORTE ESPECIFICADOS': 'V98',
    'OUTROS ACIDENTES ISQUEMICOS CEREBRAIS TRANSITORIOS E SINDROMES CORRELATAS': 'G45.8',
    'OUTROS ACONSELHAMENTOS ESPECIFICADOS': 'Z71.8',
    'OUTROS AGENTES BACTERIANOS COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B96',
    'OUTROS AGENTES VIRAIS, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97.8',
    'OUTROS ANEURISMAS': 'I72',
    'OUTROS BOCIOS NAO-TOXICOS': 'E04',
    'OUTROS CALCULOS DO TRATO URINARIO INFERIOR': 'N21.8',
    'OUTROS CISTOS DAS MANDIBULAS': 'K09.2',
    'OUTROS CISTOS DE BOLSA SINOVIAL': 'M71.3',
    'OUTROS CISTOS OVARIANOS E OS NAO ESPECIFICADOS': 'N83.2',
    'OUTROS CUIDADOS DE SEGUIMENTO ORTOPEDICO': 'Z47',
    'OUTROS CUIDADOS MEDICOS': 'Z51',
    'OUTROS DEFEITOS DA COAGULACAO': 'D68',
    'OUTROS DEFEITOS ESPECIFICADOS DA COAGULACAO': 'D68.8',
    'OUTROS DESCOLAMENTOS DA RETINA': 'H33.5',
    'OUTROS DESCONFORTOS RESPIRATORIOS DO RECEM-NASCIDO': 'P22.8',
    'OUTROS DESLOCAMENTOS DISCAIS INTERVERTEBRAIS ESPECIFICADOS': 'M51.2',
    'OUTROS DISTURBIOS DA ABSORCAO INTESTINAL DE CARBOIDRATOS': 'E74.3',
    'OUTROS DISTURBIOS DA COORDENACAO': 'R27',
    'OUTROS DISTURBIOS DA COORDENACAO E OS NAO ESPECIFICADOS': 'R27.8',
    'OUTROS DISTURBIOS DA FALA E OS NAO ESPECIFICADOS': 'R47.8',
    'OUTROS DISTURBIOS DA LACTACAO E OS NAO ESPECIFICADOS': 'O92.7',
    'OUTROS DISTURBIOS DA VOZ E OS NAO ESPECIFICADOS': 'R49.8',
    'OUTROS DISTURBIOS DO DESENVOLVIMENTO DOS DENTES': 'K00.8',
    'OUTROS DISTURBIOS DO METABOLISMO DA BILIRRUBINA': 'E80.6',
    'OUTROS DISTURBIOS DO METABOLISMO DE LIPOPROTEINAS': 'E78.8',
    'OUTROS DISTURBIOS DO OLFATO E DO PALADAR E OS NAO ESPECIFICADOS': 'R43.8',
    'OUTROS DISTURBIOS DO SONO': 'G47.8',
    'OUTROS DISTURBIOS E OS NAO ESPECIFICADAS DA SENSIBILIDADE CUTANEA': 'R20.8',
    'OUTROS DISTURBIOS METABOLICOS': 'E88',
    'OUTROS DISTURBIOS VISUAIS': 'H53.8',
    'OUTROS EDEMAS DA CORNEA': 'H18.2',
    'OUTROS EFEITOS ADVERSOS NAO CLASSIFICADOS EM OUTRA PARTE': 'T78.8',
    'OUTROS EFEITOS DA PRESSAO ATMOSFERICA OU DA PRESSAO DA AGUA': 'T70.8',
    'OUTROS EFEITOS DA TEMPERATURA REDUZIDA': 'T69',
    'OUTROS EFEITOS DE PRIVACAO': 'T73.8',
    'OUTROS EFEITOS DO CALOR E DA LUZ': 'T67.8',
    'OUTROS EPISODIOS DEPRESSIVOS': 'F32.8',
    'OUTROS ESTADOS POS-CIRURGICOS ESPECIFICADOS': 'Z98.8',
    'OUTROS ESTAFILOCOCOS COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B95.7',
    'OUTROS ESTRABISMOS PARALITICOS': 'H49.8',
    'OUTROS EXAMES E INVESTIGACOES ESPECIAIS DE PESSOAS SEM QUEIXA OU DIAGNOSTICO RELATADO': 'Z01',
    'OUTROS EXAMES ESPECIAIS ESPECIFICADOS': 'Z01.8',
    'OUTROS EXAMES GERAIS': 'Z00.8',
    'OUTROS EXAMES PARA PROPOSITOS ADMINISTRATIVOS': 'Z02.8',
    'OUTROS HIPOTIREOIDISMOS': 'E03',
    'OUTROS LINFOMAS DE CELULAS T E OS NAO ESPECIFICADOS': 'C84.5',
    'OUTROS MOVIMENTOS INVOLUNTARIOS ANORMAIS E OS NAO ESPECIFICADOS': 'R25.8',
    'OUTROS ORIFICIOS ARTIFICIAIS DO TRATO GASTROINTESTINAL': 'Z93.4',
    'OUTROS PROBLEMAS DE ALIMENTACAO DO RECEM-NASCIDO': 'P92.8',
    'OUTROS PROBLEMAS E OS NAO ESPECIFICADOS RELACIONADOS COM O EMPREGO': 'Z56.7',
    'OUTROS PROBLEMAS RELACIONADOS COM A HABITACAO E COM AS CIRCUNSTANCIAS ECONOMICAS': 'Z59.8',
    'OUTROS PROBLEMAS RELACIONADOS COM O GRUPO PRIMARIO DE APOIO INCLUSIVE COM A SITUACAO FAMILIAR': 'Z63',
    'OUTROS PROBLEMAS RELACIONADOS COM O MEIO SOCIAL': 'Z60.8',
    'OUTROS RECEM-NASCIDOS DE PESO BAIXO': 'P07.1',
    'OUTROS RECEM-NASCIDOS DE PRE-TERMO': 'P07.3',
    'OUTROS RESULTADOS ANORMAIS DE EXAMES PARA DIAGNOSTICO POR IMAGEM DO SISTEMA NERVOSO CENTRAL': 'R90.8',
    'OUTROS SANGRAMENTOS ANORMAIS DO UTERO E DA VAGINA': 'N93',
    'OUTROS SANGRAMENTOS ANORMAIS ESPECIFICADOS DO UTERO E DA VAGINA': 'N93.8',
    'OUTROS SINTOMAS E SINAIS DA MAMA': 'N64.5',
    'OUTROS SINTOMAS E SINAIS ESPECIFICADOS RELATIVOS AO APARELHO DIGESTIVO E AO ABDOME': 'R19.8',
    'OUTROS SINTOMAS E SINAIS ESPECIFICADOS RELATIVOS AOS APARELHOS CIRCULATORIO E RESPIRATORIO': 'R09.8',
    'OUTROS SINTOMAS E SINAIS ESPECIFICADOS RELATIVOS AS FUNCOES COGNITIVAS E A CONSCIENCIA': 'R41.8',
    'OUTROS SINTOMAS E SINAIS ESPECIFICADOS RELATIVOS AS SENSACOES E PERCEPCOES GERAIS': 'R44.8',
    'OUTROS SINTOMAS E SINAIS GERAIS': 'R68',
    'OUTROS SINTOMAS E SINAIS GERAIS ESPECIFICADOS': 'R68.8',
    'OUTROS SINTOMAS E SINAIS RELATIVOS A FUNCAO COGNITIVA E A CONSCIENCIA': 'R41',
    'OUTROS SINTOMAS E SINAIS RELATIVOS A INGESTAO DE ALIMENTOS E DE LIQUIDOS': 'R63.8',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AO APARELHO DIGESTIVO E AO ABDOME': 'R19',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AO APARELHO URINARIO': 'R39',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AO APARELHO URINARIO E OS NAO ESPECIFICADOS': 'R39.8',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AO ESTADO EMOCIONAL': 'R45.8',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AOS APARELHOS CIRCULATORIO E RESPIRATORIO': 'R09',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AOS SISTEMAS NERVOSO E OSTEOMUSCULAR': 'R29',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AOS SISTEMAS NERVOSO E OSTEOMUSCULAR E OS NAO ESPECIFICADOS': 'R29.8',
    'OUTROS SINTOMAS E SINAIS RELATIVOS AS SENSACOES E AS PERCEPCOES GERAIS': 'R44',
    'OUTROS TIPOS DE ABORTO': 'O05',
    'OUTROS TIPOS DE ABORTO - INCOMPLETO, COMPLICADO POR HEMORRAGIA EXCESSIVA OU TARDIA': 'O05.1',
    'OUTROS TIPOS DE ABORTO - INCOMPLETO, SEM COMPLICACOES': 'O05.4',
    'OUTROS TIPOS DE HIPOTENSAO': 'I95.8',
    'OUTROS TIPOS DE LINFOMA NAO-HODGKIN DIFUSO': 'C83.8',
    'OUTROS TIPOS DE TETANO': 'A35',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS': 'E13',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM CETOACIDOSE': 'E13.1',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM COMPLICACOES CIRCULATORIAS PERIFERICAS': 'E13.5',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM COMPLICACOES MULTIPLAS': 'E13.7',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM COMPLICACOES NAO ESPECIFICADAS': 'E13.8',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM COMPLICACOES NEUROLOGICAS': 'E13.4',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM COMPLICACOES RENAIS': 'E13.2',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - COM OUTRAS COMPLICACOES ESPECIFICADAS': 'E13.6',
    'OUTROS TIPOS ESPECIFICADOS DE DIABETES MELLITUS - SEM COMPLICACOES': 'E13.9',
    'OUTROS TIPOS ESPECIFICADOS DE IRREGULARIDADE DA MENSTRUACAO': 'N92.5',
    'OUTROS TIPOS ESPECIFICADOS DE LINFOMA NAO-HODGKIN': 'C85.7',
    'OUTROS TRANSTORNOS ANSIOSOS ESPECIFICADOS': 'F41.8',
    'OUTROS TRANSTORNOS ANSIOSOS MISTOS': 'F41.3',
    'OUTROS TRANSTORNOS ARTICULARES ESPECIFICADOS': 'M25.8',
    'OUTROS TRANSTORNOS ARTICULARES ESPECIFICOS, NAO CLASSIFICADOS EM OUTRA PARTE': 'M24.8',
    'OUTROS TRANSTORNOS CEREBROVASCULARES EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'I68.8',
    'OUTROS TRANSTORNOS DA ALIMENTACAO': 'F50.8',
    'OUTROS TRANSTORNOS DA CONJUNTIVA EM DOENCA CLASSIFICADAS EM OUTRA PARTE': 'H13.8',
    'OUTROS TRANSTORNOS DA ESCLEROTICA': 'H15.8',
    'OUTROS TRANSTORNOS DA FUNCAO VESTIBULAR': 'H81.8',
    'OUTROS TRANSTORNOS DA GLANDULA LACRIMAL': 'H04.1',
    'OUTROS TRANSTORNOS DA IRIS E DO CORPO CILIAR EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H22.8',
    'OUTROS TRANSTORNOS DA ORBITA': 'H05.8',
    'OUTROS TRANSTORNOS DA ORBITA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H06.3',
    'OUTROS TRANSTORNOS DA ROTULA': 'M22.8',
    'OUTROS TRANSTORNOS DA SECRECAO PANCREATICA INTERNA': 'E16',
    'OUTROS TRANSTORNOS DA URETRA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N37.8',
    'OUTROS TRANSTORNOS DA VISAO BINOCULAR': 'H53.3',
    'OUTROS TRANSTORNOS DAS CARTILAGENS ARTICULARES': 'M24.1',
    'OUTROS TRANSTORNOS DE DISCOS CERVICAIS': 'M50.8',
    'OUTROS TRANSTORNOS DEGENERATIVOS DA PALPEBRA E DA AREA PERIOCULAR': 'H02.7',
    'OUTROS TRANSTORNOS DEGENERATIVOS DO GLOBO OCULAR': 'H44.3',
    'OUTROS TRANSTORNOS DELIRANTES PERSISTENTES': 'F22.8',
    'OUTROS TRANSTORNOS DEPRESSIVOS RECORRENTES': 'F33.8',
    'OUTROS TRANSTORNOS DISSOCIATIVOS [DE CONVERSAO]': 'F44.8',
    'OUTROS TRANSTORNOS DO APARELHO CIRCULATORIO E OS NAO ESPECIFICADOS': 'I99',
    'OUTROS TRANSTORNOS DO APARELHO CIRCULATORIO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'I98',
    'OUTROS TRANSTORNOS DO APARELHO DIGESTIVO, POS-CIRURGICOS, NAO CLASSIFICADOS EM OUTRA PARTE': 'K91.8',
    'OUTROS TRANSTORNOS DO APARELHO LACRIMAL': 'H04.8',
    'OUTROS TRANSTORNOS DO DESENVOLVIMENTO E DO CRESCIMENTO OSSEO': 'M89.2',
    'OUTROS TRANSTORNOS DO DISCO OPTICO': 'H47.3',
    'OUTROS TRANSTORNOS DO EQUILIBRIO HIDROELETROLITICO NAO CLASSIFICADOS EM OUTRA PARTE': 'E87.8',
    'OUTROS TRANSTORNOS DO GLOBO OCULAR': 'H44.8',
    'OUTROS TRANSTORNOS DO MENISCO': 'M23.3',
    'OUTROS TRANSTORNOS DO NERVO FACIAL': 'G51.8',
    'OUTROS TRANSTORNOS DO NERVO TRIGEMEO': 'G50.8',
    'OUTROS TRANSTORNOS DO OLHO E ANEXOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H58',
    'OUTROS TRANSTORNOS DO OLHO E ANEXOS POS-PROCEDIMENTOS': 'H59.8',
    'OUTROS TRANSTORNOS DO OUVIDO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H94',
    'OUTROS TRANSTORNOS DO OUVIDO EXTERNO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H62.8',
    'OUTROS TRANSTORNOS DO OUVIDO MEDIO E DA MASTOIDE EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H75',
    'OUTROS TRANSTORNOS DO RIM E DO URETER EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N29.8',
    'OUTROS TRANSTORNOS DO SANGUE E DOS ORGAOS HEMATOPOETICOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'D77',
    'OUTROS TRANSTORNOS DO SISTEMA NERVOSO AUTONOMO': 'G90.8',
    'OUTROS TRANSTORNOS DO SISTEMA NERVOSO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G99',
    'OUTROS TRANSTORNOS DO SISTEMA NERVOSO NAO CLASSIFICADOS EM OUTRA PARTE': 'G98',
    'OUTROS TRANSTORNOS DO SISTEMA NERVOSO PERIFERICO': 'G64',
    'OUTROS TRANSTORNOS DOS GLOBULOS BRANCOS': 'D72',
    'OUTROS TRANSTORNOS DOS HABITOS E DOS IMPULSOS': 'F63.8',
    'OUTROS TRANSTORNOS DOS ORGAOS GENITAIS MASCULINOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N51.8',
    'OUTROS TRANSTORNOS DOS TECIDOS MOLES EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M73.8',
    'OUTROS TRANSTORNOS DOS TECIDOS MOLES RELACIONADOS COM O USO, USO EXCESSIVO E PRESSAO': 'M70.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA BEXIGA': 'N32.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA CARTILAGEM': 'M94.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA CONDUCAO': 'I45.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA CONJUNTIVA': 'H11.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA DENSIDADE E DA ESTRUTURA OSSEAS': 'M85.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA GENGIVA E DO REBORDO ALVEOLAR SEM DENTES': 'K06.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA MAMA': 'N64.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA MEMBRANA DO TIMPANO': 'H73.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA MENOPAUSA E DA PERIMENOPAUSA': 'N95.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA PIGMENTACAO': 'L81.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA RETINA': 'H35.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA SECRECAO PANCREATICA INTERNA': 'E16.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DA SINOVIA E DO TENDAO': 'M67.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DAS PALPEBRAS': 'H02.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DE DISCOS INTERVERTEBRAIS': 'M51.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO APARELHO URINARIO': 'N39.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO NARIZ E DOS SEIOS PARANASAIS': 'J34.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OLHO E ANEXOS': 'H57.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OLHO E ANEXOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H58.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OUVIDO': 'H93.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OUVIDO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H94.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OUVIDO EXTERNO': 'H61.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OUVIDO INTERNO': 'H83.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO OUVIDO MEDIO E DA MASTOIDE': 'H74.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DO PENIS': 'N48.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DOS DENTES E DAS ESTRUTURAS DE SUSTENTACAO': 'K08.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DOS GLOBULOS BRANCOS': 'D72.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DOS ORGAOS GENITAIS MASCULINOS': 'N50.8',
    'OUTROS TRANSTORNOS ESPECIFICADOS DOS TECIDOS MOLES': 'M79.8',
    'OUTROS TRANSTORNOS FIBROBLASTICOS': 'M72.8',
    'OUTROS TRANSTORNOS FOBICO-ANSIOSOS': 'F40.8',
    'OUTROS TRANSTORNOS FUNCIONAIS ESPECIFICADOS DO INTESTINO': 'K59.8',
    'OUTROS TRANSTORNOS INFLAMATORIOS DO PENIS': 'N48.2',
    'OUTROS TRANSTORNOS INTERNOS DO JOELHO': 'M23.8',
    'OUTROS TRANSTORNOS MUSCULARES EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M63.8',
    'OUTROS TRANSTORNOS MUSCULARES ESPECIFICADOS': 'M62.8',
    'OUTROS TRANSTORNOS NAO-INFLAMATORIOS DO OVARIO, DA TROMPA DE FALOPIO E DO LIGAMENTO LARGO': 'N83.8',
    'OUTROS TRANSTORNOS NAO-INFLAMATORIOS ESPECIFICADOS DA VAGINA': 'N89.8',
    'OUTROS TRANSTORNOS NAO-INFLAMATORIOS ESPECIFICADOS DA VULVA E DO PERINEO': 'N90.8',
    'OUTROS TRANSTORNOS POS-PROCEDIMENTO DO SISTEMA NERVOSO': 'G97.8',
    'OUTROS TRANSTORNOS POS-PROCEDIMENTOS DO APARELHO GENITURINARIO': 'N99.8',
    'OUTROS TRANSTORNOS PRIMARIOS DOS MUSCULOS': 'G71.8',
    'OUTROS TRANSTORNOS PSICOTICOS AGUDOS E TRANSITORIOS': 'F23.8',
    'OUTROS TRANSTORNOS PSICOTICOS AGUDOS, ESSENCIALMENTE DELIRANTES': 'F23.3',
    'OUTROS TRANSTORNOS PSICOTICOS NAO-ORGANICOS': 'F28',
    'OUTROS TRANSTORNOS PULMONARES': 'J98.4',
    'OUTROS TRANSTORNOS QUE AFETAM A FUNCAO DA PALPEBRA': 'H02.5',
    'OUTROS TRANSTORNOS QUE COMPROMETEM O MECANISMO IMUNITARIO NAO CLASSIFICADOS EM OUTRA PARTE': 'D89',
    'OUTROS TRANSTORNOS RESPIRATORIOS ESPECIFICADOS': 'J98.8',
    'OUTROS TRANSTORNOS SOMATOFORMES': 'F45.8',
    'OUTROS TRANSTORNOS VASCULARES E CISTOS CONJUNTIVAIS': 'H11.4',
    'OUTROS TRANSTORNOS VENOSOS ESPECIFICADOS': 'I87.8',
    'OUTROS TRAUMATISMOS DA CABECA E OS NAO ESPECIFICADOS': 'S09',
    'OUTROS TRAUMATISMOS DE COLUNA E TRONCO NIVEL NAO ESPECIFICADO': 'T09',
    'OUTROS TRAUMATISMOS DE MEMBRO INFERIOR NIVEL NAO ESPECIFICADO': 'T13',
    'OUTROS TRAUMATISMOS DE MEMBRO SUPERIOR NIVEL NAO ESPECIFICADO': 'T11',
    'OUTROS TRAUMATISMOS DE PARTO DO SISTEMA NERVOSO CENTRAL': 'P11',
    'OUTROS TRAUMATISMOS DE PARTO ESPECIFICADOS': 'P15.8',
    'OUTROS TRAUMATISMOS DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.8',
    'OUTROS TRAUMATISMOS DO ANTEBRACO E OS NAO ESPECIFICADOS': 'S59',
    'OUTROS TRAUMATISMOS DO OLHO E DA ORBITA': 'S05.8',
    'OUTROS TRAUMATISMOS DO PESCOCO E OS NAO ESPECIFICADOS': 'S19',
    'OUTROS TRAUMATISMOS DO PULMAO': 'S27.3',
    'OUTROS TRAUMATISMOS DO TORAX E OS NAO ESPECIFICADOS': 'S29',
    'OUTROS TRAUMATISMOS E OS NAO ESPECIFICADOS DA MEDULA CERVICAL': 'S14.1',
    'OUTROS TRAUMATISMOS E OS NAO ESPECIFICADOS DA PERNA': 'S89',
    'OUTROS TRAUMATISMOS E OS NAO ESPECIFICADOS DO ABDOME DO DORSO E DA PELVE': 'S39',
    'OUTROS TRAUMATISMOS E OS NAO ESPECIFICADOS DO QUADRIL E DA COXA': 'S79',
    'OUTROS TRAUMATISMOS E OS NAO ESPECIFICADOS DO TORNOZELO E DO PE': 'S99',
    'OUTROS TRAUMATISMOS ENVOLVENDO REGIOES MULTIPLAS DO CORPO NAO CLASSIFICADOS EM OUTRA PARTE': 'T06',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DA CABECA': 'S09.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DA PERNA': 'S89.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO ANTEBRACO': 'S59.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO MEMBRO SUPERIOR NIVEL NAO ESPECIFICADO': 'T11.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO OMBRO E DO BRACO': 'S49.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO PESCOCO': 'S19.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO PUNHO E DA MAO': 'S69.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO QUADRIL E DA COXA': 'S79.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO TORAX': 'S29.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO TORNOZELO E DO PE': 'S99.8',
    'OUTROS TRAUMATISMOS ESPECIFICADOS DO TRONCO, NIVEL NAO ESPECIFICADO': 'T09.8',
    'OUTROS TRAUMATISMOS INTRACRANIANOS': 'S06.8',
    'OUTROS TRAUMATISMOS MULTIPLOS DO ABDOME, DO DORSO E DA PELVE': 'S39.7',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA GARGANTA E OS NAO ESPECIFICADOS': 'S10.1',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA MAMA E OS NAO ESPECIFICADOS': 'S20.1',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA PALPEBRA E DA REGIAO PERIOCULAR': 'S00.2',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA PAREDE ANTERIOR DO TORAX': 'S20.3',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA PAREDE POSTERIOR DO TORAX': 'S20.4',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DA PERNA': 'S80.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO ABDOME, DO DORSO E DA PELVE': 'S30.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO ANTEBRACO': 'S50.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO OMBRO E DO BRACO': 'S40.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO PUNHO E DA MAO': 'S60.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO QUADRIL E DA COXA': 'S70.8',
    'OUTROS TRAUMATISMOS SUPERFICIAIS DO TORNOZELO E DO PE': 'S90.8',
    'OVULACAO DOLOROSA [MITTELSCHMERZ]': 'N94.0',
    'OXIURIASE': 'B80',
    'PALIDEZ': 'R23.1',
    'PALPITACOES': 'R00.2',
    'PANCREATITE AGUDA': 'K85',
    'PANCREATITE AGUDA BILIAR': 'K85.1',
    'PANCREATITE AGUDA IDIOPATICA': 'K85.0',
    'PANCREATITE AGUDA INDUZIDA POR ALCOOL': 'K85.2',
    'PANCREATITE AGUDA, NAO ESPECIFICADA': 'K85.9',
    'PANCREATITE CRONICA INDUZIDA POR ALCOOL': 'K86.0',
    'PANCREATITE POR CAXUMBA [PAROTIDITE EPIDEMICA]': 'B26.3',
    'PANCREATITE POR CITOMEGALOVIRUS': 'B25.2',
    'PANICULITE ATINGINDO REGIOES DO PESCOCO E DO DORSO': 'M54.0',
    'PANICULITE NAO ESPECIFICADA': 'M79.3',
    'PANSINUSITE AGUDA': 'J01.4',
    'PANSINUSITE CRONICA': 'J32.4',
    'PAPILEDEMA NAO ESPECIFICADO': 'H47.1',
    'PAPILOMAS MULTIPLOS E BOUBA PLANTAR UMIDA (CRAVO DE BOUBA)': 'A66.1',
    'PAPILOMAVIRUS, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97.7',
    'PARACOCCIDIOIDOMICOSE': 'B41',
    'PARADA CARDIACA NAO ESPECIFICADA': 'I46.9',
    'PARAGEUSIA': 'R43.2',
    'PARALISIA CEREBRAL DIPLEGICA ESPASTICA': 'G80.1',
    'PARALISIA CEREBRAL NAO ESPECIFICADA': 'G80.9',
    'PARALISIA DE BELL': 'G51.0',
    'PARALISIA DO OLHAR CONJUGADO': 'H51.0',
    'PARALISIA DO SEXTO PAR [ABDUCENTE]': 'H49.2',
    'PARALISIA DO TERCEIRO PAR [OCULOMOTOR]': 'H49.0',
    'PARALISIA PERIODICA': 'G72.3',
    'PARAMETRITE E CELULITE PELVICAS AGUDAS': 'N73.0',
    'PARAPLEGIA NAO ESPECIFICADA': 'G82.2',
    'PARASITOSE INTESTINAL NAO ESPECIFICADA': 'B82.9',
    'PARESTESIAS CUTANEAS': 'R20.2',
    'PARKINSONISMO SECUNDARIO': 'G21',
    'PARVOVIRUS, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97.6',
    'PE CHATO [PE PLANO] (ADQUIRIDO)': 'M21.4',
    'PE TORTO CALCANEOVALGO': 'Q66.4',
    'PEDESTRE TRAUMATIZADO EM COLISAO COM UM AUTOMOVEL (CARRO), PICK-UP OU CAMINHONETE': 'V03',
    'PEDESTRE TRAUMATIZADO EM COLISAO COM UM VEICULO A MOTOR DE DUAS OU TRES RODAS': 'V02',
    'PEDESTRE TRAUMATIZADO EM COLISAO COM UM VEICULO A PEDAL': 'V01',
    'PEDESTRE TRAUMATIZADO EM COLISAO COM UM VEICULO DE TRANSPORTE PESADO OU COM UM ONIBUS': 'V04',
    'PEDICULOSE DEVIDA A PEDICULUS HUMANUS CAPITIS': 'B85.0',
    'PEDICULOSE DEVIDA A PEDICULUS HUMANUS CORPORIS': 'B85.1',
    'PEDICULOSE E FTIRIASE': 'B85',
    'PEDICULOSE NAO ESPECIFICADA': 'B85.2',
    'PELVIPERITONITE AGUDA FEMININA': 'N73.3',
    'PELVIPERITONITE GONOCOCICA E OUTRAS INFECCOES GENITURINARIAS GONOCOCICAS': 'A54.2',
    'PENETRACAO DE CORPO ESTRANHO NO OU ATRAVES DE OLHO OU ORIFICIO NATURAL': 'W44',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE': 'W45',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - AREAS DE COMERCIO E DE SERVICOS': 'W45.5',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W45.6',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - HABITACAO COLETIVA': 'W45.1',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - LOCAL NAO ESPECIFICADO': 'W45.9',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - OUTROS LOCAIS ESPECIFICADOS': 'W45.8',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - RESIDENCIA': 'W45.0',
    'PENETRACAO DE CORPO OU OBJETO ESTRANHO ATRAVES DA PELE - RUA E ESTRADA': 'W45.4',
    'PENFIGO': 'L10',
    'PENFIGO BRASILEIRO [FOGO SELVAGEM]': 'L10.3',
    'PENFIGO ERITEMATOSO': 'L10.4',
    'PENFIGO FOLIACEO': 'L10.2',
    'PENFIGO VULGAR': 'L10.0',
    'PENFIGO, NAO ESPECIFICADO': 'L10.9',
    'PENFIGOIDE BOLHOSO': 'L12.0',
    'PEQUENO MAL NAO ESPECIFICADO, SEM CRISES DE GRANDE MAL': 'G40.7',
    'PERDA DE AUDICAO BILATERAL DEVIDA A TRANSTORNO DE CONDUCAO': 'H90.0',
    'PERDA DE AUDICAO BILATERAL MISTA, DE CONDUCAO E NEURO-SENSORIAL': 'H90.6',
    'PERDA DE AUDICAO MISTA, DE CONDUCAO E NEURO-SENSORIAL, NAO ESPECIFICADA': 'H90.8',
    'PERDA DE AUDICAO NEURO-SENSORIAL NAO ESPECIFICADA': 'H90.5',
    'PERDA DE AUDICAO POR TRANSTORNO DE CONDUCAO E/OU NEURO-SENSORIAL': 'H90',
    'PERDA DE AUDICAO SUBITA IDIOPATICA': 'H91.2',
    'PERDA DE AUDICAO UNILATERAL NEURO-SENSORIAL, SEM RESTRICAO DE AUDICAO CONTRALATERAL': 'H90.4',
    'PERDA DE CABELOS OU PELOS NAO CICATRICIAIS, NAO ESPECIFICADA': 'L65.9',
    'PERDA DE DENTES DEVIDA A ACIDENTE, EXTRACAO OU A DOENCAS PERIODONTAIS LOCALIZADAS': 'K08.1',
    'PERDA DE LIQUOR RESULTANTE DE PUNCAO ESPINAL': 'G97.0',
    'PERDA DE PESO ANORMAL': 'R63.4',
    'PERDA E ATROFIA MUSCULAR NAO CLASSIFICADAS EM OUTRA PARTE': 'M62.5',
    'PERDA NAO ESPECIFICADA DA VISAO': 'H54.7',
    'PERDA NAO ESPECIFICADA DE AUDICAO': 'H91.9',
    'PERDA NAO ESPECIFICADA DE AUDICAO DEVIDA A TRANSTORNO DE CONDUCAO': 'H90.2',
    'PERDA NAO QUALIFICADA DA VISAO EM AMBOS OS OLHOS': 'H54.3',
    'PERDA NAO QUALIFICADA DA VISAO EM UM OLHO': 'H54.6',
    'PERDA SANGUINEA FETAL NAO ESPECIFICADA': 'P50.9',
    'PERFURACAO CENTRAL DA MEMBRANA DO TIMPANO': 'H72.0',
    'PERFURACAO DA MEMBRANA DO TIMPANO': 'H72',
    'PERFURACAO DO INTESTINO (NAO-TRAUMATICA)': 'K63.1',
    'PERFURACAO DO LOBO DA ORELHA': 'Z41.3',
    'PERFURACAO E LACERACAO ACIDENTAIS DURANTE UM PROCEDIMENTO NAO CLASSIFICADO EM OUTRA PARTE': 'T81.2',
    'PERFURACAO NAO ESPECIFICADA DA MEMBRANA DO TIMPANO': 'H72.9',
    'PERIARTRITE DO PUNHO': 'M77.2',
    'PERICARDITE AGUDA': 'I30',
    'PERICARDITE AGUDA IDIOPATICA NAO ESPECIFICA': 'I30.0',
    'PERICARDITE AGUDA NAO ESPECIFICADA': 'I30.9',
    'PERICARDITE REUMATICA AGUDA': 'I01.0',
    'PERICONDRITE DO PAVILHAO DA ORELHA': 'H61.0',
    'PERIFOLICULITE ABSCEDANTE DA CABECA': 'L66.3',
    'PERIODONTITE AGUDA': 'K05.2',
    'PERIODONTITE APICAL AGUDA DE ORIGEM PULPAR': 'K04.4',
    'PERIODONTITE CRONICA': 'K05.3',
    'PERIODONTOSE': 'K05.4',
    'PERITONITE': 'K65',
    'PERSONALIDADE ANSIOSA [ESQUIVA]': 'F60.6',
    'PERSONALIDADE SUSPEITA E EVASIVA': 'R46.5',
    'PESSOA COM MEDO DE UMA QUEIXA PARA A QUAL NAO FOI FEITO DIAGNOSTICO': 'Z71.1',
    'PESSOA EM BOA SAUDE ACOMPANHANDO PESSOA DOENTE': 'Z76.3',
    'PESSOA EM CONTATO COM SERVICOS DE SAUDE EM CIRCUNSTANCIAS NAO ESPECIFICADAS': 'Z76.9',
    'PESSOA FINGINDO SER DOENTE [SIMULACAO CONSCIENTE]': 'Z76.5',
    'PESSOA QUE CONSULTA NO INTERESSE DE UM TERCEIRO': 'Z71.0',
    'PESSOA QUE CONSULTA PARA EXPLICACAO DE ACHADOS DE EXAME': 'Z71.2',
    'PESSOAS EM CONTATO COM SERVICOS DE SAUDE PARA PROCEDIMENTOS ESPECIFICOS NAO REALIZADOS': 'Z53',
    'PESTE BUBONICA': 'A20.0',
    'PESTE PNEUMONICA': 'A20.2',
    'PICA DO LACTENTE OU DA CRIANCA': 'F98.3',
    'PIELONEFRITE NAO-OBSTRUTIVA CRONICA ASSOCIADA A REFLUXO': 'N11.0',
    'PIELONEFRITE OBSTRUTIVA CRONICA': 'N11.1',
    'PIODERMITE': 'L08.0',
    'PIODERMITE GANGRENOSA': 'L88',
    'PIONEFROSE': 'N13.6',
    'PIROSE': 'R12',
    'PITIRIASE ALBA': 'L30.5',
    'PITIRIASE ROSEA': 'L42',
    'PITIRIASE VERSICOLOR': 'B36.0',
    'PLEURISIA': 'R09.1',
    'PLICOMAS HEMORROIDARIOS RESIDUAIS': 'I84.6',
    'PNEUMOCONIOSE DEVIDA A OUTRAS POEIRAS QUE CONTENHAM SILICA': 'J62.8',
    'PNEUMONIA BACTERIANA NAO CLASSIFICADA EM OUTRA PARTE': 'J15',
    'PNEUMONIA BACTERIANA NAO ESPECIFICADA': 'J15.9',
    'PNEUMONIA CONGENITA': 'P23',
    'PNEUMONIA CONGENITA DEVIDA A AGENTE VIRAL': 'P23.0',
    'PNEUMONIA CONGENITA DEVIDA A ESTREPTOCOCO DO GRUPO B': 'P23.3',
    'PNEUMONIA CONGENITA NAO ESPECIFICADA': 'P23.9',
    'PNEUMONIA DEVIDA A ADENOVIRUS': 'J12.0',
    'PNEUMONIA DEVIDA A CLAMIDIAS': 'J16.0',
    'PNEUMONIA DEVIDA A HAEMOPHILUS INFUENZAE': 'J14',
    'PNEUMONIA DEVIDA A KLEBSIELLA PNEUMONIAE': 'J15.0',
    'PNEUMONIA DEVIDA A MYCOPLASMA PNEUMONIAE': 'J15.7',
    'PNEUMONIA DEVIDA A OUTRAS BACTERIAS AEROBICAS GRAM-NEGATIVAS': 'J15.6',
    'PNEUMONIA DEVIDA A OUTROS ESTREPTOCOCOS': 'J15.4',
    'PNEUMONIA DEVIDA A OUTROS MICROORGANISMOS INFECCIOSOS ESPECIFICADOS': 'J16.8',
    'PNEUMONIA DEVIDA A OUTROS MICROORGANISMOS INFECCIOSOS ESPECIFICADOS NAO CLASSIFICADOS EM OUTRA PARTE': 'J16',
    'PNEUMONIA DEVIDA A PARAINFLUENZA': 'J12.2',
    'PNEUMONIA DEVIDA A STAPHYLOCOCCUS': 'J15.2',
    'PNEUMONIA DEVIDA A STREPTOCOCCUS DO GRUPO B': 'J15.3',
    'PNEUMONIA DEVIDA A STREPTOCOCCUS PNEUMONIAE': 'J13',
    'PNEUMONIA DEVIDA A VIRUS RESPIRATORIO SINCICIAL': 'J12.1',
    'PNEUMONIA EM DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'J17.0',
    'PNEUMONIA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'J17',
    'PNEUMONIA EM DOENCAS VIRAIS CLASSIFICADAS EM OUTRA PARTE': 'J17.1',
    'PNEUMONIA EM MICOSES CLASSIFICADAS EM OUTRA PARTE': 'J17.2',
    'PNEUMONIA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'J17.8',
    'PNEUMONIA LOBAR NAO ESPECIFICADA': 'J18.1',
    'PNEUMONIA NAO ESPECIFICADA': 'J18.9',
    'PNEUMONIA POR MICROORGANISMO NAO ESPECIFICADA': 'J18',
    'PNEUMONIA VIRAL NAO CLASSIFICADA EM OUTRA PARTE': 'J12',
    'PNEUMONIA VIRAL NAO ESPECIFICADA': 'J12.9',
    'PNEUMONITE DEVIDA A ALIMENTO OU VOMITO': 'J69.0',
    'PNEUMONITE DEVIDA A SOLIDOS E LIQUIDOS': 'J69',
    'PNEUMONITE POR CITOMEGALOVIRUS': 'B25.0',
    'PNEUMONITES DE HIPERSENSIBILIDADE, DEVIDAS A OUTRAS POEIRAS ORGANICAS': 'J67.8',
    'PNEUMOTORAX': 'J93',
    'PNEUMOTORAX DE TENSAO, ESPONTANEO': 'J93.0',
    'PNEUMOTORAX NAO ESPECIFICADO': 'J93.9',
    'POLIARTERITE NODOSA E AFECCOES CORRELATAS': 'M30',
    'POLIARTRITE NAO ESPECIFICADA': 'M13.0',
    'POLIARTROPATIA INFLAMATORIA': 'M06.4',
    'POLIARTROSE': 'M15',
    'POLIARTROSE NAO ESPECIFICADA': 'M15.9',
    'POLICITEMIA VERA': 'D45',
    'POLICONDRITE RECIDIVANTE': 'M94.1',
    'POLIDACTILIA': 'Q69',
    'POLIDIPSIA': 'R63.1',
    'POLIMIALGIA REUMATICA': 'M35.3',
    'POLIMIOSITE': 'M33.2',
    'POLINEUROPATIA ALCOOLICA': 'G62.1',
    'POLINEUROPATIA DIABETICA': 'G63.2',
    'POLINEUROPATIA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G63.8',
    'POLINEUROPATIA EM OUTRAS DOENCAS ENDOCRINAS E METABOLICAS': 'G63.3',
    'POLINEUROPATIA EM OUTROS TRANSTORNOS OSTEOMUSCULARES': 'G63.6',
    'POLINEUROPATIA INFLAMATORIA': 'G61',
    'POLINEUROPATIA INFLAMATORIA NAO ESPECIFICADA': 'G61.9',
    'POLINEUROPATIA NAO ESPECIFICADA': 'G62.9',
    'POLIOMIELITE PARALITICA AGUDA, ASSOCIADA AO VIRUS VACINAL': 'A80.0',
    'POLIPO ANAL': 'K62.0',
    'POLIPO DA CAVIDADE NASAL': 'J33.0',
    'POLIPO DAS CORDAS VOCAIS E DA LARINGE': 'J38.1',
    'POLIPO DO COLON': 'K63.5',
    'POLIPO DO OUVIDO MEDIO': 'H74.4',
    'POLIPO DO TRATO GENITAL FEMININO': 'N84',
    'POLIPO NASAL': 'J33',
    'POLIPO NASAL NAO ESPECIFICADO': 'J33.9',
    'POLIPO RETAL': 'K62.1',
    'POLIURIA': 'R35',
    'PORFIRIA HEREDITARIA ERITROPOETICA': 'E80.0',
    'PORTADOR DE DOENCA INFECCIOSA NAO ESPECIFICADA': 'Z22.9',
    'PORTADOR DE HEPATITE VIRAL': 'Z22.5',
    'PORTADOR DE OUTRAS DOENCAS INFECCIOSAS': 'Z22.8',
    'POS-CATARATA': 'H26.4',
    'PRE-ECLAMPSIA GRAVE': 'O14.1',
    'PRE-ECLAMPSIA MODERADA': 'O14.0',
    'PRE-ECLAMPSIA NAO ESPECIFICADA': 'O14.9',
    'PRESBIACUSIA': 'H91.1',
    'PRESBIOPIA': 'H52.4',
    'PRESENCA DE ALCOOL NO SANGUE': 'R78.0',
    'PRESENCA DE ALCOOL NO SANGUE, TAXA NAO ESPECIFICADA': 'Y90.9',
    'PRESENCA DE COCAINA NO SANGUE': 'R78.2',
    'PRESENCA DE DISPOSITIVO ANTICONCEPCIONAL INTRA-UTERINO [DIU]': 'Z97.5',
    'PRESENCA DE IMPLANTE E ENXERTO CARDIACO E VASCULAR NAO ESPECIFICADO': 'Z95.9',
    'PRESENCA DE IMPLANTE E ENXERTO DE ANGIOPLASTIA CORONARIA': 'Z95.5',
    'PRESENCA DE IMPLANTE OTOLOGICOS E AUDIOLOGICOS': 'Z96.2',
    'PRESENCA DE IMPLANTES ARTICULARES ORTOPEDICOS': 'Z96.6',
    'PRESENCA DE IMPLANTES E ENXERTOS CARDIACOS E VASCULARES': 'Z95',
    'PRESENCA DE MARCA-PASSO CARDIACO': 'Z95.0',
    'PRESENCA DE OUTRAS DROGAS COM POTENCIAL DE CAUSAR DEPENDENCIA, NO SANGUE': 'R78.4',
    'PRESENCA DE PROTESE DE VALVULA CARDIACA': 'Z95.2',
    'PRIAPISMO': 'N48.3',
    'PRISAO OU ENCARCERAMENTO': 'Z65.1',
    'PROBLEMA NAO ESPECIFICADO DE ALIMENTACAO DO RECEM-NASCIDO': 'P92.9',
    'PROBLEMA NAO ESPECIFICADO RELACIONADO COM O MEIO SOCIAL': 'Z60.9',
    'PROBLEMAS DE ALIMENTACAO DO RECEM-NASCIDO': 'P92',
    'PROBLEMAS LIGADOS A LIBERTACAO DE PRISAO': 'Z65.2',
    'PROBLEMAS NAS RELACOES COM CONJUGE OU PARCEIRO': 'Z63.0',
    'PROBLEMAS NAS RELACOES COM OS PAIS OU COM OS SOGROS': 'Z63.1',
    'PROBLEMAS RELACIONADOS COM A DEPENDENCIA DE UMA PESSOA QUE OFERECE CUIDADOS DE SAUDE': 'Z74',
    'PROBLEMAS RELACIONADOS COM A HABITACAO E COM AS CONDICOES ECONOMICAS': 'Z59',
    'PROBLEMAS RELACIONADOS COM A ORGANIZACAO DE SEU MODO DE VIDA': 'Z73',
    'PROBLEMAS RELACIONADOS COM ABUSO FISICO ALEGADO DA CRIANCA': 'Z61.6',
    'PROBLEMAS RELACIONADOS COM ABUSO SEXUAL ALEGADO DE UMA CRIANCA POR PESSOA DE FORA DE SEU GRUPO': 'Z61.5',
    'PROBLEMAS RELACIONADOS COM ABUSO SEXUAL ALEGADO DE UMA CRIANCA POR UMA PESSOA DE DENTRO DE SEU GRUPO': 'Z61.4',
    'PROBLEMAS RELACIONADOS COM EVENTOS NEGATIVOS DE VIDA NA INFANCIA': 'Z61',
    'PROBLEMAS RELACIONADOS COM O EMPREGO E COM O DESEMPREGO': 'Z56',
    'PROBLEMAS RELACIONADOS COM O ESTILO DE VIDA': 'Z72',
    'PROBLEMAS RELACIONADOS COM O MEIO SOCIAL': 'Z60',
    'PROCEDIMENTO NAO REALIZADO DEVIDO A CONTRA-INDICACAO': 'Z53.0',
    'PROCEDIMENTO NAO REALIZADO DEVIDO A DECISAO DO PACIENTE POR OUTRAS RAZOES E AS NAO ESPECIFICADAS': 'Z53.2',
    'PROCEDIMENTO NAO REALIZADO DEVIDO A DECISAO DO PACIENTE POR RAZOES DE CRENCA OU GRUPO DE PRESSAO': 'Z53.1',
    'PROCEDIMENTO NAO REALIZADO POR OUTRAS RAZOES': 'Z53.8',
    'PROCEDIMENTO NAO REALIZADO POR RAZAO NAO ESPECIFICADA': 'Z53.9',
    'PROCEDIMENTOS PARA OUTROS PROPOSITOS EXCETO CUIDADOS DE SAUDE': 'Z41',
    'PROCTITE ULCERATIVA (CRONICA)': 'K51.2',
    'PROCTOCOLITE MUCOSA': 'K51.5',
    'PROLAPSO (DA VALVA) MITRAL': 'I34.1',
    'PROLAPSO ANAL': 'K62.2',
    'PROLAPSO DA MUCOSA URETRAL': 'N36.3',
    'PROLAPSO DE CUPULA DE VAGINA POS-HISTERECTOMIA': 'N99.3',
    'PROLAPSO GENITAL FEMININO': 'N81',
    'PROLAPSO GENITAL FEMININO NAO ESPECIFICADO': 'N81.9',
    'PROLAPSO RETAL': 'K62.3',
    'PROLAPSO UTEROVAGINAL COMPLETO': 'N81.3',
    'PROLAPSO UTEROVAGINAL INCOMPLETO': 'N81.2',
    'PROLAPSO UTEROVAGINAL NAO ESPECIFICADO': 'N81.4',
    'PROSTATITE AGUDA': 'N41.0',
    'PROSTATITE CRONICA': 'N41.1',
    'PROSTATOCISTITE': 'N41.3',
    'PROTEINURIA ISOLADA': 'R80',
    'PROTEINURIA ISOLADA COM LESAO MORFOLOGICA ESPECIFICADA - NAO ESPECIFICADA': 'N06.9',
    'PROTEINURIA ORTOSTATICA NAO ESPECIFICADA': 'N39.2',
    'PROTEINURIA PERSISTENTE NAO ESPECIFICADA': 'N39.1',
    'PRURIDO': 'L29',
    'PRURIDO ANAL': 'L29.0',
    'PRURIDO ANOGENITAL, NAO ESPECIFICADO': 'L29.3',
    'PRURIDO ESCROTAL': 'L29.1',
    'PRURIDO NAO ESPECIFICADO': 'L29.9',
    'PRURIDO VULVAR': 'L29.2',
    'PRURIGO DE BESNIER': 'L20.0',
    'PRURIGO NODULAR': 'L28.1',
    'PSEUDOCOXALGIA': 'M91.3',
    'PSEUDOFOLICULITE DA BARBA': 'L73.1',
    'PSEUDOPOLIPOSE DO COLON': 'K51.4',
    'PSICOSE NAO-ORGANICA NAO ESPECIFICADA': 'F29',
    'PSORIASE': 'L40',
    'PSORIASE GUTATA': 'L40.4',
    'PSORIASE NAO ESPECIFICADA': 'L40.9',
    'PSORIASE PUSTULOSA GENERALIZADA': 'L40.1',
    'PSORIASE VULGAR': 'L40.0',
    'PTERIGIO': 'H11.0',
    'PTOSE DA PALPEBRA': 'H02.4',
    'PUBERDADE PRECOCE': 'E30.1',
    'PULPITE': 'K04.0',
    'PURPURA ALERGICA': 'D69.0',
    'PURPURA E OUTRAS AFECCOES HEMORRAGICAS': 'D69',
    'PURPURA TROMBOCITOPENICA IDIOPATICA': 'D69.3',
    'PUSTULOSE PALMAR E PLANTAR': 'L40.3',
    'QUADRIL ESTALANTE': 'R29.4',
    'QUADRIL INSTAVEL': 'Q65.6',
    'QUEDA DE ARVORE': 'W14',
    'QUEDA DE ARVORE - LOCAL NAO ESPECIFICADO': 'W14.9',
    'QUEDA DE ARVORE - RESIDENCIA': 'W14.0',
    'QUEDA DE OU PARA FORA DE EDIFICIOS OU OUTRAS ESTRUTURAS': 'W13',
    'QUEDA DE OUTRO TIPO DE MOBILIA': 'W08',
    'QUEDA DE OUTRO TIPO DE MOBILIA - AREAS DE COMERCIO E DE SERVICOS': 'W08.5',
    'QUEDA DE OUTRO TIPO DE MOBILIA - HABITACAO COLETIVA': 'W08.1',
    'QUEDA DE OUTRO TIPO DE MOBILIA - LOCAL NAO ESPECIFICADO': 'W08.9',
    'QUEDA DE OUTRO TIPO DE MOBILIA - OUTROS LOCAIS ESPECIFICADOS': 'W08.8',
    'QUEDA DE OUTRO TIPO DE MOBILIA - RESIDENCIA': 'W08.0',
    'QUEDA DE OUTRO TIPO DE MOBILIA - RUA E ESTRADA': 'W08.4',
    'QUEDA DE PENHASCO': 'W15',
    'QUEDA DE UM LEITO': 'W06',
    'QUEDA DE UM LEITO - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W06.3',
    'QUEDA DE UM LEITO - AREAS DE COMERCIO E DE SERVICOS': 'W06.5',
    'QUEDA DE UM LEITO - HABITACAO COLETIVA': 'W06.1',
    'QUEDA DE UM LEITO - LOCAL NAO ESPECIFICADO': 'W06.9',
    'QUEDA DE UM LEITO - RESIDENCIA': 'W06.0',
    'QUEDA DE UM LEITO - RUA E ESTRADA': 'W06.4',
    'QUEDA DE UMA CADEIRA': 'W07',
    'QUEDA DE UMA CADEIRA - AREAS DE COMERCIO E DE SERVICOS': 'W07.5',
    'QUEDA DE UMA CADEIRA - LOCAL NAO ESPECIFICADO': 'W07.9',
    'QUEDA DE UMA CADEIRA - OUTROS LOCAIS ESPECIFICADOS': 'W07.8',
    'QUEDA DE UMA CADEIRA - RESIDENCIA': 'W07.0',
    'QUEDA DE UMA CADEIRA - RUA E ESTRADA': 'W07.4',
    'QUEDA EM OU DE ESCADAS DE MAO': 'W11',
    'QUEDA EM OU DE ESCADAS DE MAO - FAZENDA': 'W11.7',
    'QUEDA EM OU DE ESCADAS DE MAO - HABITACAO COLETIVA': 'W11.1',
    'QUEDA EM OU DE ESCADAS DE MAO - LOCAL NAO ESPECIFICADO': 'W11.9',
    'QUEDA EM OU DE ESCADAS DE MAO - OUTROS LOCAIS ESPECIFICADOS': 'W11.8',
    'QUEDA EM OU DE ESCADAS DE MAO - RESIDENCIA': 'W11.0',
    'QUEDA EM OU DE ESCADAS DE MAO - RUA E ESTRADA': 'W11.4',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS': 'W10',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - AREAS DE COMERCIO E DE SERVICOS': 'W10.5',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W10.6',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - HABITACAO COLETIVA': 'W10.1',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - LOCAL NAO ESPECIFICADO': 'W10.9',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - OUTROS LOCAIS ESPECIFICADOS': 'W10.8',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - RESIDENCIA': 'W10.0',
    'QUEDA EM OU DE ESCADAS OU DEGRAUS - RUA E ESTRADA': 'W10.4',
    'QUEDA EM OU DE UM ANDAIME': 'W12',
    'QUEDA EM OU DE UM ANDAIME - AREAS DE COMERCIO E DE SERVICOS': 'W12.5',
    'QUEDA EM OU DE UM ANDAIME - LOCAL NAO ESPECIFICADO': 'W12.9',
    'QUEDA EM OU DE UM ANDAIME - RESIDENCIA': 'W12.0',
    'QUEDA ENQUANTO ESTAVA SENDO CARREGADO OU APOIADO POR OUTRA(S) PESSOA(S)': 'W04',
    'QUEDA ENVOLVENDO EQUIPAMENTO DE PLAYGROUND': 'W09',
    'QUEDA ENVOLVENDO EQUIPAMENTO DE PLAYGROUND - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W09.3',
    'QUEDA ENVOLVENDO EQUIPAMENTO DE PLAYGROUND - OUTROS LOCAIS ESPECIFICADOS': 'W09.8',
    'QUEDA ENVOLVENDO EQUIPAMENTO DE PLAYGROUND - RESIDENCIA': 'W09.0',
    'QUEDA ENVOLVENDO PATINS DE RODAS OU PARA GELO ESQUI OU PRANCHAS DE RODAS': 'W02',
    'QUEDA ENVOLVENDO UMA CADEIRA DE RODAS': 'W05',
    'QUEDA ENVOLVENDO UMA CADEIRA DE RODAS - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W05.3',
    'QUEDA ENVOLVENDO UMA CADEIRA DE RODAS - LOCAL NAO ESPECIFICADO': 'W05.9',
    'QUEDA ENVOLVENDO UMA CADEIRA DE RODAS - RUA E ESTRADA': 'W05.4',
    'QUEDA NO MESMO NIVEL ENVOLVENDO GELO E NEVE': 'W00',
    'QUEDA NO MESMO NIVEL POR ESCORREGAO TROPECAO OU PASSOS EM FALSOS (TRASPES)': 'W01',
    'QUEDA SALTO OU EMPURRADO DE UM LUGAR ELEVADO INTENCAO NAO DETERMINADA': 'Y30',
    'QUEDA SEM ESPECIFICACAO': 'W19',
    'QUEDA SEM ESPECIFICACAO - AREA PARA A PRATICA DE ESPORTES E ATLETISMO': 'W19.3',
    'QUEDA SEM ESPECIFICACAO - AREAS DE COMERCIO E DE SERVICOS': 'W19.5',
    'QUEDA SEM ESPECIFICACAO - AREAS INDUSTRIAIS E EM CONSTRUCAO': 'W19.6',
    'QUEDA SEM ESPECIFICACAO - HABITACAO COLETIVA': 'W19.1',
    'QUEDA SEM ESPECIFICACAO - LOCAL NAO ESPECIFICADO': 'W19.9',
    'QUEDA SEM ESPECIFICACAO - OUTROS LOCAIS ESPECIFICADOS': 'W19.8',
    'QUEDA SEM ESPECIFICACAO - RESIDENCIA': 'W19.0',
    'QUEDA SEM ESPECIFICACAO - RUA E ESTRADA': 'W19.4',
    'QUEIMADURA DA BOCA E DA FARINGE': 'T28.0',
    'QUEIMADURA DA CABECA E DO PESCOCO, GRAU NAO ESPECIFICADO': 'T20.0',
    'QUEIMADURA DA CORNEA E DO SACO CONJUNTIVAL': 'T26.1',
    'QUEIMADURA DA LARINGE E TRAQUEIA': 'T27.0',
    'QUEIMADURA DA PALPEBRA E DA REGIAO PERIOCULAR': 'T26.0',
    'QUEIMADURA DE OUTRAS PARTES DO OLHO E ANEXOS': 'T26.3',
    'QUEIMADURA DE PRIMEIRO GRAU DA CABECA E DO PESCOCO': 'T20.1',
    'QUEIMADURA DE PRIMEIRO GRAU DO OMBRO E DO MEMBRO SUPERIOR, EXCETO PUNHO E MAO': 'T22.1',
    'QUEIMADURA DE PRIMEIRO GRAU DO PUNHO E DA MAO': 'T23.1',
    'QUEIMADURA DE PRIMEIRO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.1',
    'QUEIMADURA DE PRIMEIRO GRAU DO TORNOZELO E DO PE': 'T25.1',
    'QUEIMADURA DE PRIMEIRO GRAU DO TRONCO': 'T21.1',
    'QUEIMADURA DE PRIMEIRO GRAU, PARTE DO CORPO NAO ESPECIFICADA': 'T30.1',
    'QUEIMADURA DE SEGUNDO GRAU DA CABECA E DO PESCOCO': 'T20.2',
    'QUEIMADURA DE SEGUNDO GRAU DO OMBRO E DO MEMBRO SUPERIOR, EXCETO PUNHO E MAO': 'T22.2',
    'QUEIMADURA DE SEGUNDO GRAU DO PUNHO E DA MAO': 'T23.2',
    'QUEIMADURA DE SEGUNDO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.2',
    'QUEIMADURA DE SEGUNDO GRAU DO TORNOZELO E DO PE': 'T25.2',
    'QUEIMADURA DE SEGUNDO GRAU DO TRONCO': 'T21.2',
    'QUEIMADURA DE SEGUNDO GRAU, PARTE DO CORPO NAO ESPECIFICADA': 'T30.2',
    'QUEIMADURA DE TERCEIRO GRAU DO OMBRO E DO MEMBRO SUPERIOR, EXCETO PUNHO E MAO': 'T22.3',
    'QUEIMADURA DE TERCEIRO GRAU DO PUNHO E DA MAO': 'T23.3',
    'QUEIMADURA DE TERCEIRO GRAU DO QUADRIL E DO MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE': 'T24.3',
    'QUEIMADURA DE TERCEIRO GRAU DO TORNOZELO E DO PE': 'T25.3',
    'QUEIMADURA DE TERCEIRO GRAU DO TRONCO': 'T21.3',
    'QUEIMADURA DE TERCEIRO GRAU, PARTE DO CORPO NAO ESPECIFICADA': 'T30.3',
    'QUEIMADURA DO OLHO E ANEXOS, PARTE NAO ESPECIFICADA': 'T26.4',
    'QUEIMADURA DO OMBRO E DO MEMBRO SUPERIOR, EXCETO PUNHO E MAO, GRAU NAO ESPECIFICADO': 'T22.0',
    'QUEIMADURA DO PUNHO E DA MAO, GRAU NAO ESPECIFICADO': 'T23.0',
    'QUEIMADURA DO QUADRIL E MEMBRO INFERIOR, EXCETO TORNOZELO E DO PE, GRAU NAO ESPECIFICADO': 'T24.0',
    'QUEIMADURA DO TORNOZELO E DO PE, GRAU NAO ESPECIFICADO': 'T25.0',
    'QUEIMADURA DO TRATO RESPIRATORIO, PARTE NAO ESPECIFICADA': 'T27.3',
    'QUEIMADURA DO TRONCO, GRAU NAO ESPECIFICADO': 'T21.0',
    'QUEIMADURA E CORROSAO DA CABECA E PESCOCO': 'T20',
    'QUEIMADURA E CORROSAO DE OUTROS ORGAOS INTERNOS': 'T28',
    'QUEIMADURA E CORROSAO DO OMBRO E MEMBRO SUPERIOR EXCETO PUNHO E MAO': 'T22',
    'QUEIMADURA E CORROSAO DO PUNHO E DA MAO': 'T23',
    'QUEIMADURA E CORROSAO DO QUADRIL E MEMBRO INFERIOR EXCETO TORNOZELO E DO PE': 'T24',
    'QUEIMADURA E CORROSAO DO TORNOZELO E DO PE': 'T25',
    'QUEIMADURA E CORROSAO DO TRONCO': 'T21',
    'QUEIMADURA E CORROSAO LIMITADAS AO OLHO E SEUS ANEXOS': 'T26',
    'QUEIMADURA E CORROSAO PARTE NAO ESPECIFICADA DO CORPO': 'T30',
    'QUEIMADURA SOLAR': 'L55',
    'QUEIMADURA SOLAR DE PRIMEIRO GRAU': 'L55.0',
    'QUEIMADURA SOLAR DE SEGUNDO GRAU': 'L55.1',
    'QUEIMADURA SOLAR DE TERCEIRO GRAU': 'L55.2',
    'QUEIMADURA SOLAR, NAO ESPECIFICADA': 'L55.9',
    'QUEIMADURA, PARTE DO CORPO NAO ESPECIFICADA, GRAU NAO ESPECIFICADO': 'T30.0',
    'QUEIMADURAS CLASSIFICADAS SEGUNDO A EXTENSAO DA SUPERFICIE CORPORAL ATINGIDA': 'T31',
    'QUEIMADURAS E CORROSOES DE MULTIPLAS REGIOES DO CORPO': 'T29',
    'QUEIMADURAS ENVOLVENDO DE 10 - 19% DA SUPERFICIE CORPORAL': 'T31.1',
    'QUEIMADURAS ENVOLVENDO MENOS DE 10% DA SUPERFICIE CORPORAL': 'T31.0',
    'QUEIMADURAS MULTIPLAS, GRAU NAO ESPECIFICADO': 'T29.0',
    'QUEIMADURAS MULTIPLAS, MENCIONANDO AO MENOS UMA QUEIMADURA DE TERCEIRO GRAU': 'T29.3',
    'QUEIMADURAS MULTIPLAS, SEM MENCIONAR QUEIMADURA(S) ULTRAPASSANDO O PRIMEIRO GRAU': 'T29.1',
    'QUEIMADURAS MULTIPLAS, SEM MENCIONAR QUEIMADURA(S) ULTRAPASSANDO O SEGUNDO GRAU': 'T29.2',
    'RADICULOPATIA': 'M54.1',
    'RADIODERMATITE, NAO ESPECIFICADA': 'L58.9',
    'RAIVA NAO ESPECIFICADA': 'A82.9',
    'RAIZ DENTARIA RETIDA': 'K08.3',
    'RASTREAMENTO (SCREENING) PRE-NATAL': 'Z36',
    'REABILITACAO DE ALCOOLATRA': 'Z50.2',
    'REABILITACAO DE TOXICODEPENDENTES': 'Z50.3',
    'REABSORCAO PATOLOGICA DOS DENTES': 'K03.3',
    'REACAO AGUDA AO STRESS': 'F43.0',
    'REACAO DE HIPERSENSIBILIDADE DAS VIAS AEREAS SUPERIORES DE LOCALIZACAO NAO ESPECIFICADA': 'J39.3',
    'REACAO NAO ESPECIFICADA A UM STRESS GRAVE': 'F43.9',
    'REACOES E INTOXICACOES DEVIDAS A DROGAS ADMINISTRADAS AO FETO E AO RECEM-NASCIDO': 'P93',
    'REFLEXOS ANORMAIS': 'R29.2',
    'REGIME E HABITOS ALIMENTARES INADEQUADOS': 'Z72.4',
    'REGURGITACAO E RUMINACAO NO RECEM-NASCIDO': 'P92.1',
    'RESPIRACAO OFEGANTE': 'R06.2',
    'RESPIRACAO PELA BOCA': 'R06.5',
    'RESPIRACAO PERIODICA': 'R06.3',
    'RESPOSTA FOTOALERGICA A DROGAS': 'L56.1',
    'RESULTADOS ANORMAIS DE ESTUDOS DE FUNCAO': 'R94',
    'RESULTADOS ANORMAIS DE ESTUDOS DE FUNCAO DE OUTROS ORGAOS, APARELHOS E SISTEMAS': 'R94.8',
    'RESULTADOS ANORMAIS DE ESTUDOS DE FUNCAO HEPATICA': 'R94.5',
    'RESULTADOS ANORMAIS DE ESTUDOS DE FUNCAO RENAL': 'R94.4',
    'RESULTADOS ANORMAIS DE EXAMES PARA DIAGNOSTICO POR IMAGEM DO SISTEMA NERVOSO CENTRAL': 'R90',
    'RETARDO MENTAL GRAVE - OUTROS COMPROMETIMENTOS DO COMPORTAMENTO': 'F72.8',
    'RETARDO MENTAL LEVE': 'F70',
    'RETENCAO DE DISPOSITIVO INTRA-UTERINO CONTRACEPTIVO (DIU) NA GRAVIDEZ': 'O26.3',
    'RETENCAO URINARIA': 'R33',
    'RETINOPATIA DIABETICA': 'H36.0',
    'RETINOPATIAS DE FUNDO E ALTERACOES VASCULARES DA RETINA': 'H35.0',
    'RETOCELE': 'N81.6',
    'RETOSSIGMOIDITE ULCERATIVA (CRONICA)': 'K51.3',
    'REUMATISMO NAO ESPECIFICADO': 'M79.0',
    'RICKETTSIOSE NAO ESPECIFICADA': 'A79.9',
    'RIGIDEZ ABDOMINAL': 'R19.3',
    'RIGIDEZ ARTICULAR NAO CLASSIFICADA EM OUTRA PARTE': 'M25.6',
    'RIM POLICISTICO, AUTOSSOMICO DOMINANTE': 'Q61.2',
    'RIM TRANSPLANTADO': 'Z94.0',
    'RINITE ALERGICA DEVIDA A POLEN': 'J30.1',
    'RINITE ALERGICA E VASOMOTORA': 'J30',
    'RINITE ALERGICA NAO ESPECIFICADA': 'J30.4',
    'RINITE CRONICA': 'J31.0',
    'RINITE NASOFARINGITE E FARINGITE CRONICAS': 'J31',
    'RINITE VASOMOTORA': 'J30.0',
    'RISCOS NAO ESPECIFICADOS A RESPIRACAO': 'W84',
    'RITMO DE TRABALHO PENOSO': 'Z56.3',
    'ROSACEA': 'L71',
    'ROSACEA, NAO ESPECIFICADA': 'L71.9',
    'RUBEOLA': 'B06',
    'RUBEOLA SEM COMPLICACAO': 'B06.9',
    'RUBOR': 'R23.2',
    'RUPTURA ATUAL DA CARTILAGEM DA ARTICULACAO DO JOELHO': 'S83.3',
    'RUPTURA DE CISTO POPLITEO': 'M66.0',
    'RUPTURA DE LIGAMENTOS AO NIVEL DO TORNOZELO E DO PE': 'S93.2',
    'RUPTURA DO MENISCO, ATUAL': 'S83.2',
    'RUPTURA ESPONTANEA DE OUTROS TENDOES': 'M66.4',
    'RUPTURA ESPONTANEA DE SINOVIA E DE TENDAO': 'M66',
    'RUPTURA ESPONTANEA DE TENDOES NAO ESPECIFICADOS': 'M66.5',
    'RUPTURA TRAUMATICA DE LIGAMENTO(S) DO PUNHO E DO CARPO': 'S63.3',
    'RUPTURA TRAUMATICA DO LIGAMENTO COLATERAL DO RADIO': 'S53.2',
    'RUPTURA TRAUMATICA DO TIMPANO': 'S09.2',
    'SACROILEITE NAO CLASSIFICADA EM OUTRA PARTE': 'M46.1',
    'SALPINGITE E OOFORITE': 'N70',
    'SALPINGITE E OOFORITE AGUDAS': 'N70.0',
    'SALPINGITE E OOFORITE NAO ESPECIFICADAS': 'N70.9',
    'SANGRAMENTO ABUNDANTE NA PRE-MENOPAUSA': 'N92.4',
    'SANGRAMENTO ANORMAL DO UTERO OU DA VAGINA, NAO ESPECIFICADO': 'N93.9',
    'SANGRAMENTO DA OVULACAO': 'N92.3',
    'SANGRAMENTO POS-MENOPAUSA': 'N95.0',
    'SANGRAMENTOS POS-COITO OU DE CONTATO': 'N93.0',
    'SARAMPO': 'B05',
    'SARAMPO COM OUTRAS COMPLICACOES': 'B05.8',
    'SARAMPO COMPLICADO POR OTITE MEDIA': 'B05.3',
    'SARAMPO SEM COMPLICACAO': 'B05.9',
    'SARCOIDOSE': 'D86',
    'SARCOMA DE KAPOSI DE TECIDOS MOLES': 'C46.1',
    'SEBORREIA DO COURO CABELUDO': 'L21.0',
    'SECRECAO URETRAL': 'R36',
    'SEGUIMENTO ENVOLVENDO CIRURGIA PLASTICA DE MAMA': 'Z42.1',
    'SEGUIMENTO ENVOLVENDO REMOCAO DE PLACA DE FRATURA E OUTROS DISPOSITIVOS DE FIXACAO INTERNA': 'Z47.0',
    'SEGUIMENTO ORTOPEDICO NAO ESPECIFICADO': 'Z47.9',
    'SEIO, FISTULA E CISTO PRE-AURICULAR': 'Q18.1',
    'SENILIDADE': 'R54',
    'SEPTICEMIA NAO ESPECIFICADA': 'A41.9',
    'SEPTICEMIA POR STAPHYLOCOCCUS AUREUS': 'A41.0',
    'SEPTICEMIA POR STREPTOCOCCUS PNEUMONIA': 'A40.3',
    'SEQUELAS DE ACIDENTE VASCULAR CEREBRAL NAO ESPECIFICADO COMO HEMORRAGICO OU ISQUEMICO': 'I69.4',
    'SEQUELAS DE ACIDENTES DE TRANSPORTE': 'Y85',
    'SEQUELAS DE ACIDENTES DURANTE A PRESTACAO DE CUIDADO MEDICO E CIRURGICO': 'Y88.1',
    'SEQUELAS DE DESNUTRICAO E DE OUTRAS DEFICIENCIAS NUTRICIONAIS': 'E64',
    'SEQUELAS DE DOENCA INFECCIOSA OU PARASITARIA NAO ESPECIFICADA': 'B94.9',
    'SEQUELAS DE DOENCAS INFLAMATORIAS DO SISTEMA NERVOSO CENTRAL': 'G09',
    'SEQUELAS DE FERIMENTO DA CABECA': 'T90.1',
    'SEQUELAS DE FERIMENTO DO MEMBRO INFERIOR': 'T93.0',
    'SEQUELAS DE FERIMENTO DO MEMBRO SUPERIOR': 'T92.0',
    'SEQUELAS DE FRATURA AO NIVEL DO PUNHO E DA MAO': 'T92.2',
    'SEQUELAS DE FRATURA DE CRANIO E DE OSSOS DA FACE': 'T90.2',
    'SEQUELAS DE FRATURA DO BRACO': 'T92.1',
    'SEQUELAS DE FRATURA DO FEMUR': 'T93.1',
    'SEQUELAS DE HEMORRAGIA INTRACEREBRAL': 'I69.1',
    'SEQUELAS DE HEMORRAGIA SUBARACNOIDEA': 'I69.0',
    'SEQUELAS DE HIPERALIMENTACAO': 'E68',
    'SEQUELAS DE INFARTO CEREBRAL': 'I69.3',
    'SEQUELAS DE INTOXICACAO POR DROGAS, MEDICAMENTOS E SUBSTANCIAS BIOLOGICAS': 'T96',
    'SEQUELAS DE LUXACAO, ENTORSE E DISTENSAO DO MEMBRO INFERIOR': 'T93.3',
    'SEQUELAS DE LUXACAO, ENTORSE E DISTENSAO DO MEMBRO SUPERIOR': 'T92.3',
    'SEQUELAS DE OUTRAS DOENCAS CEREBROVASCULARES E DAS NAO ESPECIFICADAS': 'I69.8',
    'SEQUELAS DE OUTRAS DOENCAS INFECCIOSAS E PARASITARIAS ESPECIFICADAS': 'B94.8',
    'SEQUELAS DE OUTRAS FRATURAS DO MEMBRO INFERIOR': 'T93.2',
    'SEQUELAS DE OUTRAS HEMORRAGIAS INTRACRANIANAS NAO TRAUMATICAS': 'I69.2',
    'SEQUELAS DE OUTROS ACIDENTES': 'Y86',
    'SEQUELAS DE OUTROS TRAUMATISMOS ESPECIFICADOS DO MEMBRO SUPERIOR': 'T92.8',
    'SEQUELAS DE QUEIMADURA, CORROSAO E GELADURA DE LOCAL NAO ESPECIFICADO': 'T95.9',
    'SEQUELAS DE QUEIMADURAS CORROSOES E GELADURAS': 'T95',
    'SEQUELAS DE TRAUMATISMO DA CABECA': 'T90',
    'SEQUELAS DE TRAUMATISMO DE MUSCULO E TENDAO DO MEMBRO INFERIOR': 'T93.5',
    'SEQUELAS DE TRAUMATISMO DE NERVO DO MEMBRO INFERIOR': 'T93.4',
    'SEQUELAS DE TRAUMATISMO NAO ESPECIFICADO DA CABECA': 'T90.9',
    'SEQUELAS DE TRAUMATISMO NAO ESPECIFICADO DO MEMBRO INFERIOR': 'T93.9',
    'SEQUELAS DE TRAUMATISMO SUPERFICIAL DA CABECA': 'T90.0',
    'SEQUELAS DE TRAUMATISMO SUPERFICIAL E FERIMENTO DO PESCOCO E DO TRONCO': 'T91.0',
    'SEQUELAS DE TRAUMATISMOS DO MEMBRO INFERIOR': 'T93',
    'SEQUELAS DE TRAUMATISMOS DO MEMBRO SUPERIOR': 'T92',
    'SEQUELAS DE TRAUMATISMOS ENVOLVENDO REGIOES MULTIPLAS DO CORPO': 'T94.0',
    'SEQUELAS DE TRAUMATISMOS NAO ESPECIFICADOS POR REGIOES DO CORPO': 'T94.1',
    'SEQUELAS DE TUBERCULOSE': 'B90',
    'SEQUELAS DE TUBERCULOSE DAS VIAS RESPIRATORIAS E DE ORGAOS NAO ESPECIFICADOS': 'B90.9',
    'SEQUELAS DOS EFEITOS DA PENETRACAO DE CORPO ESTRANHO ATRAVES DE ORIFICIO NATURAL': 'T98.0',
    'SEQUESTRO PULMONAR': 'Q33.2',
    'SESSAO DE QUIMIOTERAPIA POR NEOPLASIA': 'Z51.1',
    'SEVICIAS FISICAS': 'T74.1',
    'SEXO INDETERMINADO, NAO ESPECIFICADO': 'Q56.4',
    'SHIGUELOSE': 'A03',
    'SHIGUELOSE DEVIDA A SHIGELLA DYSENTERIAE': 'A03.0',
    'SHIGUELOSE DEVIDA A SHIGELLA SONNEI': 'A03.3',
    'SHIGUELOSE NAO ESPECIFICADA': 'A03.9',
    'SIALADENITE': 'K11.2',
    'SIALOLITIASE': 'K11.5',
    'SIFILIS ANAL PRIMARIA': 'A51.1',
    'SIFILIS CARDIOVASCULAR': 'I98.0',
    'SIFILIS CONGENITA': 'A50',
    'SIFILIS CONGENITA NAO ESPECIFICADA': 'A50.9',
    'SIFILIS CONGENITA PRECOCE NAO ESPECIFICADA': 'A50.2',
    'SIFILIS CONGENITA PRECOCE SINTOMATICA': 'A50.0',
    'SIFILIS CONGENITA PRECOCE, FORMA LATENTE': 'A50.1',
    'SIFILIS CONGENITA TARDIA LATENTE': 'A50.6',
    'SIFILIS CONGENITA TARDIA NAO ESPECIFICADA': 'A50.7',
    'SIFILIS GENITAL PRIMARIA': 'A51.0',
    'SIFILIS LATENTE, NAO ESPECIFICADA SE RECENTE OU TARDIA': 'A53.0',
    'SIFILIS NAO ESPECIFICADA': 'A53.9',
    'SIFILIS PELVICA FEMININA': 'N74.2',
    'SIFILIS PRECOCE': 'A51',
    'SIFILIS PRECOCE LATENTE': 'A51.5',
    'SIFILIS PRECOCE NAO ESPECIFICADA': 'A51.9',
    'SIFILIS PRIMARIA DE OUTRAS LOCALIZACOES': 'A51.2',
    'SIFILIS SECUNDARIA DA PELE E DAS MUCOSAS': 'A51.3',
    'SIFILIS TARDIA': 'A52',
    'SIFILIS TARDIA LATENTE': 'A52.8',
    'SIFILIS TARDIA NAO ESPECIFICADA': 'A52.9',
    'SIFILIS TARDIA RENAL': 'N29.0',
    'SINCOPE DEVIDA AO CALOR': 'T67.1',
    'SINCOPE E COLAPSO': 'R55',
    'SINDROME AMNESICA ORGANICA NAO INDUZIDA PELO ALCOOL OU POR OUTRAS SUBSTANCIAS PSICOATIVAS': 'F04',
    'SINDROME CERVICOBRAQUIAL': 'M53.1',
    'SINDROME CERVICOCRANIANA': 'M53.0',
    'SINDROME DA ARTERIA VERTEBRO-BASILAR': 'G45.0',
    'SINDROME DA CAUDA EQUINA': 'G83.4',
    'SINDROME DA CRIGLER-NAJJAR': 'E80.5',
    'SINDROME DA DEFICIENCIA CONGENITA DE IODO DO TIPO MIXEDEMATOSO': 'E00.1',
    'SINDROME DA ERUPCAO DENTARIA': 'K00.7',
    'SINDROME DA FADIGA POS-VIRAL': 'G93.3',
    'SINDROME DA HIPOTENSAO MATERNA': 'O26.5',
    'SINDROME DA JUNCAO CONDROCOSTAL [TIETZE]': 'M94.0',
    'SINDROME DA LACERACAO HEMORRAGICA GASTROESOFAGICA': 'K22.6',
    'SINDROME DA PELE ESCALDADA ESTAFILOCOCICA DO RECEM-NASCIDO': 'L00',
    'SINDROME DE ARNOLD-CHIARI': 'Q07.0',
    'SINDROME DE CLUSTER-HEADACHE': 'G44.0',
    'SINDROME DE COLISAO DO OMBRO': 'M75.4',
    'SINDROME DE CUSHING NAO ESPECIFICADA': 'E24.9',
    'SINDROME DE DOWN': 'Q90',
    'SINDROME DE DOWN NAO ESPECIFICADA': 'Q90.9',
    'SINDROME DE DRESSLER': 'I24.1',
    'SINDROME DE GILBERT': 'E80.4',
    'SINDROME DE GUILLAIN-BARRE': 'G61.0',
    'SINDROME DE HORNER': 'G90.2',
    'SINDROME DE IMOBILIDADE (PARAPLEGICA)': 'M62.3',
    'SINDROME DE INFECCAO AGUDA PELO HIV': 'B23.0',
    'SINDROME DE LESAO PELO FRIO': 'P80.0',
    'SINDROME DE LINFONODOS MUCOCUTANEOS [KAWASAKI]': 'M30.3',
    'SINDROME DE PRE-EXCITACAO': 'I45.6',
    'SINDROME DE RAYNAUD': 'I73.0',
    'SINDROME DE TENSAO PRE-MENSTRUAL': 'N94.3',
    'SINDROME DE TURNER': 'Q96',
    'SINDROME DO CHOQUE TOXICO': 'A48.3',
    'SINDROME DO COLON IRRITAVEL': 'K58',
    'SINDROME DO COLON IRRITAVEL COM DIARREIA': 'K58.0',
    'SINDROME DO COLON IRRITAVEL SEM DIARREIA': 'K58.9',
    'SINDROME DO DESCONFORTO RESPIRATORIO DO ADULTO': 'J80',
    'SINDROME DO LINFEDEMA POS-MASTECTOMIA': 'I97.2',
    'SINDROME DO MANGUITO ROTADOR': 'M75.1',
    'SINDROME DO MEMBRO FANTASMA SEM MANIFESTACAO DOLOROSA': 'G54.7',
    'SINDROME DO NO SINUSAL': 'I49.5',
    'SINDROME DO OVARIO POLICISTICO': 'E28.2',
    'SINDROME DO TUNEL DO CARPO': 'G56.0',
    'SINDROME DO TUNEL DO TARSO': 'G57.5',
    'SINDROME HEPATORRENAL': 'K76.7',
    'SINDROME MIELODISPLASICA, NAO ESPECIFICADA': 'D46.9',
    'SINDROME NAO ESPECIFICADA DE MAUS TRATOS': 'T74.9',
    'SINDROME NEFRITICA AGUDA': 'N00',
    'SINDROME NEFRITICA AGUDA - ANORMALIDADE GLOMERULAR MINOR': 'N00.0',
    'SINDROME NEFRITICA AGUDA - GLOMERULONEFRITE MEMBRANOSA DIFUSA': 'N00.2',
    'SINDROME NEFRITICA AGUDA - GLOMERULONEFRITE MESANGIOCAPILAR DIFUSA': 'N00.5',
    'SINDROME NEFRITICA AGUDA - NAO ESPECIFICADA': 'N00.9',
    'SINDROME NEFRITICA AGUDA - OUTRAS': 'N00.8',
    'SINDROME NEFRITICA CRONICA - NAO ESPECIFICADA': 'N03.9',
    'SINDROME NEFRITICA RAPIDAMENTE PROGRESSIVA - NAO ESPECIFICADA': 'N01.9',
    'SINDROME NEFROTICA': 'N04',
    'SINDROME NEFROTICA - ANORMALIDADE GLOMERULAR MINOR': 'N04.0',
    'SINDROME NEFROTICA - NAO ESPECIFICADA': 'N04.9',
    'SINDROME NEFROTICA - OUTRAS': 'N04.8',
    'SINDROME POS-COLECISTECTOMIA': 'K91.5',
    'SINDROME POS-LAMINECTOMIA NAO CLASSIFICADA EM OUTRA PARTE': 'M96.1',
    'SINDROME POS-TRAUMATICA': 'F07.2',
    'SINDROME RESPIRATORIA AGUDA GRAVE [SEVERE ACUTE RESPIRATORY SYNDROME) [SARS], NAO ESPECIFICADA': 'U04.9',
    'SINDROME SECA [SJOGREN]': 'M35.0',
    'SINDROME URETRAL, NAO ESPECIFICADA': 'N34.3',
    'SINDROME VASCULAR CEREBELAR': 'G46.4',
    'SINDROMES DE MAUS TRATOS': 'T74',
    'SINDROMES EPILEPTICAS ESPECIAIS': 'G40.5',
    'SINDROMES MIELODISPLASICAS': 'D46',
    'SINDROMES POS-CIRURGIA GASTRICA': 'K91.1',
    'SINDROMES VASCULARES CEREBRAIS QUE OCORREM EM DOENCAS CEREBROVASCULARES': 'G46',
    'SINDROMES VASCULARES DO TRONCO CEREBRAL': 'G46.3',
    'SINDROMES VERTIGINOSAS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H82',
    'SINOVITE CREPITANTE CRONICA DA MAO E DO PUNHO': 'M70.0',
    'SINOVITE E TENOSSINOVITE': 'M65',
    'SINOVITE E TENOSSINOVITE EM DOENCAS BACTERIANAS CLASSIFICADAS EM OUTRA PARTE': 'M68.0',
    'SINOVITE E TENOSSINOVITE NAO ESPECIFICADAS': 'M65.9',
    'SINOVITE TRANSITORIA': 'M67.3',
    'SINTOMAS DE ABSTINENCIA DO USO DE DROGAS TERAPEUTICAS NO RECEM-NASCIDO': 'P96.2',
    'SINTOMAS E SINAIS RELATIVOS A APARENCIA E AO COMPORTAMENTO': 'R46',
    'SINTOMAS E SINAIS RELATIVOS AO ESTADO EMOCIONAL': 'R45',
    'SINTOMAS NAO ESPECIFICOS PECULIARES A INFANCIA': 'R68.1',
    'SINUSITE AGUDA': 'J01',
    'SINUSITE AGUDA NAO ESPECIFICADA': 'J01.9',
    'SINUSITE BAROTRAUMATICA': 'T70.1',
    'SINUSITE CRONICA': 'J32',
    'SINUSITE CRONICA NAO ESPECIFICADA': 'J32.9',
    'SINUSITE ESFENOIDAL AGUDA': 'J01.3',
    'SINUSITE ETMOIDAL AGUDA': 'J01.2',
    'SINUSITE ETMOIDAL CRONICA': 'J32.2',
    'SINUSITE FRONTAL AGUDA': 'J01.1',
    'SINUSITE FRONTAL CRONICA': 'J32.1',
    'SINUSITE MAXILAR AGUDA': 'J01.0',
    'SINUSITE MAXILAR CRONICA': 'J32.0',
    'SITUS INVERSUS': 'Q89.3',
    'SOLUCO': 'R06.6',
    'SONOLENCIA': 'R40.0',
    'SONOLENCIA ESTUPOR E COMA': 'R40',
    'SOPRO CARDIACO, NAO ESPECIFICADO': 'R01.1',
    'STREPTOCOCCUS PNEUMONIAE, COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B95.3',
    'STRESS NAO CLASSIFICADO EM OUTRA PARTE': 'Z73.3',
    'SUBLUXACAO RECIDIVANTE DA ROTULA': 'M22.1',
    'SUPERVISAO DE CUIDADO DE SAUDE DE OUTRAS CRIANCAS OU RECEM-NASCIDOS SADIOS': 'Z76.2',
    'SUPERVISAO DE GRAVIDEZ COM OUTROS ANTECEDENTES DE PROCRIACAO PROBLEMATICA': 'Z35.2',
    'SUPERVISAO DE GRAVIDEZ DE ALTO RISCO': 'Z35',
    'SUPERVISAO DE GRAVIDEZ NORMAL': 'Z34',
    'SUPERVISAO DE GRAVIDEZ NORMAL, NAO ESPECIFICADA': 'Z34.9',
    'SUPERVISAO DE OUTRA GRAVIDEZ NORMAL': 'Z34.8',
    'SUPERVISAO DE PRIMEIRA GRAVIDEZ NORMAL': 'Z34.0',
    'SUPERVISAO DE PRIMIGESTA IDOSA': 'Z35.5',
    'SUPERVISAO E CUIDADO DE SAUDE DE CRIANCAS ASSISTIDAS': 'Z76.1',
    'SUPERVISAO NAO ESPECIFICADA DE GRAVIDEZ DE ALTO RISCO': 'Z35.9',
    'SURDO-MUDEZ NAO CLASSIFICADA EM OUTRA PARTE': 'H91.3',
    'SUSPEITA DE GLAUCOMA': 'H40.0',
    'SUTURAS CRANIANAS AMPLAS NO RECEM-NASCIDO': 'P96.3',
    'TALASSEMIA': 'D56',
    'TALASSEMIA NAO ESPECIFICADA': 'D56.9',
    'TAQUICARDIA NAO ESPECIFICADA': 'R00.0',
    'TAQUICARDIA PAROXISTICA': 'I47',
    'TAQUICARDIA PAROXISTICA NAO ESPECIFICADA': 'I47.9',
    'TAQUICARDIA SUPRAVENTRICULAR': 'I47.1',
    'TAQUICARDIA VENTRICULAR': 'I47.2',
    'TELANGIECTASIA HEMORRAGICA HEREDITARIA': 'I78.0',
    'TEMPERATURA INADEQUADA DURANTE APLICACAO LOCAL OU CURATIVO': 'Y63.5',
    'TENDAO DE AQUILES CURTO (ADQUIRIDO)': 'M67.0',
    'TENDENCIA A QUEDA, NAO CLASSIFICADA EM OUTRA PARTE': 'R29.6',
    'TENDINITE AQUILEANA': 'M76.6',
    'TENDINITE BICEPITAL': 'M75.2',
    'TENDINITE CALCIFICADA': 'M65.2',
    'TENDINITE CALCIFICANTE DO OMBRO': 'M75.3',
    'TENDINITE DO PERONEO': 'M76.7',
    'TENDINITE DO PSOAS': 'M76.1',
    'TENDINITE GLUTEA': 'M76.0',
    'TENDINITE PATELAR': 'M76.5',
    'TENESMO VESICAL': 'R30.1',
    'TENOSSINOVITE ESTILOIDE RADIAL [DE QUERVAIN]': 'M65.4',
    'TESTE DE TOLERANCIA A GLICOSE ANORMAL': 'R73.0',
    'TESTICULO ECTOPICO': 'Q53.0',
    'TESTICULO NAO-DESCIDO': 'Q53',
    'TESTICULO NAO-DESCIDO, NAO ESPECIFICADO': 'Q53.9',
    'TIFO EPIDEMICO TRANSMITIDO POR PIOLHOS DEVIDO A RICKETTSIA PROWAZEKII': 'A75.0',
    'TIFO EXANTEMATICO': 'A75',
    'TINEA CRURIS': 'B35.6',
    'TINHA DA BARBA E DO COURO CABELUDO': 'B35.0',
    'TINHA DA MAO': 'B35.2',
    'TINHA DAS UNHAS': 'B35.1',
    'TINHA DO CORPO': 'B35.4',
    'TINHA DOS PES': 'B35.3',
    'TINHA IMBRICADA': 'B35.5',
    'TINHA NEGRA': 'B36.1',
    'TINNITUS': 'H93.1',
    'TIQUE NAO ESPECIFICADO': 'F95.9',
    'TIQUES': 'F95',
    'TIQUES VOCAIS E MOTORES MULTIPLOS COMBINADOS [DOENCA DE GILLES DE LA TOURETTE]': 'F95.2',
    'TIREOIDITE': 'E06',
    'TIREOIDITE AGUDA': 'E06.0',
    'TIREOIDITE AUTO-IMUNE': 'E06.3',
    'TIREOIDITE SUBAGUDA': 'E06.1',
    'TIREOTOXICOSE (HIPERTIREOIDISMO)': 'E05',
    'TIREOTOXICOSE COM BOCIO DIFUSO': 'E05.0',
    'TIREOTOXICOSE NAO ESPECIFICADA': 'E05.9',
    'TONTURA E INSTABILIDADE': 'R42',
    'TORCAO DO TESTICULO': 'N44',
    'TORCICOLO': 'M43.6',
    'TORCICOLO ESPASMODICO': 'G24.3',
    'TOSSE': 'R05',
    'TOXOPLASMOSE': 'B58',
    'TOXOPLASMOSE CONGENITA': 'P37.1',
    'TOXOPLASMOSE NAO ESPECIFICADA': 'B58.9',
    'TRABALHO DE PARTO PRE-TERMO SEM PARTO': 'O60.0',
    'TRACOMA NAO ESPECIFICADO': 'A71.9',
    'TRANSFUSAO DE SANGUE, SEM DIAGNOSTICO REGISTRADO': 'Z51.3',
    'TRANSFUSAO OU INFUSAO DE MEDICAMENTO OU SUBSTANCIA BIOLOGICA CONTAMINADOS': 'Y64.0',
    'TRANSTORNO ADRENOGENITAL NAO ESPECIFICADO': 'E25.9',
    'TRANSTORNO AFETIVO BIPOLAR': 'F31',
    'TRANSTORNO AFETIVO BIPOLAR NAO ESPECIFICADO': 'F31.9',
    'TRANSTORNO AFETIVO BIPOLAR, ATUALMENTE EM REMISSAO': 'F31.7',
    'TRANSTORNO AFETIVO BIPOLAR, EPISODIO ATUAL DEPRESSIVO GRAVE SEM SINTOMAS PSICOTICOS': 'F31.4',
    'TRANSTORNO AFETIVO BIPOLAR, EPISODIO ATUAL DEPRESSIVO LEVE OU MODERADO': 'F31.3',
    'TRANSTORNO AFETIVO BIPOLAR, EPISODIO ATUAL MANIACO COM SINTOMAS PSICOTICOS': 'F31.2',
    'TRANSTORNO AFETIVO BIPOLAR, EPISODIO ATUAL MANIACO SEM SINTOMAS PSICOTICOS': 'F31.1',
    'TRANSTORNO AFETIVO BIPOLAR, EPISODIO ATUAL MISTO': 'F31.6',
    'TRANSTORNO ANSIOSO NAO ESPECIFICADO': 'F41.9',
    'TRANSTORNO ARTICULAR NAO ESPECIFICADO': 'M25.9',
    'TRANSTORNO COGNITIVO LEVE': 'F06.7',
    'TRANSTORNO DA ELIMINACAO TRANSEPIDERMICA, NAO ESPECIFICADO': 'L87.9',
    'TRANSTORNO DA ESCLEROTICA E DA CORNEA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H19',
    'TRANSTORNO DA GENGIVA E DO REBORDO ALVEOLAR SEM DENTES SEM OUTRA ESPECIFICACAO': 'K06.9',
    'TRANSTORNO DA MAMA NAO ESPECIFICADO': 'N64.9',
    'TRANSTORNO DA PERSONALIDADE E DO COMPORTAMENTO DO ADULTO, NAO ESPECIFICADO': 'F69',
    'TRANSTORNO DA PIGMENTACAO, NAO ESPECIFICADO': 'L81.9',
    'TRANSTORNO DA RETINA NAO ESPECIFICADO': 'H35.9',
    'TRANSTORNO DE ALIMENTACAO NA INFANCIA': 'F98.2',
    'TRANSTORNO DE ALIMENTACAO NAO ESPECIFICADO': 'F50.9',
    'TRANSTORNO DE CONDUTA NAO ESPECIFICADO': 'F91.9',
    'TRANSTORNO DE LABILIDADE EMOCIONAL [ASTENICO] ORGANICO': 'F06.6',
    'TRANSTORNO DE PANICO [ANSIEDADE PAROXISTICA EPISODICA]': 'F41.0',
    'TRANSTORNO DE PERSONALIDADE COM INSTABILIDADE EMOCIONAL': 'F60.3',
    'TRANSTORNO DE SOMATIZACAO': 'F45.0',
    'TRANSTORNO DELIRANTE': 'F22.0',
    'TRANSTORNO DELIRANTE ORGANICO [TIPO ESQUIZOFRENICO]': 'F06.2',
    'TRANSTORNO DELIRANTE PERSISTENTE NAO ESPECIFICADO': 'F22.9',
    'TRANSTORNO DEPRESSIVO RECORRENTE': 'F33',
    'TRANSTORNO DEPRESSIVO RECORRENTE SEM ESPECIFICACAO': 'F33.9',
    'TRANSTORNO DEPRESSIVO RECORRENTE, EPISODIO ATUAL GRAVE COM SINTOMAS PSICOTICOS': 'F33.3',
    'TRANSTORNO DEPRESSIVO RECORRENTE, EPISODIO ATUAL GRAVE SEM SINTOMAS PSICOTICOS': 'F33.2',
    'TRANSTORNO DEPRESSIVO RECORRENTE, EPISODIO ATUAL LEVE': 'F33.0',
    'TRANSTORNO DEPRESSIVO RECORRENTE, EPISODIO ATUAL MODERADO': 'F33.1',
    'TRANSTORNO DISSOCIATIVO MISTO [DE CONVERSAO]': 'F44.7',
    'TRANSTORNO DISSOCIATIVO ORGANICO': 'F06.5',
    'TRANSTORNO DISSOCIATIVO [DE CONVERSAO] NAO ESPECIFICADO': 'F44.9',
    'TRANSTORNO DO APARELHO DIGESTIVO POS PROCEDIMENTO': 'K91.9',
    'TRANSTORNO DO CICLO VIGILIA-SONO DEVIDO A FATORES NAO-ORGANICOS': 'F51.2',
    'TRANSTORNO DO DESENVOLVIMENTO PSICOLOGICO NAO ESPECIFICADO': 'F89',
    'TRANSTORNO DO DISCO CERVICAL COM MIELOPATIA': 'M50.0',
    'TRANSTORNO DO DISCO CERVICAL COM RADICULOPATIA': 'M50.1',
    'TRANSTORNO DO HUMOR [AFETIVO] NAO ESPECIFICADO': 'F39',
    'TRANSTORNO DO MENISCO DEVIDO A RUPTURA OU LESAO ANTIGA': 'M23.2',
    'TRANSTORNO DO SONO DEVIDO A FATORES NAO-ORGANICOS NAO ESPECIFICADOS': 'F51.9',
    'TRANSTORNO DOS DENTES E DE SUAS ESTRUTURAS DE SUSTENTACAO, SEM OUTRA ESPECIFICACAO': 'K08.9',
    'TRANSTORNO DOS TECIDOS MOLES NAO ESPECIFICADO': 'M79.9',
    'TRANSTORNO ENDOCRINO NAO ESPECIFICADO': 'E34.9',
    'TRANSTORNO ESPECIFICO DA ARTICULACAO DA FALA': 'F80.0',
    'TRANSTORNO ESPECIFICO DO DESENVOLVIMENTO MOTOR': 'F82',
    'TRANSTORNO ESQUIZOAFETIVO DO TIPO DEPRESSIVO': 'F25.1',
    'TRANSTORNO ESQUIZOAFETIVO NAO ESPECIFICADO': 'F25.9',
    'TRANSTORNO ESQUIZOTIPICO': 'F21',
    'TRANSTORNO FOBICO ANSIOSO DA INFANCIA': 'F93.1',
    'TRANSTORNO FOBICO-ANSIOSO NAO ESPECIFICADO': 'F40.9',
    'TRANSTORNO HEMORRAGICO DEVIDO A ANTICOAGULANTES CIRCULANTES': 'D68.3',
    'TRANSTORNO HIPERCINETICO NAO ESPECIFICADO': 'F90.9',
    'TRANSTORNO HIPOCONDRIACO': 'F45.2',
    'TRANSTORNO INFLAMATORIO DE ORGAO GENITAL MASCULINO, NAO ESPECIFICADO': 'N49.9',
    'TRANSTORNO INFLAMATORIO DO ESCROTO': 'N49.2',
    'TRANSTORNO INTERNO NAO ESPECIFICADO DO JOELHO': 'M23.9',
    'TRANSTORNO INTESTINAL FUNCIONAL, NAO ESPECIFICADO': 'K59.9',
    'TRANSTORNO LIGADO A ANGUSTIA DE SEPARACAO': 'F93.0',
    'TRANSTORNO MENTAL NAO ESPECIFICADO DEVIDO A UMA LESAO E DISFUNCAO CEREBRAL E A UMA DOENCA FISICA': 'F06.9',
    'TRANSTORNO MENTAL NAO ESPECIFICADO EM OUTRA PARTE': 'F99',
    'TRANSTORNO MENTAL ORGANICO OU SINTOMATICO NAO ESPECIFICADO': 'F09',
    'TRANSTORNO MIONEURAL NAO ESPECIFICADO': 'G70.9',
    'TRANSTORNO MISTO ANSIOSO E DEPRESSIVO': 'F41.2',
    'TRANSTORNO MUSCULAR NAO ESPECIFICADO': 'M62.9',
    'TRANSTORNO MUSCULAR PRIMARIO NAO ESPECIFICADO': 'G71.9',
    'TRANSTORNO NAO ESPECIFICADO DA BEXIGA': 'N32.9',
    'TRANSTORNO NAO ESPECIFICADO DA CONJUNTIVA': 'H11.9',
    'TRANSTORNO NAO ESPECIFICADO DA CONTINUIDADE DO OSSO': 'M84.9',
    'TRANSTORNO NAO ESPECIFICADO DA CORNEA': 'H18.9',
    'TRANSTORNO NAO ESPECIFICADO DA MEMBRANA DO TIMPANO': 'H73.9',
    'TRANSTORNO NAO ESPECIFICADO DA MENOPAUSA E DA PERIMENOPAUSA': 'N95.9',
    'TRANSTORNO NAO ESPECIFICADO DA ORBITA': 'H05.9',
    'TRANSTORNO NAO ESPECIFICADO DA PALPEBRA': 'H02.9',
    'TRANSTORNO NAO ESPECIFICADO DA REFRACAO': 'H52.7',
    'TRANSTORNO NAO ESPECIFICADO DA SECRECAO PANCREATICA INTERNA': 'E16.9',
    'TRANSTORNO NAO ESPECIFICADO DA SINOVIA E DO TENDAO': 'M67.9',
    'TRANSTORNO NAO ESPECIFICADO DA TIREOIDE': 'E07.9',
    'TRANSTORNO NAO ESPECIFICADO DA TROMPA DE EUSTAQUIO': 'H69.9',
    'TRANSTORNO NAO ESPECIFICADO DA URETRA': 'N36.9',
    'TRANSTORNO NAO ESPECIFICADO DE DISCO CERVICAL': 'M50.9',
    'TRANSTORNO NAO ESPECIFICADO DE DISCO INTERVERTEBRAL': 'M51.9',
    'TRANSTORNO NAO ESPECIFICADO DO APARELHO LACRIMAL': 'H04.9',
    'TRANSTORNO NAO ESPECIFICADO DO GLOBO OCULAR': 'H44.9',
    'TRANSTORNO NAO ESPECIFICADO DO HUMOR VITREO': 'H43.9',
    'TRANSTORNO NAO ESPECIFICADO DO NERVO FACIAL': 'G51.9',
    'TRANSTORNO NAO ESPECIFICADO DO NERVO TRIGEMEO': 'G50.9',
    'TRANSTORNO NAO ESPECIFICADO DO OLHO E ANEXOS': 'H57.9',
    'TRANSTORNO NAO ESPECIFICADO DO OLHO E ANEXOS POS-PROCEDIMENTO': 'H59.9',
    'TRANSTORNO NAO ESPECIFICADO DO OSSO': 'M89.9',
    'TRANSTORNO NAO ESPECIFICADO DO OUVIDO': 'H93.9',
    'TRANSTORNO NAO ESPECIFICADO DO OUVIDO EXTERNO': 'H61.9',
    'TRANSTORNO NAO ESPECIFICADO DO OUVIDO INTERNO': 'H83.9',
    'TRANSTORNO NAO ESPECIFICADO DO OUVIDO MEDIO E DA MASTOIDE': 'H74.9',
    'TRANSTORNO NAO ESPECIFICADO DO PENIS': 'N48.9',
    'TRANSTORNO NAO ESPECIFICADO DO RIM E DO URETER': 'N28.9',
    'TRANSTORNO NAO ESPECIFICADO DO SISTEMA NERVOSO CENTRAL': 'G96.9',
    'TRANSTORNO NAO ESPECIFICADO DOS ORGAOS GENITAIS MASCULINOS': 'N50.9',
    'TRANSTORNO NAO ESPECIFICADO DOS TECIDOS MOLES RELACIONADOS COM O USO, USO EXCESSIVO E PRESSAO': 'M70.9',
    'TRANSTORNO NAO-INFLAMATORIO DA VAGINA, NAO ESPECIFICADO': 'N89.9',
    'TRANSTORNO NAO-INFLAMATORIO E NAO ESPECIFICADO DA VULVA E DO PERINEO': 'N90.9',
    'TRANSTORNO NEUROTICO NAO ESPECIFICADO': 'F48.9',
    'TRANSTORNO NEUROVEGETATIVO SOMATOFORME': 'F45.3',
    'TRANSTORNO OBSESSIVO-COMPULSIVO': 'F42',
    'TRANSTORNO OBSESSIVO-COMPULSIVO COM PREDOMINANCIA DE COMPORTAMENTOS COMPULSIVOS [RITUAIS OBSESSIVOS]': 'F42.1',
    'TRANSTORNO OBSESSIVO-COMPULSIVO, FORMA MISTA, COM IDEIAS OBSESSIVAS E COMPORTAMENTOS COMPULSIVOS': 'F42.2',
    'TRANSTORNO ORGANICO DA PERSONALIDADE': 'F07.0',
    'TRANSTORNO OSTEOMUSCULAR NAO ESPECIFICADO POS-PROCEDIMENTO': 'M96.9',
    'TRANSTORNO POS-PROCEDIMENTO DO SISTEMA NERVOSO, NAO ESPECIFICADO': 'G97.9',
    'TRANSTORNO POS-PROCEDIMENTO NAO ESPECIFICADO DO APARELHO GENITURINARIO': 'N99.9',
    'TRANSTORNO PSICOTICO AGUDO DE TIPO ESQUIZOFRENICO (SCHIZOPHRENIA-LIKE)': 'F23.2',
    'TRANSTORNO PSICOTICO AGUDO E TRANSITORIO NAO ESPECIFICADO': 'F23.9',
    'TRANSTORNO PSICOTICO AGUDO POLIMORFO, COM SINTOMAS ESQUIZOFRENICOS': 'F23.1',
    'TRANSTORNO PSICOTICO AGUDO POLIMORFO, SEM SINTOMAS ESQUIZOFRENICOS': 'F23.0',
    'TRANSTORNO RESPIRATORIO NAO ESPECIFICADOS': 'J98.9',
    'TRANSTORNO SOMATOFORME INDIFERENCIADO': 'F45.1',
    'TRANSTORNO SOMATOFORME NAO ESPECIFICADO': 'F45.9',
    'TRANSTORNO VASCULAR DO INTESTINO, SEM OUTRA ESPECIFICACAO': 'K55.9',
    'TRANSTORNO VENOSO NAO ESPECIFICADO': 'I87.9',
    'TRANSTORNOS ADRENOGENITAIS': 'E25',
    'TRANSTORNOS DA ACOMODACAO': 'H52.5',
    'TRANSTORNOS DA ALIMENTACAO': 'F50',
    'TRANSTORNOS DA ANSIEDADE ORGANICOS': 'F06.4',
    'TRANSTORNOS DA ARTICULACAO TEMPOROMANDIBULAR': 'K07.6',
    'TRANSTORNOS DA BEXIGA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N33',
    'TRANSTORNOS DA BEXIGA EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N33.8',
    'TRANSTORNOS DA CONJUNTIVA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H13',
    'TRANSTORNOS DA ESCLEROTICA': 'H15',
    'TRANSTORNOS DA FUNCAO VESTIBULAR': 'H81',
    'TRANSTORNOS DA GLANDULA TIREOIDE EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'E35.0',
    'TRANSTORNOS DA JUNCAO MIONEURAL E DOS MUSCULOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'G73',
    'TRANSTORNOS DA MENOPAUSA E DA PERIMENOPAUSA': 'N95',
    'TRANSTORNOS DA ORBITA': 'H05',
    'TRANSTORNOS DA PROSTATA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N51.0',
    'TRANSTORNOS DA PUBERDADE NAO CLASSIFICADOS EM OUTRA PARTE': 'E30',
    'TRANSTORNOS DA REFRACAO E DA ACOMODACAO': 'H52',
    'TRANSTORNOS DA ROTULA (PATELA)': 'M22',
    'TRANSTORNOS DA VESICULA BILIAR E DAS VIAS BILIARES EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'K87.0',
    'TRANSTORNOS DAS ARTERIAS DAS ARTERIOLAS E DOS CAPILARES EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'I79',
    'TRANSTORNOS DAS RAIZES CERVICAIS NAO CLASSIFICADAS EM OUTRA PARTE': 'G54.2',
    'TRANSTORNOS DAS RAIZES E DOS PLEXOS NERVOSOS': 'G54',
    'TRANSTORNOS DAS RAIZES LOMBOSSACRAS NAO CLASSIFICADAS EM OUTRA PARTE': 'G54.4',
    'TRANSTORNOS DAS RAIZES TORACICAS NAO CLASSIFICADAS EM OUTRA PARTE': 'G54.3',
    'TRANSTORNOS DE ADAPTACAO': 'F43.2',
    'TRANSTORNOS DE DISCOS LOMBARES E DE OUTROS DISCOS INTERVERTEBRAIS COM MIELOPATIA': 'M51.0',
    'TRANSTORNOS DE DISCOS LOMBARES E DE OUTROS DISCOS INTERVERTEBRAIS COM RADICULOPATIA': 'M51.1',
    'TRANSTORNOS DE HUMOR (AFETIVOS) PERSISTENTES': 'F34',
    'TRANSTORNOS DE LIGAMENTOS': 'M24.2',
    'TRANSTORNOS DE MUSCULO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M63',
    'TRANSTORNOS DE OUTROS ORGAOS DIGESTIVOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'K93',
    'TRANSTORNOS DE PERSONALIDADE E DO COMPORTAMENTO DEVIDOS A DOENCA A LESAO E A DISFUNCAO CEREBRAL': 'F07',
    'TRANSTORNOS DELIRANTES PERSISTENTES': 'F22',
    'TRANSTORNOS DISSOCIATIVOS (DE CONVERSAO)': 'F44',
    'TRANSTORNOS DISSOCIATIVOS DO MOVIMENTO': 'F44.4',
    'TRANSTORNOS DO APARELHO LACRIMAL': 'H04',
    'TRANSTORNOS DO APARELHO LACRIMAL E DA ORBITA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H06',
    'TRANSTORNOS DO DESENVOLVIMENTO DOS MAXILARES': 'K10.0',
    'TRANSTORNOS DO FIGADO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'K77',
    'TRANSTORNOS DO GLOBO OCULAR': 'H44',
    'TRANSTORNOS DO HUMOR VITREO': 'H43',
    'TRANSTORNOS DO HUMOR VITREO E DO GLOBO OCULAR EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H45',
    'TRANSTORNOS DO HUMOR [AFETIVOS] ORGANICOS': 'F06.3',
    'TRANSTORNOS DO LIQUIDO AMNIOTICO E DAS MEMBRANAS NAO ESPECIFICADOS': 'O41.9',
    'TRANSTORNOS DO NERVO FACIAL': 'G51',
    'TRANSTORNOS DO NERVO TRIGEMEO': 'G50',
    'TRANSTORNOS DO NERVO VAGO': 'G52.2',
    'TRANSTORNOS DO OLHO E ANEXOS POS-PROCEDIMENTO NAO CLASSIFICADOS EM OUTRA PARTE': 'H59',
    'TRANSTORNOS DO OUVIDO E DA APOFISE MASTOIDE POS-PROCEDIMENTOS NAO CLASSIFICADOS EM OUTRA PARTE': 'H95',
    'TRANSTORNOS DO OUVIDO EXTERNO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'H62',
    'TRANSTORNOS DO PLEXO BRAQUIAL': 'G54.0',
    'TRANSTORNOS DO PLEXO LOMBOSSACRAL': 'G54.1',
    'TRANSTORNOS DO SISTEMA NERVOSO AUTONOMO': 'G90',
    'TRANSTORNOS DO TESTICULO E DO EPIDIDIMO EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N51.1',
    'TRANSTORNOS DOS DISCOS CERVICAIS': 'M50',
    'TRANSTORNOS DOS ORGAOS GENITAIS MASCULINOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N51',
    'TRANSTORNOS DOS TECIDOS MOLES EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'M73',
    'TRANSTORNOS ESPECIFICOS DA PERSONALIDADE': 'F60',
    'TRANSTORNOS ESPECIFICOS DO DESENVOLVIMENTO DA FALA E DA LINGUAGEM': 'F80',
    'TRANSTORNOS ESQUIZOAFETIVOS': 'F25',
    'TRANSTORNOS FALCIFORMES': 'D57',
    'TRANSTORNOS FEMUROPATELARES': 'M22.2',
    'TRANSTORNOS FIBROBLASTICOS': 'M72',
    'TRANSTORNOS FOBICO-ANSIOSOS': 'F40',
    'TRANSTORNOS GLOBAIS DO DESENVOLVIMENTO': 'F84',
    'TRANSTORNOS GLOBAIS NAO ESPECIFICADOS DO DESENVOLVIMENTO': 'F84.9',
    'TRANSTORNOS GLOMERULARES EM DOENCAS SISTEMICAS DO TECIDO CONJUNTIVO': 'N08.5',
    'TRANSTORNOS GLOMERULARES NO DIABETES MELLITUS': 'N08.3',
    'TRANSTORNOS HIPERCINETICOS': 'F90',
    'TRANSTORNOS INFLAMATORIOS CRONICOS DA ORBITA': 'H05.1',
    'TRANSTORNOS INFLAMATORIOS DA MAMA': 'N61',
    'TRANSTORNOS INFLAMATORIOS DA PELVE FEMININA EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N74',
    'TRANSTORNOS INFLAMATORIOS DE ORGAOS GENITAIS MASCULINOS NAO CLASSIFICADOS EM OUTRA PARTE': 'N49',
    'TRANSTORNOS INFLAMATORIOS DE OUTROS ORGAOS GENITAIS MASCULINOS ESPECIFICADOS': 'N49.8',
    'TRANSTORNOS INFLAMATORIOS DO CORDAO ESPERMATICO, TUNICA VAGINAL E VASOS DEFERENTES': 'N49.1',
    'TRANSTORNOS INTERNOS DOS JOELHOS': 'M23',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS ASSOCIADOS AO PUERPERIO NAO CLASSIFICADOS EM OUTRA PARTE': 'F53',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DA COCAINA': 'F14',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE ALCOOL': 'F10',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE ALUCINOGENOS': 'F16',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE CANABINOIDES': 'F12',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE FUMO': 'F17',
    'TRANSTORNOS MENTAIS E COMPORTAMENTAIS DEVIDOS AO USO DE OPIACEOS': 'F11',
    'TRANSTORNOS MENTAIS E DOENCAS DO SISTEMA NERVOSO COMPLICANDO A GRAVIDEZ, O PARTO E O PUERPERIO': 'O99.3',
    'TRANSTORNOS MISTOS DE CONDUTA E DAS EMOCOES': 'F92',
    'TRANSTORNOS NAO ESPECIFICADOS DA CARTILAGEM': 'M94.9',
    'TRANSTORNOS NAO ESPECIFICADOS DA FUNCAO VESTIBULAR': 'H81.9',
    'TRANSTORNOS NAO ESPECIFICADOS DA VALVA AORTICA': 'I35.9',
    'TRANSTORNOS NAO ESPECIFICADOS DAS VIAS OPTICAS': 'H47.7',
    'TRANSTORNOS NAO ESPECIFICADOS DO APARELHO URINARIO': 'N39.9',
    'TRANSTORNOS NAO ESPECIFICADOS DOS GLOBULOS BRANCOS': 'D72.9',
    'TRANSTORNOS NAO-INFECCIOSOS DO PAVILHAO DA ORELHA': 'H61.1',
    'TRANSTORNOS NAO-INFECCIOSOS DOS VASOS LINFATICOS E DOS GANGLIOS LINFATICOS, NAO ESPECIFICADOS': 'I89.9',
    'TRANSTORNOS NAO-ORGANICOS DO SONO DEVIDOS A FATORES EMOCIONAIS': 'F51',
    'TRANSTORNOS NAO-REUMATICOS DA VALVA MITRAL': 'I34',
    'TRANSTORNOS NUTRICIONAIS E METABOLICOS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'E90',
    'TRANSTORNOS OSTEOMUSCULARES POS-PROCEDIMENTOS NAO CLASSIFICADOS EM OUTRA PARTE': 'M96',
    'TRANSTORNOS POS-PROCEDIMENTO DO SISTEMA NERVOSO NAO CLASSIFICADOS EM OUTRA PARTE': 'G97',
    'TRANSTORNOS PRIMARIOS DOS MUSCULOS': 'G71',
    'TRANSTORNOS PSICOTICOS AGUDOS E TRANSITORIOS': 'F23',
    'TRANSTORNOS RENAIS TUBULO-INTERSTICIAIS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N16',
    'TRANSTORNOS RENAIS TUBULO-INTERSTICIAIS EM DOENCAS DO TECIDO CONJUNTIVO': 'N16.4',
    'TRANSTORNOS RENAIS TUBULO-INTERSTICIAIS EM DOENCAS METABOLICAS': 'N16.3',
    'TRANSTORNOS RESULTANTES DE FUNCAO RENAL TUBULAR ALTERADA': 'N25',
    'TRANSTORNOS SACROCCIGEOS NAO CLASSIFICADOS EM OUTRA PARTE': 'M53.3',
    'TRANSTORNOS SOMATOFORMES': 'F45',
    'TRANSTORNOS VASCULARES AGUDOS DO INTESTINO': 'K55.0',
    'TRANSTORNOS VASCULARES DO INTESTINO': 'K55',
    'TRANSTORNOS VASCULARES DOS ORGAOS GENITAIS MASCULINOS': 'N50.1',
    'TRAQUEITE AGUDA': 'J04.1',
    'TRAQUEOSTOMIA': 'Z93.0',
    'TRAUMATISMO CEREBRAL DIFUSO': 'S06.2',
    'TRAUMATISMO CEREBRAL FOCAL': 'S06.3',
    'TRAUMATISMO DA CONJUNTIVA E ABRASAO DA CORNEA SEM MENCAO DE CORPO ESTRANHO': 'S05.0',
    'TRAUMATISMO DA URETRA': 'S37.3',
    'TRAUMATISMO DE ESTRUTURAS MULTIPLAS DO JOELHO': 'S83.7',
    'TRAUMATISMO DE MEDULA ESPINHAL, NIVEL NAO ESPECIFICADO': 'T09.3',
    'TRAUMATISMO DE MULTIPLOS MUSCULOS E TENDOES AO NIVEL DO OMBRO E DO BRACO': 'S46.7',
    'TRAUMATISMO DE MULTIPLOS MUSCULOS E TENDOES AO NIVEL DO QUADRIL E DA COXA': 'S76.7',
    'TRAUMATISMO DE MUSCULO E DE TENDAO AO NIVEL DO QUADRIL E DA COXA': 'S76',
    'TRAUMATISMO DE MUSCULO E DE TENDAO AO NIVEL TORACICO': 'S29.0',
    'TRAUMATISMO DE MUSCULO E DE TENDAO NAO ESPECIFICADO AO NIVEL DA PERNA': 'S86.9',
    'TRAUMATISMO DE MUSCULO E TENDAO AO NIVEL DO PUNHO E DA MAO': 'S66',
    'TRAUMATISMO DE MUSCULO E TENDAO NAO ESPECIFICADO AO NIVEL DO OMBRO E DO BRACO': 'S46.9',
    'TRAUMATISMO DE MUSCULO E TENDAO NAO ESPECIFICADO AO NIVEL DO PUNHO E DA MAO': 'S66.9',
    'TRAUMATISMO DE MUSCULO E TENDAO NAO ESPECIFICADOS DO TORNOZELO E DO PE': 'S96.9',
    'TRAUMATISMO DE MUSCULO E TENDAO NAO ESPECIFICADOS DO TRONCO': 'T09.5',
    'TRAUMATISMO DE MUSCULO INTRINSECO E TENDAO AO NIVEL DO TORNOZELO E DO PE': 'S96.2',
    'TRAUMATISMO DE MUSCULOS E TENDOES DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.6',
    'TRAUMATISMO DE NERVO DO TORAX NAO ESPECIFICADO': 'S24.6',
    'TRAUMATISMO DE NERVO NAO ESPECIFICADO AO NIVEL DO OMBRO E DO BRACO': 'S44.9',
    'TRAUMATISMO DE NERVO NAO ESPECIFICADO, AO NIVEL DO TORNOZELO E DO PE': 'S94.9',
    'TRAUMATISMO DE NERVOS AO NIVEL DO OMBRO E DO BRACO': 'S44',
    'TRAUMATISMO DE NERVOS AO NIVEL DO PUNHO E DA MAO': 'S64',
    'TRAUMATISMO DE NERVOS AO NIVEL DO QUADRIL E DA COXA': 'S74',
    'TRAUMATISMO DE NERVOS E DA MEDULA ESPINHAL AO NIVEL CERVICAL': 'S14',
    'TRAUMATISMO DE ORGAO INTRA-ABDOMINAL NAO ESPECIFICADO': 'S36.9',
    'TRAUMATISMO DE ORGAO PELVICO NAO ESPECIFICADO': 'S37.9',
    'TRAUMATISMO DE ORGAOS INTRA-ABDOMINAIS': 'S36',
    'TRAUMATISMO DE OUTRO(S) MUSCULO(S) E TENDAO(OES) DO GRUPO MUSCULAR POSTERIOR AO NIVEL DA PERNA': 'S86.1',
    'TRAUMATISMO DE OUTROS MUSCULOS E TENDOES AO NIVEL DO PUNHO E DA MAO': 'S66.8',
    'TRAUMATISMO DE OUTROS MUSCULOS E TENDOES E DOS NAO ESPECIFICADOS AO NIVEL DO ANTEBRACO': 'S56.8',
    'TRAUMATISMO DE OUTROS NERVOS AO NIVEL DO ANTEBRACO': 'S54.8',
    'TRAUMATISMO DE OUTROS NERVOS AO NIVEL DO PUNHO E DA MAO': 'S64.8',
    'TRAUMATISMO DE OUTROS NERVOS AO NIVEL DO QUADRIL E DA COXA': 'S74.8',
    'TRAUMATISMO DE OUTROS VASOS SANGUINEOS AO NIVEL DO PUNHO E DE MAO': 'S65.8',
    'TRAUMATISMO DE PARTO DO NERVO FACIAL': 'P11.3',
    'TRAUMATISMO DE RAIZ NERVOSA DA MEDULA LOMBAR E SACRA': 'S34.2',
    'TRAUMATISMO DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14',
    'TRAUMATISMO DE TENDAO E MUSCULO AO NIVEL DO OMBRO E DO BRACO': 'S46',
    'TRAUMATISMO DE TENDOES E DE MUSCULOS DO PESCOCO': 'S16',
    'TRAUMATISMO DE VASO SANGUINEO NAO ESPECIFICADO AO NIVEL DO ABDOME, DO DORSO E DA PELVE': 'S35.9',
    'TRAUMATISMO DE VASO(S) SANGUINEO(S) DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.5',
    'TRAUMATISMO DE VASO(S) SANGUINEO(S) DO POLEGAR': 'S65.4',
    'TRAUMATISMO DE VASOS MULTIPLOS AO NIVEL DO ABDOME, DO DORSO E DA PELVE': 'S35.7',
    'TRAUMATISMO DE VASOS SANGUINEOS DA PERNA': 'S85',
    'TRAUMATISMO DE VASOS SANGUINEOS INTERCOSTAIS': 'S25.5',
    'TRAUMATISMO DE VEIA AO NIVEL DO ANTEBRACO': 'S55.2',
    'TRAUMATISMO DO APARELHO URINARIO E DE ORGAOS PELVICOS': 'S37',
    'TRAUMATISMO DO FIGADO OU DA VESICULA BILIAR': 'S36.1',
    'TRAUMATISMO DO MUSCULO E DO TENDAO DO ADUTOR DA COXA': 'S76.2',
    'TRAUMATISMO DO MUSCULO E DO TENDAO DO QUADRICEPS': 'S76.1',
    'TRAUMATISMO DO MUSCULO E DO TENDAO DO QUADRIL': 'S76.0',
    'TRAUMATISMO DO MUSCULO E DO TENDAO DOS MUSCULOS POSTERIORES AO NIVEL DA COXA': 'S76.3',
    'TRAUMATISMO DO MUSCULO E TENDAO AO NIVEL DO ANTEBRACO': 'S56',
    'TRAUMATISMO DO MUSCULO E TENDAO DA CABECA LONGA DO BICEPS': 'S46.1',
    'TRAUMATISMO DO MUSCULO E TENDAO DE OUTRAS PARTES DO BICEPS': 'S46.2',
    'TRAUMATISMO DO MUSCULO EXTENSOR E TENDAO DE OUTRO DEDO AO NIVEL DO PUNHO E DA MAO': 'S66.3',
    'TRAUMATISMO DO MUSCULO FLEXOR E TENDAO DE OUTRO DEDO AO NIVEL DO PUNHO E DA MAO': 'S66.1',
    'TRAUMATISMO DO MUSCULO FLEXOR E TENDAO DE OUTRO(S) DEDO(S) AO NIVEL DO ANTEBRACO': 'S56.1',
    'TRAUMATISMO DO MUSCULO FLEXOR E TENDAO DO POLEGAR AO NIVEL DO ANTEBRACO': 'S56.0',
    'TRAUMATISMO DO MUSCULO FLEXOR LONGO E TENDAO DO POLEGAR AO NIVEL DO PUNHO E DA MAO': 'S66.0',
    'TRAUMATISMO DO MUSCULO INTRINSECO E TENDAO DE OUTRO DEDO AO NIVEL DO PUNHO E DA MAO': 'S66.5',
    'TRAUMATISMO DO MUSCULO INTRINSECO E TENDAO DO POLEGAR AO NIVEL DO PUNHO E DA MAO': 'S66.4',
    'TRAUMATISMO DO NERVO CIATICO AO NIVEL DO QUADRIL E DA COXA': 'S74.0',
    'TRAUMATISMO DO NERVO DIGITAL DE OUTRO DEDO': 'S64.4',
    'TRAUMATISMO DO NERVO DIGITAL DO POLEGAR': 'S64.3',
    'TRAUMATISMO DO NERVO FACIAL': 'S04.5',
    'TRAUMATISMO DO NERVO FEMURAL AO NIVEL DO QUADRIL E DA COXA': 'S74.1',
    'TRAUMATISMO DO NERVO MEDIANO AO NIVEL DO PUNHO E DA MAO': 'S64.1',
    'TRAUMATISMO DO NERVO PLANTAR EXTERNO (LATERAL)': 'S94.0',
    'TRAUMATISMO DO NERVO PLANTAR INTERNO (MEDIAL)': 'S94.1',
    'TRAUMATISMO DO NERVO RADIAL AO NIVEL DO BRACO': 'S44.2',
    'TRAUMATISMO DO OLHO E DA ORBITA OCULAR': 'S05',
    'TRAUMATISMO DO OLHO E DA ORBITA, NAO ESPECIFICADO': 'S05.9',
    'TRAUMATISMO DO PLEXO BRAQUIAL': 'S14.3',
    'TRAUMATISMO DO PLEXO LOMBOSSACRO': 'S34.4',
    'TRAUMATISMO DO RIM': 'S37.0',
    'TRAUMATISMO DO TENDAO DE AQUILES': 'S86.0',
    'TRAUMATISMO DO TENDAO DO MANGUITO ROTADOR DO OMBRO': 'S46.0',
    'TRAUMATISMO DO(S) MUSCULO(S) E TENDAO(OES) DO GRUPO MUSCULAR ANTERIOR AO NIVEL DA PERNA': 'S86.2',
    'TRAUMATISMO DO(S) MUSCULO(S) E TENDAO(OES) DO GRUPO MUSCULAR PERONIAL AO NIVEL DA PERNA': 'S86.3',
    'TRAUMATISMO DOS MUSCULOS E DOS TENDOES DA CABECA': 'S09.1',
    'TRAUMATISMO DOS NERVOS AO NIVEL DO TORNOZELO E DO PE': 'S94',
    'TRAUMATISMO DOS VASOS SANGUINEOS DA CABECA NAO CLASSIFICADOS EM OUTRA PARTE': 'S09.0',
    'TRAUMATISMO INTRACRANIANO': 'S06',
    'TRAUMATISMO INTRACRANIANO COM COMA PROLONGADO': 'S06.7',
    'TRAUMATISMO INTRACRANIANO, NAO ESPECIFICADO': 'S06.9',
    'TRAUMATISMO NAO ESPECIFICADO': 'T14.9',
    'TRAUMATISMO NAO ESPECIFICADO DA CABECA': 'S09.9',
    'TRAUMATISMO NAO ESPECIFICADO DA PERNA': 'S89.9',
    'TRAUMATISMO NAO ESPECIFICADO DO ABDOME, DO DORSO E DA PELVE': 'S39.9',
    'TRAUMATISMO NAO ESPECIFICADO DO ANTEBRACO': 'S59.9',
    'TRAUMATISMO NAO ESPECIFICADO DO MEMBRO INFERIOR, NIVEL NAO ESPECIFICADO': 'T13.9',
    'TRAUMATISMO NAO ESPECIFICADO DO MEMBRO SUPERIOR NIVEL NAO ESPECIFICADO': 'T11.9',
    'TRAUMATISMO NAO ESPECIFICADO DO OMBRO E DO BRACO': 'S49.9',
    'TRAUMATISMO NAO ESPECIFICADO DO PESCOCO': 'S19.9',
    'TRAUMATISMO NAO ESPECIFICADO DO QUADRIL E DA COXA': 'S79.9',
    'TRAUMATISMO NAO ESPECIFICADO DO TORAX': 'S29.9',
    'TRAUMATISMO NAO ESPECIFICADO DO TRONCO, NIVEL NAO ESPECIFICADO': 'T09.9',
    'TRAUMATISMO NAO ESPECIFICADOS DO PUNHO E DA MAO': 'S69.9',
    'TRAUMATISMO POR ESMAGAMENTO DA PERNA': 'S87',
    'TRAUMATISMO POR ESMAGAMENTO E AMPUTACAO TRAUMATICA DE REGIOES NAO ESPECIFICADAS DO CORPO': 'T14.7',
    'TRAUMATISMO SUPERFICIAL DA CABECA': 'S00',
    'TRAUMATISMO SUPERFICIAL DA CABECA, PARTE NAO ESPECIFICADA': 'S00.9',
    'TRAUMATISMO SUPERFICIAL DA PERNA': 'S80',
    'TRAUMATISMO SUPERFICIAL DE MEMBRO INFERIOR, NIVEL NAO ESPECIFICADO': 'T13.0',
    'TRAUMATISMO SUPERFICIAL DE OUTRAS LOCALIZACOES DO PESCOCO': 'S10.8',
    'TRAUMATISMO SUPERFICIAL DE OUTRAS PARTES DA CABECA': 'S00.8',
    'TRAUMATISMO SUPERFICIAL DE OUTRAS PARTES ESPECIFICADAS DO TORAX E DAS NAO ESPECIFICADAS': 'S20.8',
    'TRAUMATISMO SUPERFICIAL DE PARTE NAO ESPECIFICADA DO ABDOME, DO DORSO E DA PELVE': 'S30.9',
    'TRAUMATISMO SUPERFICIAL DE REGIAO NAO ESPECIFICADA DO CORPO': 'T14.0',
    'TRAUMATISMO SUPERFICIAL DO ABDOME DO DORSO E DA PELVE': 'S30',
    'TRAUMATISMO SUPERFICIAL DO ANTEBRACO, NAO ESPECIFICADO': 'S50.9',
    'TRAUMATISMO SUPERFICIAL DO COTOVELO E DO ANTEBRACO': 'S50',
    'TRAUMATISMO SUPERFICIAL DO COURO CABELUDO': 'S00.0',
    'TRAUMATISMO SUPERFICIAL DO MEMBRO SUPERIOR, NIVEL NAO ESPECIFICADO': 'T11.0',
    'TRAUMATISMO SUPERFICIAL DO NARIZ': 'S00.3',
    'TRAUMATISMO SUPERFICIAL DO OMBRO E DO BRACO': 'S40',
    'TRAUMATISMO SUPERFICIAL DO OUVIDO': 'S00.4',
    'TRAUMATISMO SUPERFICIAL DO PESCOCO': 'S10',
    'TRAUMATISMO SUPERFICIAL DO PESCOCO, PARTE NAO ESPECIFICADA': 'S10.9',
    'TRAUMATISMO SUPERFICIAL DO PUNHO E DA MAO': 'S60',
    'TRAUMATISMO SUPERFICIAL DO QUADRIL E DA COXA': 'S70',
    'TRAUMATISMO SUPERFICIAL DO TORAX': 'S20',
    'TRAUMATISMO SUPERFICIAL DO TORNOZELO E DO PE': 'S90',
    'TRAUMATISMO SUPERFICIAL DO TORNOZELO E DO PE, NAO ESPECIFICADO': 'S90.9',
    'TRAUMATISMO SUPERFICIAL DO TRONCO, NIVEL NAO ESPECIFICADO': 'T09.0',
    'TRAUMATISMO SUPERFICIAL DOS LABIOS E DA CAVIDADE ORAL': 'S00.5',
    'TRAUMATISMO SUPERFICIAL NAO ESPECIFICADO DA PERNA': 'S80.9',
    'TRAUMATISMO SUPERFICIAL NAO ESPECIFICADO DO OMBRO E DO BRACO': 'S40.9',
    'TRAUMATISMO SUPERFICIAL NAO ESPECIFICADO DO PUNHO E DA MAO': 'S60.9',
    'TRAUMATISMO SUPERFICIAL NAO ESPECIFICADO DO QUADRIL E DA COXA': 'S70.9',
    'TRAUMATISMOS DE MUSCULO E DE TENDAO AO NIVEL DA PERNA': 'S86',
    'TRAUMATISMOS DE OUTROS MUSCULOS E TENDOES AO NIVEL DA PERNA': 'S86.8',
    'TRAUMATISMOS DO MUSCULO E TENDAO AO NIVEL DO TORNOZELO E DO PE': 'S96',
    'TRAUMATISMOS MULTIPLOS DA CABECA': 'S09.7',
    'TRAUMATISMOS MULTIPLOS DO COTOVELO': 'S59.7',
    'TRAUMATISMOS MULTIPLOS DO OMBRO E DO BRACO': 'S49.7',
    'TRAUMATISMOS MULTIPLOS DO PUNHO E DA MAO': 'S69.7',
    'TRAUMATISMOS MULTIPLOS DO TORNOZELO E DO PE': 'S99.7',
    'TRAUMATISMOS MULTIPLOS NAO ESPECIFICADOS': 'T07',
    'TRAUMATISMOS NAO ESPECIFICADOS DO TORNOZELO E DO PE': 'S99.9',
    'TRAUMATISMOS POR ESMAGAMENTO ENVOLVENDO MULTIPLAS REGIOES DO CORPO': 'T04',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO A CABECA COM O PESCOCO': 'T00.0',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO MULTIPLAS REGIOES DO CORPO': 'T00',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO O TORAX COM O ABDOME, PARTE INFERIOR DO DORSO E DA PELVE': 'T00.1',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO OUTRAS COMBINACOES DE REGIOES DO CORPO': 'T00.8',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO REGIOES MULTIPLAS DO(S) MEMBRO(S) INFERIOR(ES)': 'T00.3',
    'TRAUMATISMOS SUPERFICIAIS ENVOLVENDO REGIOES MULTIPLAS DO(S) MEMBRO(S) SUPERIOR(ES)': 'T00.2',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DA CABECA': 'S00.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DA PERNA': 'S80.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO ABDOME, DO DORSO E DA PELVE': 'S30.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO ANTEBRACO': 'S50.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO OMBRO E DO BRACO': 'S40.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO PUNHO E DA MAO': 'S60.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO QUADRIL E DA COXA': 'S70.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO TORAX': 'S20.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS DO TORNOZELO E DO PE': 'S90.7',
    'TRAUMATISMOS SUPERFICIAIS MULTIPLOS NAO ESPECIFICADOS': 'T00.9',
    'TREMOR ESSENCIAL': 'G25.0',
    'TREMOR INDUZIDO POR DROGAS': 'G25.1',
    'TREMOR NAO ESPECIFICADO': 'R25.1',
    'TRICOMONIASE': 'A59',
    'TRICOMONIASE NAO ESPECIFICADA': 'A59.9',
    'TRICOMONIASE UROGENITAL': 'A59.0',
    'TRICURIASE': 'B79',
    'TRISTEZA': 'R45.2',
    'TROMBOANGEITE OBLITERANTE [DOENCA DE BUERGER]': 'I73.1',
    'TROMBOCITOPENIA NAO ESPECIFICADA': 'D69.6',
    'TROMBOCITOPENIA SECUNDARIA': 'D69.5',
    'TROMBOCITOSE ESSENCIAL': 'D75.2',
    'TROMBOFLEBITE MIGRATORIA': 'I82.1',
    'TUBERCULOSA RESPIRATORIA PRIMARIA SEM MENCAO DE CONFIRMACAO BACTERIOLOGICA OU HISTOLOGICA': 'A16.7',
    'TUBERCULOSE DA LARINGE, DA TRAQUEIA E DOS BRONQUIOS, COM CONFIRMACAO BACTERIOLOGICA E HISTOLOGICA': 'A15.5',
    'TUBERCULOSE DAS VIAS RESPIRATORIAS SEM CONFIRMACAO BACTERIOLOGICA OU HISTOLOGICA': 'A16',
    'TUBERCULOSE DE OUTROS ORGAOS': 'A18',
    'TUBERCULOSE DE OUTROS ORGAOS ESPECIFICADOS': 'A18.8',
    'TUBERCULOSE DE PELE E DO TECIDO CELULAR SUBCUTANEO': 'A18.4',
    'TUBERCULOSE DO APARELHO GENITURINARIO': 'A18.1',
    'TUBERCULOSE MILIAR NAO ESPECIFICADA': 'A19.9',
    'TUBERCULOSE NAO ESPECIFICADA DAS VIAS RESPIRATORIAS, COM CONFIRMACAO BACTERIOLOGICA E HISTOLOGICA': 'A15.9',
    'TUBERCULOSE PRIMARIA DAS VIAS RESPIRATORIAS, COM CONFIRMACAO BACTERIOLOGICA E HISTOLOGICA': 'A15.7',
    'TUBERCULOSE PULMONAR COM EXAMES BACTERIOLOGICO E HISTOLOGICO NEGATIVOS': 'A16.0',
    'TUBERCULOSE PULMONAR, COM CONFIRMACAO HISTOLOGICA': 'A15.2',
    'TUBERCULOSE PULMONAR, COM CONFIRMACAO POR EXAME MICROSCOPICO DA EXPECTORACAO, COM OU SEM CULTURA': 'A15.0',
    'TUBERCULOSE PULMONAR, COM CONFIRMACAO POR MEIO NAO ESPECIFICADO': 'A15.3',
    'TUBERCULOSE PULMONAR, COM CONFIRMACAO SOMENTE POR CULTURA': 'A15.1',
    'TUBERCULOSE PULMONAR, SEM MENCAO DE CONFIRMACAO BACTERIOLOGICA OU HISTOLOGICA': 'A16.2',
    'TUBERCULOSE PULMONAR, SEM REALIZACAO DE EXAME BACTERIOLOGICO OU HISTOLOGICO': 'A16.1',
    'TUBERCULOSE RESPIRATORIA COM CONFIRMACAO BACTERIOLOGICA E HISTOLOGICA': 'A15',
    'TUBERCULOSE RESPIRATORIA, NAO ESPECIFICADA, SEM MENCAO DE CONFIRMACAO BACTERIOLOGICA OU HISTOLOGICA': 'A16.9',
    'TUMEFACAO MASSA OU TUMORACAO LOCALIZADAS DA PELE E DO TECIDO SUBCUTANEO': 'R22',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS DA CABECA': 'R22.0',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS DE MEMBRO SUPERIOR': 'R22.3',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS DE MULTIPLAS LOCALIZACOES': 'R22.7',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS DO PESCOCO': 'R22.1',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS DO TRONCO': 'R22.2',
    'TUMEFACAO, MASSA OU TUMORACAO LOCALIZADAS NO MEMBRO INFERIOR': 'R22.4',
    'TUMEFACAO, MASSA OU TUMORACAO NAO ESPECIFICADAS, LOCALIZADAS': 'R22.9',
    'TUMORES DE COMPORTAMENTO INCERTO OU DESCONHECIDO DE MASTOCITOS E CELULAS HISTIOCITICAS': 'D47.0',
    'TUNGIASE [INFESTACAO PELA PULGA DA AREIA]': 'B88.1',
    'ULCERA CRONICA DA PELE, NAO CLASSIFICADA EM OUTRA PARTE': 'L98.4',
    'ULCERA DE CORNEA': 'H16.0',
    'ULCERA DE DECUBITO': 'L89',
    'ULCERA DO ANUS E DO RETO': 'K62.6',
    'ULCERA DO PENIS': 'N48.5',
    'ULCERA DOS MEMBROS INFERIORES NAO CLASSIFICADA EM OUTRA PARTE': 'L97',
    'ULCERA DUODENAL': 'K26',
    'ULCERA DUODENAL - AGUDA COM HEMORRAGIA': 'K26.0',
    'ULCERA DUODENAL - NAO ESPECIFICADA COMO AGUDA OU CRONICA, SEM HEMORRAGIA OU PERFURACAO': 'K26.9',
    'ULCERA GASTRICA': 'K25',
    'ULCERA GASTRICA - AGUDA COM HEMORRAGIA': 'K25.0',
    'ULCERA GASTRICA - AGUDA COM HEMORRAGIA E PERFURACAO': 'K25.2',
    'ULCERA GASTRICA - AGUDA SEM HEMORRAGIA OU PERFURACAO': 'K25.3',
    'ULCERA GASTRICA - CRONICA OU NAO ESPECIFICADA COM HEMORRAGIA': 'K25.4',
    'ULCERA GASTRICA - CRONICA OU NAO ESPECIFICADA COM HEMORRAGIA E PERFURACAO': 'K25.6',
    'ULCERA GASTRICA - CRONICA SEM HEMORRAGIA OU PERFURACAO': 'K25.7',
    'ULCERA GASTRICA - NAO ESPECIFICADA COMO AGUDA OU CRONICA, SEM HEMORRAGIA OU PERFURACAO': 'K25.9',
    'ULCERA PEPTICA DE LOCALIZACAO NAO ESPECIFICADA': 'K27',
    'ULCERA PEPTICA DE LOCALIZACAO NAO ESPECIFICADA - AGUDA COM HEMORRAGIA': 'K27.0',
    'ULCERACAO DA VULVA EM DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'N77.0',
    'ULCERACAO E INFLAMACAO VULVOVAGINAIS EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N77',
    'ULCERACAO E INFLAMACAO VULVOVAGINAIS EM OUTRAS DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N77.8',
    'ULCERACAO VAGINAL': 'N76.5',
    'ULCERACAO VULVAR': 'N76.6',
    'UNHA ENCRAVADA': 'L60.0',
    'UREMIA EXTRA-RENAL': 'R39.2',
    'URETRITE E SINDROME URETRAL': 'N34',
    'URETRITE EM DOENCAS CLASSIFICADAS EM OUTRA PARTE': 'N37.0',
    'URETRITES NAO ESPECIFICAS': 'N34.1',
    'UROPATIA ASSOCIADA A REFLUXO VESICO-URETERAL': 'N13.7',
    'UROPATIA OBSTRUTIVA E POR REFLUXO': 'N13',
    'UROPATIA OBSTRUTIVA E POR REFLUXO NAO ESPECIFICADA': 'N13.9',
    'URTICARIA': 'L50',
    'URTICARIA ALERGICA': 'L50.0',
    'URTICARIA DE CONTATO': 'L50.6',
    'URTICARIA DERMATOGRAFICA': 'L50.3',
    'URTICARIA DEVIDA A FRIO E A CALOR': 'L50.2',
    'URTICARIA IDIOPATICA': 'L50.1',
    'URTICARIA NAO ESPECIFICADA': 'L50.9',
    'URTICARIA SOLAR': 'L56.3',
    'USO DE ALCOOL': 'Z72.1',
    'USO DE DROGA': 'Z72.2',
    'USO DO TABACO': 'Z72.0',
    'VACINACAO CONTRA OUTRAS DOENCAS BACTERIANAS UNICAS': 'Z23.8',
    'VAGINISMO': 'N94.2',
    'VAGINISMO NAO-ORGANICO': 'F52.5',
    'VAGINITE AGUDA': 'N76.0',
    'VAGINITE SUBAGUDA E CRONICA': 'N76.1',
    'VAGINITE, VULVITE E VULVOVAGINITE EM DOENCAS INFECCIOSAS E PARASITARIAS CLASSIFICADAS EM OUTRA PARTE': 'N77.1',
    'VALOR ANORMAL DA PRESSAO ARTERIAL SEM DIAGNOSTICO': 'R03',
    'VALOR BAIXO DA PRESSAO ARTERIAL NAO ESPECIFICO': 'R03.1',
    'VALOR ELEVADO DA PRESSAO ARTERIAL SEM O DIAGNOSTICO DE HIPERTENSAO': 'R03.0',
    'VARICELA COM OUTRAS COMPLICACOES': 'B01.8',
    'VARICELA SEM COMPLICACAO': 'B01.9',
    'VARIOLA DOS MACACOS [MONKEYPOX]': 'B04',
    'VARIZES DE OUTRAS LOCALIZACOES': 'I86',
    'VARIZES DE OUTRAS LOCALIZACOES ESPECIFICADAS': 'I86.8',
    'VARIZES DOS MEMBROS INFERIORES': 'I83',
    'VARIZES DOS MEMBROS INFERIORES COM INFLAMACAO': 'I83.1',
    'VARIZES DOS MEMBROS INFERIORES COM ULCERA': 'I83.0',
    'VARIZES DOS MEMBROS INFERIORES COM ULCERA E INFLAMACAO': 'I83.2',
    'VARIZES DOS MEMBROS INFERIORES SEM ULCERA OU INFLAMACAO': 'I83.9',
    'VARIZES ESCROTAIS': 'I86.1',
    'VARIZES ESOFAGIANAS SANGRANTES': 'I85.0',
    'VARIZES ESOFAGIANAS SEM SANGRAMENTO': 'I85.9',
    'VARIZES GASTRICAS': 'I86.4',
    'VARIZES PELVICAS': 'I86.2',
    'VASCULITE LIMITADA A PELE NAO CLASSIFICADAS EM OUTRA PARTE': 'L95',
    'VASCULITES LIMITADAS A PELE, NAO ESPECIFICADAS': 'L95.9',
    'VASCULOPATIA NECROTIZANTE NAO ESPECIFICADA': 'M31.9',
    'VERBORRAGIA E PORMENORES CIRCUNSTANCIAIS MASCARANDO O MOTIVO DA CONSULTA': 'R46.7',
    'VERRUGAS ANOGENITAIS (VENEREAS)': 'A63.0',
    'VERRUGAS DE ORIGEM VIRAL': 'B07',
    'VERTIGEM DE ORIGEM CENTRAL': 'H81.4',
    'VERTIGEM EPIDEMICA': 'A88.1',
    'VERTIGEM PAROXISTICA BENIGNA': 'H81.1',
    'VIOLENCIA FISICA': 'R45.6',
    'VIRUS COMO CAUSA DE DOENCAS CLASSIFICADAS EM OUTROS CAPITULOS': 'B97',
    'VISAO SUBNORMAL DE AMBOS OS OLHOS': 'H54.2',
    'VISAO SUBNORMAL EM UM OLHO': 'H54.5',
    'VITILIGO': 'L80',
    'VITIMA DE CRIME OU DE ATOS TERRORISTAS': 'Z65.4',
    'VOLVO': 'K56.2',
    'VOMITOS ASSOCIADOS A OUTROS DISTURBIOS PSICOLOGICOS': 'F50.5',
    'VOMITOS DA GRAVIDEZ, NAO ESPECIFICADOS': 'O21.9',
    'VOMITOS EXCESSIVOS NA GRAVIDEZ': 'O21',
    'VOMITOS NO RECEM-NASCIDO': 'P92.0',
    'VOMITOS POS-CIRURGIA GASTROINTESTINAL': 'K91.0',
    'VOMITOS TARDIOS DA GRAVIDEZ': 'O21.2',
    'VULVITE AGUDA': 'N76.2',
    'XANTELASMA DA PALPEBRA': 'H02.6',
    'XERODERMA PIGMENTOSO': 'Q82.1',
    'XEROSE CUTANEA': 'L85.3',
}

# ── Mapeamento por palavras-chave para capítulo (fallback) ───────────
# Quando não há código CID e a descrição não está no dicionário acima,
# tenta determinar o capítulo por palavras-chave na descrição.
CHAPTER_KEYWORDS = {
    'I - Doenças infecciosas e parasitárias': [
        'DENGUE', 'CHIKUNGUNYA', 'ZIKA', 'MALARIA', 'TUBERCULOSE',
        'HANSENIASE', 'LEPTOSPIROSE', 'HEPATITE', 'HIV', 'AIDS',
        'SIFILIS', 'GONOCOCIC', 'HERPES', 'VARICELA', 'SARAMPO',
        'RUBEOLA', 'CAXUMBA', 'MENINGITE', 'TETANO', 'COQUELUCHE',
        'DIARREIA', 'GASTROENTERITE DE ORIGEM INFECCIOSA', 'FEBRE TIFOIDE',
        'COLERA', 'SEPTICEMIA', 'ERISIPELA', 'ESQUISTOSSOMOSE',
        'LEISHMANIOSE', 'CHAGAS', 'TOXOPLASMOSE', 'RAIVA', 'PESTE',
        'FEBRE AMARELA', 'FEBRE MACULOSA', 'FEBRE HEMORRAGICA',
    ],
    'IX - Aparelho circulatório': [
        'HIPERTENSAO', 'INFARTO', 'ANGINA', 'INSUFICIENCIA CARDIACA',
        'ACIDENTE VASCULAR CEREBRAL', 'AVC', 'FIBRILACAO', 'FLUTTER',
        'TAQUICARDIA', 'ARRITMIA', 'EMBOLIA PULMONAR', 'TROMBOSE',
        'VARIZES', 'HEMORROIDAS', 'HIPOTENSAO', 'ENDOCARDITE',
        'MIOCARDITE', 'PERICARDITE', 'DOENCA CARDIACA',
    ],
    'X - Aparelho respiratório': [
        'PNEUMONIA', 'BRONQUITE', 'BRONQUIOLITE', 'ASMA', 'GRIPE',
        'INFLUENZA', 'SINUSITE', 'FARINGITE', 'AMIGDALITE', 'LARINGITE',
        'TRAQUEITE', 'NASOFARINGITE', 'RESFRIADO', 'INFECCAO.*VIAS AEREAS',
        'PULMONAR OBSTRUTIVA', 'RINITE', 'OTITE',
    ],
    'XI - Aparelho digestivo': [
        'GASTRITE', 'ULCERA GASTRICA', 'ULCERA DUODENAL', 'APENDICITE',
        'COLELITIASE', 'COLECISTITE', 'PANCREATITE', 'DISPEPSIA',
        'REFLUXO GASTRO', 'HERNIA INGUINAL', 'HERNIA UMBILICAL',
        'CONSTIPACAO', 'CARIE DENTARIA', 'DIVERTICULAR', 'HEMORRAGIA GASTROINTESTINAL',
    ],
    'XIII - Sistema osteomuscular': [
        'DORSALGIA', 'CERVICALGIA', 'LUMBAGO', 'LOMBALGIA', 'ARTRITE',
        'ARTROSE', 'GOTA', 'MIALGIA', 'TENDINITE', 'BURSITE',
        'CONTRATURA', 'EPICONDILITE', 'FIBROMIALGIA', 'ESPONDIL',
        'RADICULOPATIA', 'DOR LOMBAR',
    ],
    'XIX - Lesões e causas externas': [
        'TRAUMATISMO', 'FRATURA', 'LUXACAO', 'ENTORSE', 'CONTUSAO',
        'FERIMENTO', 'QUEIMADURA', 'CORPO ESTRANHO', 'INTOXICACAO',
        'ENVENENAMENTO', 'MORDEDURA', 'PICADA', 'QUEDA',
        'MOTOCICLISTA', 'PEDESTRE', 'CICLISTA', 'AGRESSAO',
        'LESAO', 'EFEITO TOXICO',
    ],
    'XIV - Aparelho geniturinário': [
        'INFECCAO.*URINARI', 'CISTITE', 'CALCULO RENAL', 'COLICA RENAL',
        'PROSTAT', 'VAGINITE', 'MENSTRUACAO', 'HEMORRAGIA VAGINAL',
    ],
    'XVIII - Sintomas e sinais': [
        'DOR ABDOMINAL', 'NAUSEA', 'VOMITO', 'FEBRE NAO ESPECIFICADA',
        'CEFALEIA', 'SINCOPE', 'CONVULS', 'DOR TORACICA',
        'DISPNEIA', 'EPISTAXE', 'ERUPCAO CUTANEA', 'TONTURA',
        'MAL ESTAR', 'VERTIGEM', 'DOR AGUDA', 'DOR NAO CLASSIF',
        'RETENCAO URINARIA', 'HEMOPTISE',
    ],
    'IV - Endócrinas, nutricionais e metabólicas': [
        'DIABETES', 'HIPOGLICEMIA', 'DESIDRATACAO', 'HIPOPOTASSEMIA',
        'OBESIDADE', 'DESNUTRICAO', 'TIREOIDE', 'HIPOTIROID', 'HIPERTIROID',
    ],
    'XII - Pele e tecido subcutâneo': [
        'ABSCESSO CUTANEO', 'FURUNCULO', 'CELULITE', 'IMPETIGO',
        'DERMATITE', 'URTICARIA', 'PIODERMITE', 'ECZEMA',
    ],
    'V - Transtornos mentais': [
        'TRANSTORNO.*MENTAL', 'DEPRESSIVO', 'ANSIOS', 'BIPOLAR',
        'ESQUIZOFRENIA', 'USO DE ALCOOL', 'USO DE DROGA', 'PANICO',
    ],
    'XXI - Fatores que influenciam o estado de saúde': [
        'EXAME MEDICO', 'EXAME GERAL', 'VACINACAO', 'SUPERVISAO DE GRAVIDEZ',
        'PESSOA EM CONTATO', 'ACOMPANHAMENTO',
    ],
}

# ── Definições de Síndromes e Sentinelas ────────────────────────────
# Cada síndrome é definida como lista de prefixos de código CID
# (match via str.startswith)
SYNDROME_DEFS = {
    # Sentinelas de Saúde Coletiva
    "Saúde Mental":           ['F'],           # CID F00-F99 (cap. V)
    "Acidentes de Trânsito":  ['V0', 'V1', 'V2', 'V3', 'V4',
                               'V5', 'V6', 'V7', 'V8', 'V9'],  # V01-V99
    "Tentativas de Suicídio": ['X6', 'X70', 'X71', 'X72', 'X73', 'X74',
                               'X75', 'X76', 'X77', 'X78', 'X79',
                               'X80', 'X81', 'X82', 'X83', 'X84'],  # X60-X84
    "Agressões":              ['X85', 'X86', 'X87', 'X88', 'X89',
                               'X90', 'X91', 'X92', 'X93', 'X94',
                               'X95', 'X96', 'X97', 'X98', 'X99',
                               'Y00', 'Y01', 'Y02', 'Y03', 'Y04',
                               'Y05', 'Y06', 'Y07', 'Y08', 'Y09'],  # X85-Y09
    # Síndromes compostas
    "Gastroenterites (A09+K52)":    ['A09', 'K52'],
    "Sínd. Gripal (IVAS+Febre)":    ['J00', 'J01', 'J02', 'J03',
                                     'J04', 'J05', 'J06', 'R50'],
    "Sínd. Respiratória (J09-J22)": ['J09', 'J10', 'J11', 'J12',
                                     'J13', 'J14', 'J15', 'J16',
                                     'J17', 'J18', 'J20', 'J21', 'J22'],
    "Geniturinária (ITU+Cólica)":   ['N30', 'N39', 'N20', 'N21', 'N23'],
    "Febril Inespecífica (R50)":    ['R50'],
    "Exantemática (R21+B05-09)":    ['R21', 'B05', 'B06', 'B07', 'B08', 'B09'],
    "Animais Peçonhentos":          ['X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'T63'],
    "Escorpionismo (X22+T632)":     ['X22', 'T63'],
    "Pneumonia (J12-J18)":          ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'],
    "DPOC (J44)":                   ['J44'],
    "Asma (J45-J46)":               ['J45', 'J46'],
    "Dor Osteomuscular":            ['M54', 'M79', 'M25'],
    "Cardiovascular Aguda":         ['I20', 'I21', 'I22', 'I24', 'I50', 'I63', 'I64'],
    "Dermatológica Aguda":          ['L01', 'L02', 'L03', 'L04', 'L08', 'L50', 'L51', 'L53'],
    "Acidentes de Trabalho":        ['W', 'X3'],
}


def desc_to_cid_code(desc):
    """Mapeia descrição CID (DATASUS) para código CID-10.

    Usa: (1) dicionário exato, (2) busca parcial no dicionário.
    Retorna o código ou None se não encontrar.
    """
    if desc is None or not isinstance(desc, str) or desc.strip() == '':
        return None

    d = desc.strip().upper()

    # 1. Busca exata
    if d in CID_DESC_TO_CODE:
        return CID_DESC_TO_CODE[d]

    # 2. Busca parcial — a descrição pode conter texto extra
    for key, code in CID_DESC_TO_CODE.items():
        if key in d or d in key:
            return code

    return None


def desc_to_chapter(desc):
    """Determina capítulo CID a partir da descrição usando palavras-chave.

    Fallback para quando não há código CID nem match no dicionário.
    """
    import re
    if desc is None or not isinstance(desc, str) or desc.strip() == '':
        return None

    d = desc.strip().upper()

    for chapter, keywords in CHAPTER_KEYWORDS.items():
        for kw in keywords:
            if '.*' in kw:
                if re.search(kw, d):
                    return chapter
            elif kw in d:
                return chapter

    return None


# ── Mapeamento CID ───────────────────────────────────────────────────

CID_CHAPTERS = {
    'A': 'I - Doenças infecciosas e parasitárias',
    'B': 'I - Doenças infecciosas e parasitárias',
    'C': 'II - Neoplasias',
    'D0': 'II - Neoplasias',
    'D1': 'II - Neoplasias',
    'D2': 'II - Neoplasias',
    'D3': 'II - Neoplasias',
    'D4': 'II - Neoplasias',
    'D5': 'III - Sangue e órgãos hematopoéticos',
    'D6': 'III - Sangue e órgãos hematopoéticos',
    'D7': 'III - Sangue e órgãos hematopoéticos',
    'D8': 'III - Sangue e órgãos hematopoéticos',
    'D9': 'III - Sangue e órgãos hematopoéticos',  # D80-D89 = IV, simplificação
    'E': 'IV - Endócrinas, nutricionais e metabólicas',
    'F': 'V - Transtornos mentais',
    'G': 'VI - Sistema nervoso',
    'H0': 'VII - Olho e anexos',
    'H1': 'VII - Olho e anexos',
    'H2': 'VII - Olho e anexos',
    'H3': 'VII - Olho e anexos',
    'H4': 'VII - Olho e anexos',
    'H5': 'VII - Olho e anexos',
    'H6': 'VIII - Ouvido e apófise mastóide',
    'H7': 'VIII - Ouvido e apófise mastóide',
    'H8': 'VIII - Ouvido e apófise mastóide',
    'H9': 'VIII - Ouvido e apófise mastóide',
    'I': 'IX - Aparelho circulatório',
    'J': 'X - Aparelho respiratório',
    'K': 'XI - Aparelho digestivo',
    'L': 'XII - Pele e tecido subcutâneo',
    'M': 'XIII - Sistema osteomuscular',
    'N': 'XIV - Aparelho geniturinário',
    'O': 'XV - Gravidez, parto e puerpério',
    'P': 'XVI - Afecções perinatais',
    'Q': 'XVII - Malformações congênitas',
    'R': 'XVIII - Sintomas e sinais',
    'S': 'XIX - Lesões e causas externas',
    'T': 'XIX - Lesões e causas externas',
    'U': 'XXII - Códigos para propósitos especiais',
    'V': 'XX - Causas externas de morbidade',
    'W': 'XX - Causas externas de morbidade',
    'X': 'XX - Causas externas de morbidade',
    'Y': 'XX - Causas externas de morbidade',
    'Z': 'XXI - Fatores que influenciam o estado de saúde',
}

def cid_to_chapter(cid_str):
    """Mapeia código ou descrição CID para capítulo."""
    if cid_str is None or not isinstance(cid_str, str) or cid_str == '':
        return None

    code = cid_str.strip().upper()

    # Se é descrição com código entre parênteses: "Dengue (A90)"
    import re
    m = re.search(r'\b([A-Z]\d{2})', code)
    if m:
        code = m.group(1)

    if len(code) < 2 or not code[0].isalpha():
        return None

    # Tentar mapeamento com 2 chars, depois 1
    key2 = code[:2]
    key1 = code[0]

    if key2 in CID_CHAPTERS:
        return CID_CHAPTERS[key2]
    if key1 in CID_CHAPTERS:
        return CID_CHAPTERS[key1]

    return None


# Agravos de notificação SINAN (CIDs de interesse)
SINAN_MAP = {
    'A90': 'Dengue', 'A91': 'Dengue hemorrágica',
    'A92.0': 'Chikungunya', 'A92.8': 'Zika',
    'A01': 'Febre tifóide', 'A09': 'Diarréia/gastroenterite',
    'A15': 'Tuberculose respiratória', 'A16': 'Tuberculose respiratória',
    'A17': 'Tuberculose SNC', 'A18': 'Tuberculose outros órgãos',
    'A19': 'Tuberculose miliar',
    'A27': 'Leptospirose',
    'A30': 'Hanseníase',
    'A33': 'Tétano neonatal', 'A34': 'Tétano obstétrico', 'A35': 'Tétano acidental',
    'A37': 'Coqueluche',
    'A50': 'Sífilis congênita', 'A51': 'Sífilis precoce', 'A52': 'Sífilis tardia', 'A53': 'Sífilis NE',
    'A69.2': 'Doença de Lyme',
    'A77': 'Febre maculosa',
    'A78': 'Febre Q',
    'A82': 'Raiva',
    'A95': 'Febre amarela',
    'B05': 'Sarampo',
    'B06': 'Rubéola',
    'B15': 'Hepatite A', 'B16': 'Hepatite B', 'B17': 'Hepatite C/outras',
    'B18': 'Hepatite crônica viral',
    'B19': 'Hepatite viral NE',
    'B20': 'HIV/AIDS', 'B21': 'HIV/AIDS', 'B22': 'HIV/AIDS', 'B23': 'HIV/AIDS', 'B24': 'HIV NE',
    'B26': 'Caxumba',
    'B50': 'Malária P.falciparum', 'B51': 'Malária P.vivax', 'B52': 'Malária P.malariae', 'B53': 'Malária outras',
    'B54': 'Malária NE',
    'B55': 'Leishmaniose',
    'B57': 'Doença de Chagas',
    'B65': 'Esquistossomose',
    'J09': 'Influenza pandêmica', 'J10': 'Influenza identificada', 'J11': 'Influenza NE',
    'J12': 'Pneumonia viral', 'J18': 'Pneumonia NE',
    'P35.0': 'Rubéola congênita',
    'U04': 'SRAG', 'U07.1': 'COVID-19', 'U07.2': 'COVID-19 suspeito',
}

def cid_to_sinan(cid_str):
    """Mapeia CID para agravo SINAN. Retorna 'Outros' se não for SINAN."""
    if cid_str is None or not isinstance(cid_str, str) or cid_str == '':
        return 'Outros'

    import re
    code = cid_str.strip().upper()
    m = re.search(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)', code)
    if not m:
        return 'Outros'

    code = m.group(1)

    # Tentar match exato, depois sem decimal, depois só 3 chars
    if code in SINAN_MAP:
        return SINAN_MAP[code]
    base = code.split('.')[0]
    if base in SINAN_MAP:
        return SINAN_MAP[base]

    return 'Outros'


def extract_cid_code(desc):
    """Extrai código CID de uma descrição como 'A90 - DENGUE [DENGUE CLÁSSICO]'."""
    import re
    if desc is None or not isinstance(desc, str) or desc == '':
        return None
    m = re.match(r'^([A-Z]\d{2}(?:\.\d{1,2})?)\b', desc.strip().upper())
    if m:
        return m.group(1)
    m = re.search(r'\(([A-Z]\d{2}(?:\.\d{1,2})?)\)', desc.strip().upper())
    if m:
        return m.group(1)
    return None


# ── Função principal ──────────────────────────────────────────────────

def run_pipeline(input_file, populations, output_file,
                 agravos='all', col_date='data', col_cid='cid_descricao',
                 col_qty='quantidade', monitor_year=None,
                 base_hist_years=None, skip_channel_estimation=False):
    # Usar BASE_HIST_YEARS global se não especificado
    if base_hist_years is None:
        base_hist_years = BASE_HIST_YEARS
    """
    Pipeline completo: CSV → canais endêmicos → JSON.

    Parâmetros:
        input_file: caminho para CSV com dados brutos ou pré-agregados
        populations: dict {ano: pop} ou int (pop constante)
        output_file: caminho para JSON de saída
        agravos: 'all', 'chapters', 'sinan', 'top_N' (ex: 'top_20'), ou lista de nomes
        col_date, col_cid, col_qty: nomes das colunas
        monitor_year: ano para monitorar (None = todos)
    """
    print(f"[1/5] Lendo dados de {input_file}...")

    # Ler CSV
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(input_file, sep=';', encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        df = pd.read_csv(input_file, encoding='latin-1')

    print(f"   {len(df)} registros, colunas: {list(df.columns)}")

    # Detectar colunas
    cols_lower = {c.lower().strip(): c for c in df.columns}

    if col_date not in df.columns:
        for key in ['data', 'dt_atendimento', 'date', 'dt_notif']:
            if key in cols_lower:
                col_date = cols_lower[key]
                break

    if col_cid not in df.columns:
        for key in ['cid_descricao', 'cid_codigo', 'cid', 'hipotese']:
            if key in cols_lower:
                col_cid = cols_lower[key]
                break

    if col_qty not in df.columns:
        if 'quantidade' in cols_lower:
            col_qty = cols_lower['quantidade']
        else:
            df['quantidade'] = 1
            col_qty = 'quantidade'

    # Extrair código CID — 3 estratégias em cascata:
    # 1. Coluna cid_codigo (se existir e tiver dados)
    # 2. extract_cid_code() — regex em descrições tipo "A90 - Dengue"
    # 3. desc_to_cid_code() — dicionário DATASUS descrição→código
    col_desc = None
    for key in ['cid_descricao', 'cid_desc']:
        if key in cols_lower:
            col_desc = cols_lower[key]
            break
    if col_desc is None:
        col_desc = col_cid  # fallback

    if 'cid_codigo' in df.columns:
        # Coluna cid_codigo existe no CSV → limpar valores vazios
        df['cid_codigo'] = df['cid_codigo'].astype(str).str.strip()
        df.loc[df['cid_codigo'].isin(['', 'nan', 'None', 'NaN']), 'cid_codigo'] = pd.NA
        # Estratégia 2: regex na descrição
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(extract_cid_code))
        # Estratégia 3: dicionário DATASUS
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(desc_to_cid_code))
        print(f"   Coluna cid_codigo: {df['cid_codigo'].notna().mean():.0%} com código")
    else:
        # Estratégia 2
        df['cid_codigo'] = df[col_cid].apply(extract_cid_code)
        # Estratégia 3
        mask_no_code = df['cid_codigo'].isna()
        if mask_no_code.any() and col_desc in df.columns:
            df.loc[mask_no_code, 'cid_codigo'] = (
                df.loc[mask_no_code, col_desc].apply(desc_to_cid_code))

    # Verificar cobertura
    cid_coverage = df['cid_codigo'].notna().mean()
    print(f"   Cobertura CID: {cid_coverage:.0%}")

    # Se cobertura ainda baixa, usar desc_to_chapter como fallback para capítulos
    use_desc_fallback = cid_coverage < 0.2
    if use_desc_fallback:
        print(f"   ⚠ Cobertura CID baixa ({cid_coverage:.0%}).")
        print(f"   → Usando desc_to_chapter() por palavras-chave para capítulos.")
        # Criar coluna auxiliar de capítulo por descrição
        if col_desc in df.columns:
            df['_chapter_from_desc'] = df[col_desc].apply(desc_to_chapter)
            chapter_coverage = df['_chapter_from_desc'].notna().mean()
            print(f"   → Capítulos por palavras-chave: {chapter_coverage:.0%} cobertos")

    # Normalizar populações
    if isinstance(populations, (int, float)):
        pop_dict = {}
        for y in df['ano_epi'].unique() if 'ano_epi' in df.columns else range(2020, 2027):
            pop_dict[int(y)] = int(populations)
        populations = pop_dict
    else:
        populations = {int(k): int(v) for k, v in populations.items()}

    # ── Modo incremental: filtrar CSV ao ano monitorado ──────────────
    _channel_state_path = Path(output_file).parent / 'channel_state.json'
    _incremental = skip_channel_estimation and _channel_state_path.exists()
    _mon_year = monitor_year or _ano_atual

    if _incremental:
        print(f"   Modo INCREMENTAL detectado (channel_state.json presente)")
        df_proc = df[df['ano_epi'] == _mon_year].copy()
        print(f"   Filtrando para {_mon_year}: {len(df_proc)} linhas (de {len(df)} total)")
    else:
        df_proc = df
    # ─────────────────────────────────────────────────────────────────

    print(f"[2/5] Agregando dados por agravo...")

    # Determinar agrupamentos
    results = {}

    # Sempre incluir "Todos"
    agg_all = aggregate_raw_data(df_proc, col_date, col_cid, col_qty, group_by='all')
    results.update(agg_all)

    if not use_desc_fallback:
        # Caminho normal: tem códigos CID → capítulos e SINAN
        if agravos in ('all', 'chapters') or (isinstance(agravos, str) and agravos.startswith('top')):
            agg_ch = aggregate_raw_data(df_proc, col_date, 'cid_codigo', col_qty, group_by='chapter')
            results.update(agg_ch)

        if agravos in ('all', 'sinan'):
            agg_sinan = aggregate_raw_data(df_proc, col_date, 'cid_codigo', col_qty, group_by='sinan')
            # Prefixar com "SINAN: " para identificação no dashboard
            agg_sinan = {f"SINAN: {k}": v for k, v in agg_sinan.items()}
            results.update(agg_sinan)

        if agravos == 'all' or (isinstance(agravos, str) and agravos.startswith('top')):
            n = 20
            if isinstance(agravos, str) and agravos.startswith('top_'):
                n = int(agravos.split('_')[1])

            # Em modo incremental, usar df completo (todos os anos) para selecionar
            # os top CIDs — garante que canais históricos (ex: A90 com pico em 2024)
            # não desaparecem em anos de baixo volume. A agregação dos dados continua
            # usando df_proc (ano monitorado apenas). Fix: bug A90 2026 = 0. [ekokubun]
            _df_for_topn = df if _incremental else df_proc
            top_cids = (volume_por(_df_for_topn, 'cid_codigo', col_qty)
                        .sort_values(ascending=False)
                        .head(n))

            # Tabela invertida code→desc para enriquecer nomes quando cid_descricao==cid_codigo
            _code_to_desc = {}
            for _d, _c in CID_DESC_TO_CODE.items():
                if _c not in _code_to_desc:
                    _code_to_desc[_c] = _d

            for cid_code in top_cids.index:
                if pd.isna(cid_code) or cid_code is None:
                    continue
                df_cid = df_proc[df_proc['cid_codigo'] == cid_code].copy()
                desc = df_cid[col_cid].mode().iloc[0] if len(df_cid) > 0 else cid_code
                # Se desc == cid_code (CSV histórico sem descrição), enriquecer via DATASUS
                if desc == cid_code and cid_code in _code_to_desc:
                    desc = _code_to_desc[cid_code]
                name = f"{cid_code} - {desc}" if cid_code != desc else cid_code
                agg = contar_casos(df_cid, col_qty)
                results[name] = agg

        # ── Síndromes e sentinelas de saúde coletiva ──────────────────
        if agravos == 'all' and 'cid_codigo' in df_proc.columns:
            for syn_name, syn_prefixes in SYNDROME_DEFS.items():
                mask = df_proc['cid_codigo'].apply(
                    lambda x: any(str(x).startswith(p) for p in syn_prefixes)
                    if pd.notna(x) else False
                )
                df_syn = df_proc[mask]
                if len(df_syn) > 0:
                    agg_syn = contar_casos(df_syn, col_qty)
                    if len(agg_syn) > 0 and agg_syn['casos'].sum() > 0:
                        results[syn_name] = agg_syn
            syn_ct = sum(1 for k in results if k in SYNDROME_DEFS)
            if syn_ct > 0:
                print(f"   → {syn_ct} síndromes/sentinelas computadas")

    else:
        # Fallback: poucos códigos CID → usar descrições + palavras-chave
        print(f"   Usando fallback por descrição...")

        # Capítulos via desc_to_chapter()
        if agravos in ('all', 'chapters') or (isinstance(agravos, str) and agravos.startswith('top')):
            if '_chapter_from_desc' in df_proc.columns:
                df_ch = df_proc[df_proc['_chapter_from_desc'].notna()].copy()
                if len(df_ch) > 0:
                    for ch, gdf in df_ch.groupby('_chapter_from_desc'):
                        agg_c = contar_casos(gdf, col_qty)
                        if len(agg_c) > 0:
                            results[str(ch)] = agg_c
                    print(f"   → {len([k for k in results if k not in ['Todos os atendimentos']])} capítulos gerados")

        # SINAN via palavras-chave na descrição
        if agravos in ('all', 'sinan'):
            # Mapear descrições para códigos CID via dicionário, depois checar SINAN
            if col_desc in df_proc.columns:
                df_proc['_sinan_from_desc'] = df_proc[col_desc].apply(
                    lambda d: cid_to_sinan(desc_to_cid_code(d)) if desc_to_cid_code(d) else 'Outros'
                )
                df_sinan = df_proc[df_proc['_sinan_from_desc'] != 'Outros']
                if len(df_sinan) > 0:
                    for sinan_name, gdf in df_sinan.groupby('_sinan_from_desc'):
                        agg_s = contar_casos(gdf, col_qty)
                        if len(agg_s) > 0:
                            results[f"SINAN: {sinan_name}"] = agg_s
                    sinan_count = len([k for k in results if k.startswith('SINAN:')])
                    print(f"   → {sinan_count} agravos SINAN detectados")

        # Top N descrições como CIDs individuais
        n = 30
        df_proc['_desc_clean'] = df_proc[col_desc].astype(str).str.strip()
        df_proc = df_proc[df_proc['_desc_clean'] != '']
        df_proc = df_proc[df_proc['_desc_clean'] != 'nan']

        top_descs = (volume_por(df_proc, '_desc_clean', col_qty)
                     .sort_values(ascending=False)
                     .head(n))

        print(f"   Top {len(top_descs)} descrições por volume:")
        for desc_name in top_descs.index:
            if pd.isna(desc_name) or not desc_name:
                continue
            # Tentar obter código CID para nome mais limpo
            cid_code = desc_to_cid_code(str(desc_name))
            display_name = f"{cid_code} - {desc_name}" if cid_code else str(desc_name)
            df_desc = df_proc[df_proc['_desc_clean'] == desc_name].copy()
            agg = contar_casos(df_desc, col_qty)
            if len(agg) > 0:
                results[display_name] = agg
                print(f"     {display_name}: {int(agg['casos'].sum())} atendimentos")

    print(f"   {len(results)} agravos/grupos identificados")

    # Verificar SE incompleta
    # ── Denominador dos canais de proporção ───────────────────────────
    # Todo agravo (menos o próprio total) passa a ser medido como FRAÇÃO dos
    # atendimentos da semana. Em modo incremental o total dos anos-base vem do
    # channel_state.json, porque df_proc só tem o ano monitorado.
    # O denominador é o total de LINHAS de CID da semana, não de atendimentos:
    # é o que põe numerador e denominador na mesma unidade e cancela a deriva.
    _denom = {}
    _lin = (df_proc.groupby(['ano_epi', 'semana_epi']).size()
            if len(df_proc) else pd.Series(dtype=int))
    for (_a, _s_), _v in _lin.items():
        if 1 <= int(_s_) <= MAX_SE:
            _denom[(int(_a), int(_s_))] = int(_v)
    if _incremental and _channel_state_path.exists():
        try:
            with open(_channel_state_path, encoding='utf-8') as _f:
                _st = json.load(_f)
            for _key, _v in (_st.get('denominador_linhas') or {}).items():
                _a_, _w_ = _key.split('-')
                _denom.setdefault((int(_a_), int(_w_)), int(_v))
        except Exception as _e:
            print(f"   ⚠ denominador histórico indisponível no state: {_e}")
    print(f"   denominador de proporção: {len(_denom)} pares (ano, SE)")

    print(f"[3/5] Verificando completude da última SE...")
    # A completude da última semana é propriedade da EXTRAÇÃO, não do agravo: ou a
    # fonte tem o sábado daquela SE, ou a semana está truncada para todos. O
    # critério é a data máxima do arquivo — a SE só está fechada se o dado alcança
    # o sábado (SE-SUS vai de domingo a sábado).
    #
    # Antes: detectar_se_incompleta() era chamada SEM col_data, e nesse caminho ela
    # soma 2 aos critérios "por não ter data" — com o terceiro critério (ratio>=0,5)
    # bastando para fechar em 3, ela NUNCA excluía nada. Era código morto. E quando
    # excluía, zerava os casos, o que é pior do que publicar: zero cai abaixo do p25
    # e a semana truncada sai classificada como 'sucesso'. Agora a semana incompleta
    # simplesmente não é publicada (ver se_max_observada).
    _dt_max = pd.to_datetime(df_proc[col_date], dayfirst=True, errors='coerce').max()
    _se_max_pub = 0
    if pd.notna(_dt_max):
        _ae_max, _se_dtmax = epi_week(_dt_max)
        _fechada = _dt_max.isoweekday() == 6            # 6 = sábado
        _se_max_pub = _se_dtmax if _fechada else _se_dtmax - 1
        if _ae_max != _mon_year:
            _se_max_pub = 0
        print(f"   última data: {_dt_max.date()} (SE {_se_dtmax}/{_ae_max}, "
              f"{'fechada' if _fechada else 'em curso'}) → publica até a SE {_se_max_pub}")

    info_se = {'ultima_se': _se_max_pub, 'ano': _mon_year, 'ano_atual': _mon_year,
               'completa': True, 'decisao': 'INCLUIR'}

    print(f"[4/5] Computando canais endêmicos (Gamma-Poisson)...")

    all_channels = {}

    if _incremental:
        # ── Modo incremental: sem MLE/MC ──────────────────────────────
        print(f"   Modo INCREMENTAL: carregando params congelados de channel_state.json...")
        with open(_channel_state_path, encoding='utf-8') as f:
            _state = json.load(f)
        _state_channels = _state.get('channels', {})

        # Índice de 2026: código CID → (nome, agg_df) para match robusto
        # O código é a parte antes do primeiro ' - ' (ex: 'A09', 'X - Aparelho respiratório')
        _results_by_code = {}
        for rname, rdf in results.items():
            code = rname.split(' - ')[0].strip()
            _results_by_code[code] = (rname, rdf)

        # 1. Reconstruir TODOS os canais históricos do state (preserva nome canônico)
        for state_name, state_ch in _state_channels.items():
            state_code = state_name.split(' - ')[0].strip()

            # Match 1: nome exato
            if state_name in results:
                agg_2026 = results[state_name][~results[state_name]['ano'].isin(EXCLUDED_YEARS)].copy()
                output_name = state_name
            # Match 2: código CID (robusto a drift de descrição)
            elif state_code in _results_by_code:
                result_name, agg_2026 = _results_by_code[state_code]
                agg_2026 = agg_2026[~agg_2026['ano'].isin(EXCLUDED_YEARS)].copy()
                # Preferir nome enriquecido quando state_name é código nu — o frontend
                # busca chaves enriquecidas ("A09 - DIARREIA...", "A90 - DENGUE...").
                # Revertido fix 2 (532e595) que quebrou todos os CIDs ao usar bare codes.
                output_name = result_name if (state_name == state_code and result_name != state_code) else state_name
            else:
                # Canal histórico sem dados em 2026 (c2026 = 0 para todas as SEs)
                agg_2026 = pd.DataFrame({'ano': pd.Series(dtype=int),
                                         'se':  pd.Series(dtype=int),
                                         'casos': pd.Series(dtype=int)})
                output_name = state_name

            ch = _rebuild_from_state(state_ch, agg_2026, populations, _mon_year,
                                     denominadores=_denom)
            if ch is not None:
                all_channels[output_name] = ch

        # 2. Adicionar canais genuinamente novos (em 2026 mas não no state)
        _state_codes = {sn.split(' - ')[0].strip() for sn in _state_channels}
        for rname, agg_df in results.items():
            rcode = rname.split(' - ')[0].strip()
            if rname not in all_channels and rcode not in _state_codes:
                print(f"   ⚡ Novo agravo '{rname}' — computando...")
                agg_df = agg_df[~agg_df['ano'].isin(EXCLUDED_YEARS)].copy()
                if len(agg_df) > 0:
                    ch = compute_endemic_channel(
                        agg_df, populations, agravo_name=rname,
                        leave_one_out=False, base_hist_years=BASE_HIST_YEARS,
                        use_mle=True, monitor_year=_mon_year,
                        denominadores=(None if rname == 'Todos os atendimentos'
                                       or not _denom or not USAR_PROPORCAO else _denom))
                    all_channels[rname] = ch

        print(f"   {len(all_channels)} canais atualizados (sem MLE/MC — params congelados)")

    else:
        # ── Modo completo: MLE + Monte Carlo ─────────────────────────
        for i, (name, agg_df) in enumerate(results.items()):
            # Filtrar anos excluídos (implantação / dados inconsistentes)
            agg_df = agg_df[~agg_df['ano'].isin(EXCLUDED_YEARS)].copy()
            years_available = sorted(agg_df['ano'].unique())
            if len(years_available) < 1:
                continue

            ch = compute_endemic_channel(
                agg_df, populations,
                agravo_name=name,
                leave_one_out=False,
                base_hist_years=BASE_HIST_YEARS,
                use_mle=True,
                monitor_year=monitor_year,
                denominadores=(None if name == 'Todos os atendimentos' or not _denom
                               or not USAR_PROPORCAO else _denom),
            )
            all_channels[name] = ch

            if (i + 1) % 5 == 0 or i == len(results) - 1:
                print(f"   {i+1}/{len(results)} agravos processados...")

        # Salvar params congelados para runs incrementais futuros
        _save_channel_state(all_channels, _channel_state_path, BASE_HIST_YEARS, _mon_year,
                            denominador=_denom)

    print(f"[5/5] Exportando JSON para {output_file}...")

    output = {
        'metadata': {
            'generated': pd.Timestamp.now().isoformat(),
            'model': 'Gamma-Poisson (hierárquico bayesiano)',
            'estimation': 'MLE com grid-search + Monte Carlo quantiles',
            'mc_samples': MC_SAMPLES,
            'quantiles': QUANTILES,
            'zones': ZONE_NAMES,
            'max_se': MAX_SE,
            'n_agravos': len(all_channels),
            'source': str(input_file),
            'base_hist_years': base_hist_years,
            'se_atual': info_se.get('ultima_se', 0),
            'ano_atual': info_se.get('ano_atual', monitor_year),
            # Última SE do ano monitorado que TEM dado. Sem isto a carga grava as
            # semanas que ainda não aconteceram com casos=0, e zero cai abaixo do
            # p25: setembro a dezembro do ano corrente entravam no banco como
            # 'sucesso' (1.713 linhas da UPA e 1.675 da APS em 2026).
            'se_max_observada': int(_se_max_pub),
            'ano_monitorado': int(_mon_year),
        },
        'channels': all_channels,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=None, separators=(',', ':'), cls=NumpyEncoder)

    size_kb = Path(output_file).stat().st_size / 1024
    print(f"\n✓ Concluído! {len(all_channels)} canais → {output_file} ({size_kb:.1f} KB)")

    return output


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Canal Endêmico Bayesiano Hierárquico (Gamma-Poisson)')
    parser.add_argument('input', help='CSV de entrada (dados brutos ou agregados)')
    parser.add_argument('--pop', required=True,
                        help='Populações por ano (JSON dict) ou valor único')
    parser.add_argument('--output', '-o', default='channel_data.json',
                        help='JSON de saída (default: channel_data.json)')
    parser.add_argument('--base-hist-years', default=None,
                        help='Anos históricos fixos separados por vírgula (ex: 2023,2024,2025)')
    parser.add_argument('--skip-channel-estimation', action='store_true',
                        help='Atualiza dados sem recalcular limiares dos canais (usa channel_data.json existente)')
    parser.add_argument('--agravos', default='all',
                        help='Agrupamento: all, chapters, sinan, top_N')
    parser.add_argument('--monitor-year', type=int, default=None,
                        help='Ano específico para monitorar')
    parser.add_argument('--col-date', default='data')
    parser.add_argument('--col-cid', default='cid_descricao')
    parser.add_argument('--col-qty', default='quantidade')

    args = parser.parse_args()

    # Parse populations
    try:
        pop = json.loads(args.pop)
        if isinstance(pop, (int, float)):
            pop = int(pop)
    except json.JSONDecodeError:
        pop = int(args.pop)

    # Processar --base-hist-years
    bhy = None
    if args.base_hist_years:
        bhy = [int(y.strip()) for y in args.base_hist_years.split(",")]

    run_pipeline(
        args.input, pop, args.output,
        agravos=args.agravos,
        col_date=args.col_date,
        col_cid=args.col_cid,
        col_qty=args.col_qty,
        monitor_year=args.monitor_year,
        base_hist_years=bhy,
        skip_channel_estimation=args.skip_channel_estimation,
    )
