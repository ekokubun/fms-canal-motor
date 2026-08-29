"""Testes do que mudou no motor em 2026-08-29 (v0.3.x): contagem por atendimento,
lista de agravos sem zona epidemiológica e janela de SE na estimação.

Ficam num arquivo próprio porque `compute_channels` puxa pandas/scipy — os testes
de `carga_postgres` são de propósito leves e não carregam o motor.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fms_canal_motor.compute_channels as cc


# ── sem_zona_epidemica ───────────────────────────────────────────────────────

@pytest.mark.parametrize("nome", [
    "Todos os atendimentos",
    "Z000 - EXAME MEDICO GERAL",
    "Z10 - EXAME GERAL DE ROTINA",
    "Z34 - SUPERVISAO DE GRAVIDEZ NORMAL",
    "Z532 - PROCEDIMENTO NAO REALIZADO",
    "Z760 - EMISSAO DE PRESCRICAO DE REPETICAO",
    "E10 - DIABETES INSULINO-DEPENDENTE",
    "E11 - DIABETES NAO-INSULINO-DEPENDENTE",
    "E149 - DIABETES NAO ESPECIFICADO",
    "E78 - OUTRAS LIPIDEMIAS",
    "XXI - Fatores que influenciam o estado de saúde",
])
def test_sem_zona(nome):
    assert cc.sem_zona_epidemica(nome)


@pytest.mark.parametrize("nome", [
    "X - Aparelho respiratório",
    "I - Doenças infecciosas e parasitárias",
    "SINAN: Dengue",
    "A09 - DIARREIA E GASTROENTERITE",
    "J00 - NASOFARINGITE AGUDA",
    "M545 - DOR LOMBAR BAIXA",
    "E86 - DEPLECAO DE VOLUME",
    "Dor Osteomuscular",
])
def test_com_zona(nome):
    assert not cc.sem_zona_epidemica(nome)


def test_e149_nao_escapa():
    """Regressão: o regex usava E1[014] seguido de \\b, e o \\b falha diante do
    dígito — E149, E119 e E109 recebiam zona quando não deviam."""
    for n in ("E109 - x", "E119 - x", "E149 - x"):
        assert cc.sem_zona_epidemica(n)


# ── contar_casos ─────────────────────────────────────────────────────────────

def _df():
    """Duas semanas; na SE 1 o mesmo atendimento carrega 3 CID."""
    return pd.DataFrame([
        {"ano_epi": 2026, "semana_epi": 1, "quantidade": 1, "atend_id": "a"},
        {"ano_epi": 2026, "semana_epi": 1, "quantidade": 1, "atend_id": "a"},
        {"ano_epi": 2026, "semana_epi": 1, "quantidade": 1, "atend_id": "a"},
        {"ano_epi": 2026, "semana_epi": 2, "quantidade": 1, "atend_id": "b"},
    ])


def test_dedup_conta_atendimentos():
    r = cc.contar_casos(_df(), dedup=True).set_index("se").casos
    assert r[1] == 1 and r[2] == 1


def test_sem_dedup_conta_linhas():
    """O numerador dos canais de agravo é LINHA, para casar com o denominador
    (também linha) do canal de proporção — é assim que a deriva se cancela."""
    r = cc.contar_casos(_df(), dedup=False).set_index("se").casos
    assert r[1] == 3 and r[2] == 1


def test_csv_antigo_sem_atend_id_ainda_funciona():
    df = _df().drop(columns=["atend_id"])
    r = cc.contar_casos(df, dedup=True).set_index("se").casos
    assert r[1] == 3          # cai na soma de quantidade, sem quebrar


def test_se_acima_do_maximo_e_descartada():
    df = pd.DataFrame([{"ano_epi": 2026, "semana_epi": cc.MAX_SE + 1,
                        "quantidade": 1, "atend_id": "z"}])
    assert cc.contar_casos(df).empty


# ── canal de proporção ───────────────────────────────────────────────────────

def test_betabinom_faixa_cresce_com_o_denominador():
    """A faixa é sobre a FRAÇÃO: com o mesmo p, um denominador maior dá um p90
    maior em contagem absoluta."""
    k = [10, 12, 11]
    n = [1000, 1000, 1000]
    qs_p, _, _ = cc._betabinom_channel_se(k, n, 1000)
    qs_g, _, _ = cc._betabinom_channel_se(k, n, 2000)
    assert qs_g[4] > qs_p[4]


def test_betabinom_denominador_zero_nao_quebra():
    assert cc._betabinom_channel_se([1, 2], [10, 10], 0) == ([0.0] * 5, 0.0, 0.0)


def test_betabinom_sem_casos_devolve_zeros():
    assert cc._betabinom_channel_se([0, 0], [100, 100], 100) == ([0.0] * 5, 0.0, 0.0)


# ── janela de SE ─────────────────────────────────────────────────────────────

def test_janela_configurada():
    """Com 3 anos-base a SE isolada dá 3 observações e a dispersão não é
    identificável: o p90 da APS ficava 9% acima do p50. A janela de ±2 leva a
    15 observações e a faixa para ~23%."""
    assert cc.JANELA_SE >= 1
