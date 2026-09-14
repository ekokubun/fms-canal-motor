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


# ── se_publicavel ────────────────────────────────────────────────────────────
# A regra de completude olha os dias em que a FONTE opera, não o sábado do
# calendário. Exigir sábado deixava o canal da APS uma semana atrasado para
# sempre, porque a Atenção Básica não abre sábado.

def _uteis(ini, fim):
    return [d for d in pd.date_range(ini, fim) if d.dayofweek < 5]


def test_aps_semana_fechada_na_sexta():
    """SE 34 de 2026 (24-28/08, seg-sex) está completa e tem de sair."""
    se, diag = cc.se_publicavel(_uteis("2026-07-20", "2026-08-28"), 2026)
    assert se == 34, diag


def test_aps_semana_em_curso_nao_sai():
    se, _ = cc.se_publicavel(_uteis("2026-07-20", "2026-08-26"), 2026)
    assert se == 33


def test_upa_fecha_no_sabado():
    se, _ = cc.se_publicavel(pd.date_range("2026-07-19", "2026-08-29"), 2026)
    assert se == 34


def test_upa_sexta_ainda_em_curso():
    """A UPA atende 7 dias: sexta não fecha a semana dela."""
    se, _ = cc.se_publicavel(pd.date_range("2026-07-19", "2026-08-28"), 2026)
    assert se == 33


def test_upa_domingo_da_semana_seguinte():
    se, _ = cc.se_publicavel(pd.date_range("2026-07-19", "2026-08-30"), 2026)
    assert se == 34


def test_sem_datas():
    se, diag = cc.se_publicavel([], 2026)
    assert se == 0 and "nada a publicar" in diag


def test_ano_diferente_do_monitorado_nao_publica():
    se, _ = cc.se_publicavel(pd.date_range("2025-07-19", "2025-08-29"), 2026)
    assert se == 0


# ── feriado (v0.3.8) ─────────────────────────────────────────────────────────
# Dia útil que ficou para trás sem dado não chega mais e não segura a semana.

def _sem(datas, *tirar):
    fora = set(pd.to_datetime(list(tirar)))
    return [d for d in datas if d not in fora]


def test_aps_segunda_feriado_fecha_na_sexta():
    """SE 36 de 2026: segunda 07/09 feriado, ter-sex entregues. Na v0.3.7 saía 35
    e o boletim, pelo calendário, rotulou a tabela da 35 como SE 36."""
    se, diag = cc.se_publicavel(_sem(_uteis("2026-07-20", "2026-09-11"), "2026-09-07"), 2026)
    assert se == 36, diag
    assert "seg, que ficou para trás" in diag


def test_aps_segunda_feriado_semana_ainda_em_curso():
    """Quarta chegou, quinta e sexta ainda podem chegar: não fecha."""
    se, _ = cc.se_publicavel(_sem(_uteis("2026-07-20", "2026-09-09"), "2026-09-07"), 2026)
    assert se == 35


def test_carnaval_seg_e_ter():
    """SE 7 de 2026: 16 e 17/02 sem atendimento na APS."""
    se, _ = cc.se_publicavel(_sem(_uteis("2026-01-05", "2026-02-20"),
                                  "2026-02-16", "2026-02-17"), 2026)
    assert se == 7


def test_sexta_feriado_continua_segurando():
    """Sexta de feriado é indistinguível de sexta não extraída: segura (lado
    seguro) até o dado da semana seguinte."""
    se, _ = cc.se_publicavel(_uteis("2026-07-20", "2026-09-10"), 2026)
    assert se == 35
    se, _ = cc.se_publicavel(_uteis("2026-07-20", "2026-09-10") + [pd.Timestamp("2026-09-14")], 2026)
    assert se == 36


def test_upa_domingo_ausente_nao_vira_pendente():
    """Semana epidemiológica começa no domingo. Sem a posição epidemiológica o
    domingo (dayofweek 6) pareceria vir depois do sábado (5) e seguraria a SE."""
    se, _ = cc.se_publicavel(_sem(pd.date_range("2026-07-19", "2026-09-12"), "2026-09-06"), 2026)
    assert se == 36


def test_upa_sabado_ainda_pode_chegar():
    se, _ = cc.se_publicavel(_sem(pd.date_range("2026-07-19", "2026-09-11"), "2026-09-07"), 2026)
    assert se == 35
