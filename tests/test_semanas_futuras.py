"""v0.3.7: semana do ano monitorado que ainda não fechou não tem observado nem zona,
e a faixa segue até a SE 52. Até a v0.3.6 essas semanas saíam com casos = 0 (a semana
em curso, com o valor parcial) e zona 'sucesso', e a faixa da família proporção era
zero porque dependia do denominador de uma semana que não existia.
"""
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fms_canal_motor.compute_channels as cc
import fms_canal_motor.carga_postgres as cp

OK = {"shape": 10.0, "rate": 500.0}
ZERO = {"shape": 0.0, "rate": 0.0}


def _state():
    # SE 3 e 4 ainda não tinham acontecido em 2026 no último recompute completo.
    return {
        "agravo": "X", "familia": "proporcao", "se_list": [1, 2, 3, 4],
        "params": {"2025": [OK] * 4, "2026": [OK, OK, ZERO, ZERO]},
        "channels": {"2025": [[1, 2, 3, 4, 5]] * 4,
                     "2026": [[1, 2, 3, 4, 5]] * 2 + [[0.0] * 5] * 2},
        "raw_hist": [{"se": s, "c2025": 20} for s in (1, 2, 3, 4)],
    }


# 2026: SE 1-2 fechadas com 10% mais atendimentos que 2025; SE 3 em curso (parcial).
DEN = {(2025, 1): 1000, (2025, 2): 1000, (2025, 3): 1000, (2025, 4): 1000,
       (2026, 1): 1100, (2026, 2): 1100, (2026, 3): 300}
OBS = pd.DataFrame({"ano": [2026, 2026, 2026], "se": [1, 2, 3], "casos": [20, 30, 5]})


def _canal():
    ch = cc._rebuild_from_state(_state(), OBS, {2025: 1, 2026: 1}, 2026, denominadores=DEN)
    antes = copy.deepcopy(ch)
    cc._sem_avaliacao_futura(ch, 2026, 2, DEN, [2025])
    return antes, ch


def test_semana_futura_e_em_curso_sem_observado_nem_zona():
    _, ch = _canal()
    assert ch["raw"][0]["c2026"] == 20 and ch["raw"][1]["c2026"] == 30
    assert "c2026" not in ch["raw"][2]          # em curso: o parcial (5) não aparece
    assert "c2026" not in ch["raw"][3]
    assert ch["classifications"]["2026"][2:] == [None, None]
    assert ch["exceedance"]["2026"][2:] == [None, None]


def test_faixa_da_proporcao_vai_ate_o_fim_com_denominador_projetado():
    _, ch = _canal()
    for i in (2, 3):
        q = ch["channels"]["2026"][i]
        assert q[4] > 0
        # n projetado = 1000 x 1,10 = 1100; média beta 10/510 -> mediana ~21-22
        assert 18 <= q[2] <= 26
        assert q[0] <= q[1] <= q[2] <= q[3] <= q[4]


def test_semana_fechada_e_anos_historicos_nao_mudam():
    antes, ch = _canal()
    assert ch["channels"]["2025"] == antes["channels"]["2025"]
    assert ch["classifications"]["2025"] == antes["classifications"]["2025"]
    assert ch["channels"]["2026"][:2] == antes["channels"]["2026"][:2]
    assert ch["classifications"]["2026"][:2] == antes["classifications"]["2026"][:2]
    assert ch["params"] == antes["params"]


def test_kpis_so_com_semanas_fechadas():
    _, ch = _canal()
    k = ch["kpis"]["2026"]
    assert k["total"] == 50                     # não 55 (sem o parcial da SE 3)
    assert (k["pico"], k["pico_se"]) == (30, 2)


def test_contagem_mantem_faixa_e_tira_zona():
    ch = {"familia": "contagem", "se_list": [1, 2, 3],
          "raw": [{"se": 1, "c2026": 7}, {"se": 2, "c2026": 0}, {"se": 3, "c2026": 0}],
          "channels": {"2026": [[1, 2, 3, 4, 5]] * 3},
          "classifications": {"2026": ["alerta", "sucesso", "sucesso"]},
          "exceedance": {"2026": [1.4, 0.0, 0.0]},
          "kpis": {"2026": {"total": 7, "pico": 7, "pico_se": 1, "se_acima_p90": 1}}}
    cc._sem_avaliacao_futura(ch, 2026, 1)
    assert ch["channels"]["2026"] == [[1, 2, 3, 4, 5]] * 3
    assert ch["classifications"]["2026"] == ["alerta", None, None]
    assert [("c2026" in r) for r in ch["raw"]] == [True, False, False]
    assert ch["kpis"]["2026"] == {"total": 7, "pico": 7, "pico_se": 1, "se_acima_p90": 1}


def test_projecao_do_denominador_segue_o_nivel_do_ano():
    den = {(2024, 1): 900, (2025, 1): 1100, (2024, 2): 500, (2025, 2): 700, (2026, 1): 2000}
    # mediana SE1 = 1000, ano corrente = 2x -> SE2: mediana 600 x 2
    assert cc._projeta_denominador(den, 2026, 1, [2024, 2025], [1, 2]) == {2: 1200}


def test_boletim_ultima_se_pelo_se_list_esparso():
    # Canal raro do SINAN: só as SE com caso na base. A posição na lista não é SE - 1.
    import fms_canal_motor.pipeline as pl
    ch = {"se_list": [3, 20, 40], "years": [2025, 2026],
          "raw": [{"se": 3, "c2025": 1, "c2026": 2}, {"se": 20, "c2025": 0, "c2026": 1},
                  {"se": 40, "c2025": 1}],
          "classifications": {"2025": ["alerta", "sucesso", "alerta"],
                              "2026": ["alerta", "epidemico", None]},
          "channels": {"2026": [[0, 0, 1, 2, 3], [0, 0, 1, 2, 4], [0, 0, 1, 2, 5]]}}
    item = next(b for b in pl.step4_boletim({"channels": {"SINAN: Sífilis NE": ch}})
                if b["name"] == "SINAN: Sífilis NE")
    assert item["se_2026"] == 20
    assert item["ultima_se_zona"] == "epidemico"
    assert (item["ultima_se_obs"], item["ultima_se_p90"]) == (1, 4)


def test_carga_por_faixa_etaria_nao_grava_semana_futura():
    age = {"X": {"0-4": {
        "channels": {"1": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5},
                     "3": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5}},
        "raw": {2025: {"1": 4, "3": 6}, 2026: {"1": 5, "3": 2}},
        "classifications": {2025: {"1": "alerta", "3": "alerta"},
                            2026: {"1": "alerta", "3": "sucesso"}}}}}
    obs, canal, clf, metas = [], [], [], {}
    cp.rows_from_age_inmem(age, obs, canal, clf, metas,
                           meta={"ano_monitorado": 2026, "se_max_observada": 2})
    assert (2026, 3) not in {(o[2], o[3]) for o in obs}
    assert (2026, 3) not in {(c[2], c[3]) for c in clf}
    assert (2025, 3) in {(o[2], o[3]) for o in obs}
    assert (2026, 1) in {(c[2], c[3]) for c in clf}
    # sem metadata (JSON antigo) nada é cortado
    obs2, clf2 = [], []
    cp.rows_from_age_inmem(age, obs2, [], clf2, {})
    assert len(obs2) == 4 and len(clf2) == 4
