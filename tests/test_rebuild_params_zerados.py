"""Regressão da v0.3.3: parâmetros do ano monitorado zerados nas SE que ainda não
tinham acontecido no último recompute completo não podem zerar o limiar dos outros
anos, nem o do próprio ano quando a SE chega. Até a v0.3.2 zeravam: todo caso
virava 'emergencia' (visto em 2026-09-10 na UPA e na APS).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fms_canal_motor.compute_channels as cc

VAZIO = pd.DataFrame({"ano": pd.Series(dtype=int), "se": pd.Series(dtype=int),
                      "casos": pd.Series(dtype=int)})


def _state():
    ok = {"shape": 10.0, "rate": 500.0}
    zero = {"shape": 0.0, "rate": 0.0}          # SE 2 ainda não tinha acontecido em 2026
    return {
        "agravo": "X", "familia": "proporcao", "se_list": [1, 2],
        "params": {"2025": [ok, ok], "2026": [ok, zero]},
        "channels": {"2025": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                     "2026": [[1, 2, 3, 4, 5], [0.0, 0.0, 0.0, 0.0, 0.0]]},
        "raw_hist": [{"se": 1, "c2025": 20}, {"se": 2, "c2025": 20}],
    }


def test_ano_historico_nao_herda_limiar_zero():
    den = {(2025, 1): 1000, (2025, 2): 1000, (2026, 1): 1000}
    r = cc._rebuild_from_state(_state(), VAZIO, {2025: 1, 2026: 1}, 2026, denominadores=den)
    assert r["channels"]["2025"][1][3] > 0
    # 20 em 1000 com média beta 10/510 (~19,6) não é emergência
    assert r["classifications"]["2025"][1] != "emergencia"


def test_ano_corrente_quando_a_se_chega():
    den = {(2025, 1): 1000, (2025, 2): 1000, (2026, 1): 1000, (2026, 2): 1000}
    obs = pd.DataFrame({"ano": [2026, 2026], "se": [1, 2], "casos": [20, 20]})
    r = cc._rebuild_from_state(_state(), obs, {2025: 1, 2026: 1}, 2026, denominadores=den)
    assert r["channels"]["2026"][1][3] > 0
    assert r["classifications"]["2026"][1] != "emergencia"
