"""
Testes de step4_boletim / build_age_channels_compact / _zona_label (pipeline.py) --
atomização, continuação da suíte de bootstrap_thresholds_age.

Escopo deliberadamente limitado: cobre as funções PURAS (sem I/O) de pipeline.py.
step1 (subprocess), step2 (pandas+import dinâmico de compute_channels.py por
caminho relativo -- funciona só com cwd correto), step5/step6 (geração de HTML/
DOCX, exigem fixture pesada de template/matplotlib/python-docx) e
_gerar_grafico_canal ficam FORA desta suíte -- registrado como lacuna em
contrato_apresentacao.md, não como esquecimento.

    pip install pytest
    pytest test_pipeline_steps.py -v
"""
import pytest

from fms_canal_motor.pipeline import step4_boletim, build_age_channels_compact, _zona_label


# ── step4_boletim ────────────────────────────────────────────────────────────

def _channel_data_dengue():
    return {
        "channels": {
            "SINAN: Dengue": {
                "se_list": [1, 2, 3],
                "years": [2022, 2023, 2024, 2025, 2026],
                "raw": [
                    {"se": 1, "c2022": 5, "c2023": 6, "c2024": 7, "c2025": 20, "c2026": 25},
                    {"se": 2, "c2022": 4, "c2023": 5, "c2024": 6, "c2025": 10, "c2026": 15},
                    {"se": 3, "c2022": 3, "c2023": 4, "c2024": 5, "c2025": 5, "c2026": 0},
                ],
                "classifications": {
                    "2025": ["emergencia", "alerta", "sucesso"],
                    "2026": ["epidemico", "alerta"],
                },
                "channels": {
                    "2026": [[1, 2, 3, 4, 5], [2, 3, 4, 6, 8]],
                },
            }
        }
    }


def test_step4_boletim_calcula_totais_e_variacao():
    boletim = step4_boletim(_channel_data_dengue())
    assert len(boletim) == 1
    b = boletim[0]
    assert b["name"] == "SINAN: Dengue"
    assert b["prioridade"] == "ALTA"
    assert b["total_2025"] == 35   # 20+10+5
    assert b["total_2026"] == 40   # 25+15+0
    assert b["media_hist"] == 15   # média de [12,15,18] (soma por ano 2022-2024)
    assert b["variacao_pct"] == 133.3
    assert "Aumento de 133.3%" in b["tendencia"]


def test_step4_boletim_ultima_se_com_dado_e_zona_correta():
    boletim = step4_boletim(_channel_data_dengue())
    b = boletim[0]
    # última SE com c2026>0 é a SE 2 (SE 3 tem c2026=0)
    assert b["se_2026"] == 2
    assert b["ultima_se_obs"] == 15
    assert b["ultima_se_zona"] == "alerta"   # classifications['2026'][1]
    assert b["ultima_se_p90"] == 8           # channels['2026'][1][4]
    assert b["ultima_se_p50"] == 4           # channels['2026'][1][2]


def test_step4_boletim_pico_e_sazonalidade():
    boletim = step4_boletim(_channel_data_dengue())
    b = boletim[0]
    assert b["pico_val_2025"] == 20
    assert b["pico_se_2025"] == 1
    assert b["sazonalidade"] == "Pico na SE 1."
    assert b["pico_val_2026"] == 25
    assert b["pico_se_2026"] == 1


def test_step4_boletim_zone_counts():
    boletim = step4_boletim(_channel_data_dengue())
    b = boletim[0]
    assert b["classificacao_2025"] == {
        "sucesso": 1, "seguranca": 0, "alerta": 1, "epidemico": 0, "emergencia": 1,
    }
    assert b["classificacao_2026"] == {
        "sucesso": 0, "seguranca": 0, "alerta": 1, "epidemico": 1, "emergencia": 0,
    }


def test_step4_boletim_agravo_ausente_no_channel_data_e_pulado():
    """Nenhum dos priority_agravos existe -> boletim vazio, sem erro."""
    boletim = step4_boletim({"channels": {"Um agravo qualquer": {}}})
    assert boletim == []


def test_step4_boletim_find_channel_case_insensitive_fuzzy():
    """find_channel casa por substring case-insensitive -- CID em caixa diferente
    da lista de priority_agravos deve casar mesmo assim."""
    cd = _channel_data_dengue()
    cd["channels"]["sinan: dengue (confirmado)"] = cd["channels"].pop("SINAN: Dengue")
    boletim = step4_boletim(cd)
    assert len(boletim) == 1
    assert boletim[0]["name"] == "sinan: dengue (confirmado)"


# ── build_age_channels_compact ───────────────────────────────────────────────

def test_build_age_channels_compact_shape():
    ac_data = {
        "Todos os atendimentos": {
            "18-39 anos": {
                "channels": {"1": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5}},
                "raw": {"2023": {"1": 7}},
            }
        }
    }
    compact = build_age_channels_compact(ac_data)
    entry = compact["Todos os atendimentos"]["18-39 anos"]
    assert entry["years"] == [2023]
    assert entry["se_list"] == list(range(1, 53))
    assert len(entry["channels"]["2023"]) == 52
    assert entry["channels"]["2023"][0] == [1, 2, 3, 4, 5]  # SE 1
    assert entry["channels"]["2023"][1] == [0, 0, 0, 0, 0]  # SE 2 sem dado -> zero
    assert entry["raw"][0]["c2023"] == 7
    assert entry["raw"][1]["c2023"] == 0


def test_build_age_channels_compact_multiplos_anos_ordenados():
    ac_data = {
        "X": {
            "Y": {
                "channels": {},
                "raw": {"2024": {}, "2022": {}, "2023": {}},
            }
        }
    }
    compact = build_age_channels_compact(ac_data)
    assert compact["X"]["Y"]["years"] == [2022, 2023, 2024]  # ordenado, não a ordem de inserção


# ── _zona_label ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("zona,esperado", [
    ("sucesso", ("SUCESSO", "1A7942")),
    ("seguranca", ("SEGURANÇA", "2E75B6")),
    ("alerta", ("ALERTA", "F5A623")),
    ("epidemico", ("EPIDÊMICO", "D62828")),
    ("emergencia", ("EMERGÊNCIA", "7B0D8E")),
    ("sem dados", ("SEM DADOS", "AAAAAA")),
])
def test_zona_label_mapeamento(zona, esperado):
    assert _zona_label(zona) == esperado


def test_zona_label_desconhecida_cai_no_fallback():
    assert _zona_label("zona_que_nao_existe") == ("?", "AAAAAA")
