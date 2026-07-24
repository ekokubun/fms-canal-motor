"""Testes de carga_postgres.py -- atomização, item 'teste' do critério de pronto
(contrato+idempotente+teste+sem PII+roda isolado).

compute_channels/pipeline só são importados DENTRO de obter_recompute() (import
local, não de módulo) -- então classify_agravo/_intq/rows_from_*/obter_from_json/
resumo_dry_run/gravar_postgres são importáveis e testáveis SEM precisar do motor
nem de scipy/pandas carregados. gravar_postgres é testado com psycopg2.connect
mockado (sem banco real -- ao contrário de agregar-canonico, aqui ESCREVE dados,
não é seguro rodar contra produção mesmo em modo leitura).
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fms_canal_motor.carga_postgres as CP


# ── classify_agravo ───────────────────────────────────────────────────────────

def test_classify_agravo_todos():
    assert CP.classify_agravo("Todos os atendimentos") == ("todos", None, None)


def test_classify_agravo_sinan():
    assert CP.classify_agravo("SINAN: Dengue") == ("sinan", None, None)


def test_classify_agravo_capitulo_romano_valido():
    nome = "I - Algumas doenças infecciosas e parasitárias"
    assert CP.classify_agravo(nome) == ("capitulo", None, nome)


def test_classify_agravo_romano_fora_do_range_cai_em_sindrome():
    # "XXV" não está em ROMANOS (só até XXII) -- não é capítulo válido do CID-10
    assert CP.classify_agravo("XXV - Não é capítulo real") == ("sindrome", None, None)


def test_classify_agravo_cid_sem_decimal():
    assert CP.classify_agravo("J45 - Asma") == ("cid", "J45", None)


def test_classify_agravo_cid_com_decimal():
    assert CP.classify_agravo("J45.0 - Asma com exacerbação") == ("cid", "J45.0", None)


def test_classify_agravo_sindrome_fallback():
    assert CP.classify_agravo("Síndrome Gripal") == ("sindrome", None, None)


# ── _intq ──────────────────────────────────────────────────────────────────────

def test_intq_valido():
    assert CP._intq([10, 20, 30], 1) == 20


def test_intq_indice_fora_do_range():
    assert CP._intq([10, 20], 5) is None


def test_intq_valor_none():
    assert CP._intq([None, 20], 0) is None


def test_intq_valor_nao_numerico():
    assert CP._intq(["abc"], 0) is None


def test_intq_trunca_float():
    assert CP._intq([1.9], 0) == 1


# ── rows_from_channel_data ───────────────────────────────────────────────────

def test_rows_from_channel_data_shapes_correctly():
    channel_data = {
        "channels": {
            "Todos os atendimentos": {
                "se_list": [1],
                "raw": [{"se": 1, "c2023": 10, "c2024": 12}],
                "channels": {"2023": [[1, 2, 3, 4, 5]]},
                "params": {"2023": [{"shape": 1.1, "rate": 2.2}]},
                "classifications": {"2023": ["sucesso"]},
                "exceedance": {"2023": [0.5]},
            }
        }
    }
    obs, canal, clf, metas = [], [], [], {}
    CP.rows_from_channel_data(channel_data, obs, canal, clf, metas)

    assert metas == {"Todos os atendimentos": ("todos", None, None)}
    assert sorted(obs) == sorted([
        ("Todos os atendimentos", "Todas", 2023, 1, 10),
        ("Todos os atendimentos", "Todas", 2024, 1, 12),
    ])
    assert canal == [("Todos os atendimentos", "Todas", 2023, 1, 1, 2, 3, 4, 5, 1.1, 2.2)]
    assert clf == [("Todos os atendimentos", "Todas", 2023, 1, "sucesso", 0.5)]


def test_rows_from_channel_data_sem_params_usa_none():
    channel_data = {
        "channels": {
            "SINAN: Dengue": {
                "se_list": [1],
                "raw": [{"se": 1, "c2024": 3}],
                "channels": {"2024": [[1, 2, 3, 4, 5]]},
                "params": {},
                "classifications": {},
                "exceedance": {},
            }
        }
    }
    obs, canal, clf, metas = [], [], [], {}
    CP.rows_from_channel_data(channel_data, obs, canal, clf, metas)
    assert metas == {"SINAN: Dengue": ("sinan", None, None)}
    assert canal == [("SINAN: Dengue", "Todas", 2024, 1, 1, 2, 3, 4, 5, None, None)]
    assert clf == []


# ── rows_from_age_inmem / rows_from_age_compact ─────────────────────────────

def test_rows_from_age_inmem():
    age_results = {
        "Todos os atendimentos": {
            "18-39 anos": {
                "raw": {"2023": {"1": 5}},
                "channels": {"1": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5}},
                "classifications": {"2023": {"1": "alerta"}},
            }
        }
    }
    obs, canal, clf, metas = [], [], [], {}
    CP.rows_from_age_inmem(age_results, obs, canal, clf, metas)
    assert obs == [("Todos os atendimentos", "18-39 anos", 2023, 1, 5)]
    assert canal == [("Todos os atendimentos", "18-39 anos", 2023, 1, 1, 2, 3, 4, 5, None, None)]
    assert clf == [("Todos os atendimentos", "18-39 anos", 2023, 1, "alerta", None)]


def test_rows_from_age_compact():
    age_compact = {
        "Todos os atendimentos": {
            "18-39 anos": {
                "se_list": [1],
                "raw": [{"se": 1, "c2023": 5}],
                "channels": {"2023": [[1, 2, 3, 4, 5]]},
            }
        }
    }
    obs, canal, metas = [], [], {}
    CP.rows_from_age_compact(age_compact, obs, canal, metas)
    assert obs == [("Todos os atendimentos", "18-39 anos", 2023, 1, 5)]
    assert canal == [("Todos os atendimentos", "18-39 anos", 2023, 1, 1, 2, 3, 4, 5, None, None)]


# ── obter_from_json ───────────────────────────────────────────────────────────

def test_obter_from_json_completo(tmp_path):
    (tmp_path / "channel_data.json").write_text(json.dumps({"channels": {}, "metadata": {"x": 1}}))
    (tmp_path / "age_channels.json").write_text(json.dumps({"y": 2}))
    (tmp_path / "boletim_data.json").write_text(json.dumps([{"name": "A", "prioridade": "alta"}]))

    channel_data, age_compact, prioridades, is_compact = CP.obter_from_json(str(tmp_path))
    assert channel_data == {"channels": {}, "metadata": {"x": 1}}
    assert age_compact == {"y": 2}
    assert prioridades == {"A": "alta"}
    assert is_compact is True


def test_obter_from_json_sem_age_nem_boletim(tmp_path):
    (tmp_path / "channel_data.json").write_text(json.dumps({"channels": {}}))
    channel_data, age_compact, prioridades, is_compact = CP.obter_from_json(str(tmp_path))
    assert age_compact == {}
    assert prioridades == {}


# ── resumo_dry_run ────────────────────────────────────────────────────────────

def test_resumo_dry_run_reporta_contagens(capsys):
    metas = {"Todos os atendimentos": ("todos", None, None)}
    obs = [("Todos os atendimentos", "Todas", 2026, 18, 10)]
    canal = [("Todos os atendimentos", "Todas", 2026, 18, 1, 2, 3, 4, 5, None, None)]
    clf = [("Todos os atendimentos", "Todas", 2026, 18, "sucesso", 0.5)]
    CP.resumo_dry_run(metas, obs, canal, clf)
    out = capsys.readouterr().out
    assert "1 agravos" in out
    assert "todas as zonas válidas" in out


def test_resumo_dry_run_sinaliza_zona_invalida(capsys):
    metas = {"X": ("sindrome", None, None)}
    obs = [("X", "Todas", 2026, 1, 1)]
    clf = [("X", "Todas", 2026, 1, "ZONA_QUEBRADA", None)]
    CP.resumo_dry_run(metas, obs, [], clf)
    out = capsys.readouterr().out
    assert "ZONAS INVÁLIDAS" in out
    assert "ZONA_QUEBRADA" in out


# ── gravar_postgres (psycopg2 mockado -- escreve dado, não roda contra produção) ──

def _fake_pg(fetchall_return):
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchall.return_value = fetchall_return
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    conn.cursor.return_value = cur
    return conn, cur


def test_gravar_postgres_filtra_zona_invalida_e_seta_schema():
    conn, cur = _fake_pg(fetchall_return=[(1, "Todos os atendimentos")])
    ev_calls = []

    def fake_execute_values(cur_, sql, values, **kw):
        ev_calls.append((sql, values))

    metas = {"Todos os atendimentos": ("todos", None, None)}
    obs = [("Todos os atendimentos", "Todas", 2026, 1, 10)]
    canal = [("Todos os atendimentos", "Todas", 2026, 1, 1, 2, 3, 4, 5, 1.1, 2.2)]
    clf = [
        ("Todos os atendimentos", "Todas", 2026, 1, "sucesso", 0.5),
        ("Todos os atendimentos", "Todas", 2026, 2, "ZONA_INVALIDA", 9.9),
    ]

    with patch("psycopg2.connect", return_value=conn), \
         patch("psycopg2.extras.execute_values", side_effect=fake_execute_values):
        CP.gravar_postgres("rio_claro", "upa_187", metas, {}, obs, canal, clf,
                            {"model": "x", "modo": "recompute"})

    assert len(ev_calls) == 4  # dim_agravo, fato_observacao_se, canal_endemico, classificacao_se
    clf_sql, clf_values = ev_calls[3]
    assert "classificacao_se" in clf_sql
    assert len(clf_values) == 1  # zona inválida filtrada antes do INSERT
    assert clf_values[0][5] == "sucesso"

    set_calls = [c for c in cur.execute.call_args_list if c.args and "SET search_path" in c.args[0]]
    assert set_calls and set_calls[0].args[0] == "SET search_path TO rio_claro"

    delete_calls = [c for c in cur.execute.call_args_list if c.args and "DELETE FROM" in c.args[0]]
    assert len(delete_calls) == 3  # classificacao_se, canal_endemico, fato_observacao_se


def test_gravar_postgres_fecha_conexao_mesmo_com_erro():
    conn, cur = _fake_pg(fetchall_return=[])
    conn.__enter__.side_effect = RuntimeError("falha simulada dentro da transação")

    with patch("psycopg2.connect", return_value=conn), \
         patch("psycopg2.extras.execute_values"):
        try:
            CP.gravar_postgres("rio_claro", "upa_187", {}, {}, [], [], [], {})
        except RuntimeError:
            pass

    conn.close.assert_called_once()
