"""
Testes de regressão de bootstrap_thresholds_age() (pipeline.py), decisão 2026-07-19.

Achado no mesmo dia em que o motor principal (_bootstrap_channel_se, ver
test_bootstrap_channel.py) foi corrigido: step3_age_channels() tinha sua PRÓPRIA
cópia do motor antigo (grid-search+MC), sujeita ao mesmo risco de
não-identificabilidade do parâmetro de dispersão -- potencialmente pior aqui,
já que contagem por faixa etária é subconjunto do total (amostras menores).
bootstrap_thresholds_age() é a mesma solução (bootstrap sobre anos-base), sem
termo de exposição (este código nunca normalizou por população).

    pip install pytest
    pytest test_bootstrap_thresholds_age.py -v
"""
import numpy as np
import pytest

from fms_canal_motor.pipeline import bootstrap_thresholds_age, QUANTILES


def _rng(seed=1):
    return np.random.default_rng(seed)


def test_ordem_monotona_p10_a_p90():
    train = [3, 5, 4]
    qs = bootstrap_thresholds_age(train, rng=_rng())
    assert qs == sorted(qs)


def test_determinismo_mesma_seed_mesmo_resultado():
    train = [10, 15, 8, 12, 20]
    qs1 = bootstrap_thresholds_age(train, rng=_rng(42))
    qs2 = bootstrap_thresholds_age(train, rng=_rng(42))
    assert qs1 == qs2


def test_retorna_5_quantis():
    qs = bootstrap_thresholds_age([1, 2, 3], rng=_rng())
    assert len(qs) == len(QUANTILES) == 5


def test_anos_identicos_variancia_zero_usa_fallback():
    """Todos os anos-base com a mesma contagem -> toda reamostragem é degenerada
    (variância zero) -> cai no fallback de média/variância mínima, não deve
    explodir nem quebrar (bug 2 do motor principal: a_hat/b_hat clipados
    independentemente já causou p90 = população inteira num caso real)."""
    train = [7, 7, 7]
    qs = bootstrap_thresholds_age(train, rng=_rng())
    assert qs == sorted(qs)
    assert qs[4] < 10_000  # não deve explodir


def test_um_ano_so_variancia_sempre_zero():
    qs = bootstrap_thresholds_age([12], rng=_rng())
    assert qs == sorted(qs)
    assert qs[4] < 10_000


def test_contagens_maiores_produzem_limiares_maiores():
    """Não é um teste de valor exato (é estocástico) -- é teste de direção: mais
    casos nos anos-base -> canal desloca pra cima. Mesma seed pros dois pra
    isolar o efeito da escala, não do ruído do bootstrap."""
    baixo = bootstrap_thresholds_age([2, 3, 2], rng=_rng(7))
    alto = bootstrap_thresholds_age([200, 250, 220], rng=_rng(7))
    assert alto[2] > baixo[2]  # mediana


def test_variando_apenas_seed_muda_resultado_mas_mesma_ordem_de_grandeza():
    train = [5, 8, 6, 9]
    qs_a = bootstrap_thresholds_age(train, rng=_rng(1))
    qs_b = bootstrap_thresholds_age(train, rng=_rng(2))
    # bootstrap é estocástico -- seeds diferentes podem (não precisam) diferir,
    # mas nunca devem divergir em ordem de grandeza nem quebrar a ordenação.
    assert qs_a == sorted(qs_a)
    assert qs_b == sorted(qs_b)
    assert max(qs_a[4], qs_b[4]) < 100  # train é pequeno, p90 não deveria disparar
