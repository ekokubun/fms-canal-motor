"""
Testes de regressão do motor único do canal (_bootstrap_channel_se), decisão 2026-07-19.

Cobrem especificamente os dois bugs reais encontrados e corrigidos durante o desenvolvimento
(ver arquitetura/prototipo_motor_canal/RESULTADOS.md seção 4, no repo `pasta sem título`) —
não são testes genéricos, são regressão desses bugs específicos. Rodar com:

    pip install pytest
    pytest test_bootstrap_channel.py -v
"""
import numpy as np
import pytest

from fms_canal_motor.compute_channels import _bootstrap_channel_se, QUANTILES


def _rng(seed=1):
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────
# Bug 1 — MoM tem que ser em espaço de TAXA (contagem/exposição), não bruta.
# Ajustar direto em contagem com exposição grande (população) fazia o fallback de
# variância degenerada colapsar num b_hat sem relação de escala com a exposição.
# ─────────────────────────────────────────────────────────────────────────
class TestEspacoDeTaxa:
    def test_exposicao_grande_nao_explode_p90(self):
        """P90 nunca pode se aproximar da exposição (população) — sintoma direto do bug 1."""
        cases_train = [4057, 4091, 4056]  # baixa variância entre anos, caso real (SE31)
        exp_train = [210_000, 210_000, 210_000]
        e_mon = 210_000
        qs, a, b = _bootstrap_channel_se(cases_train, exp_train, e_mon, rng=_rng())
        assert qs[4] < e_mon * 0.1, (
            f"P90={qs[4]} está na escala da exposição ({e_mon}) — regressão do bug 1"
        )

    def test_p90_fica_proximo_da_escala_observada(self):
        """P90 deve ficar numa faixa razoável em torno do que foi observado, não em outra
        ordem de grandeza — mesmo com exposição bem maior que as contagens."""
        cases_train = [4057, 4091, 4056]
        exp_train = [210_000, 210_000, 210_000]
        qs, a, b = _bootstrap_channel_se(cases_train, exp_train, 210_000, rng=_rng())
        obs_max = max(cases_train)
        assert obs_max * 0.5 < qs[4] < obs_max * 3, (
            f"P90={qs[4]} fora da faixa razoável em torno do observado (max={obs_max})"
        )

    def test_exposicao_pequena_ainda_funciona(self):
        """Lado Santa Casa (exposição=1 implícito) precisa continuar funcionando — a
        correção do bug 1 não pode quebrar o caso de exposição pequena/unitária."""
        cases_train = [10, 12, 9]
        qs, a, b = _bootstrap_channel_se(cases_train, [1.0, 1.0, 1.0], 1.0, rng=_rng())
        assert qs[4] < 100, f"P90={qs[4]} implausível para contagens ~10 com exposure=1"


# ─────────────────────────────────────────────────────────────────────────
# Bug 2 — a_hat/b_hat nunca podem ser clipados independentemente depois de calculados.
# Isso quebra a razão a_hat/b_hat = m (taxa média), que trava a média da NB na média
# observada — chegou a produzir P90 igual à própria população num teste real.
# ─────────────────────────────────────────────────────────────────────────
class TestDerivacaoConsistente:
    def test_baixa_variancia_nao_produz_p90_igual_exposicao(self):
        """Caso real que expôs o bug: 3 anos quase idênticos, exposição grande.
        Antes da correção, produzia P90 = 210082 (a própria população)."""
        cases_train = [4057, 4091, 4056]
        qs, a, b = _bootstrap_channel_se(cases_train, [210_000] * 3, 210_000, rng=_rng())
        assert qs[4] != 210_000 and qs[4] < 50_000

    def test_variancia_zero_entre_anos_nao_quebra(self):
        """3 anos EXATAMENTE iguais — variância zero, caso extremo do bug 2."""
        cases_train = [4000, 4000, 4000]
        qs, a, b = _bootstrap_channel_se(cases_train, [210_000] * 3, 210_000, rng=_rng())
        assert 0 <= qs[0] <= qs[4] < 20_000

    @pytest.mark.parametrize("seed", [1, 2, 3, 42, 100])
    def test_multiplas_seeds_nao_produzem_outlier(self, seed):
        """A correção não pode depender de sorte de seed — testar várias."""
        cases_train = [4057, 4091, 4056]
        qs, a, b = _bootstrap_channel_se(cases_train, [210_000] * 3, 210_000, rng=_rng(seed))
        assert qs[4] < 20_000, f"seed={seed} produziu P90={qs[4]} fora do razoável"


# ─────────────────────────────────────────────────────────────────────────
# Propriedades gerais — não são sobre os bugs específicos, mas protegem contra
# regressões estruturais (ordem quebrada, formato errado, etc.)
# ─────────────────────────────────────────────────────────────────────────
class TestPropriedadesGerais:
    def test_ordem_dos_percentis_sempre_monotona(self):
        casos = [
            ([4057, 4091, 4056], 210_000),   # baixa variância
            ([100, 5000, 200], 210_000),      # alta variância
            ([0, 0, 0], 210_000),             # tudo zero
            ([1, 0, 2], 50_000),              # contagens muito baixas
            ([50000, 48000, 52000], 210_000), # contagens quase iguais à exposição/4
        ]
        for cases_train, pop in casos:
            qs, a, b = _bootstrap_channel_se(cases_train, [pop] * 3, pop, rng=_rng())
            assert qs == sorted(qs), f"ordem quebrada pra {cases_train}: {qs}"
            assert all(v >= 0 for v in qs), f"valor negativo pra {cases_train}: {qs}"

    def test_todos_zero_retorna_canal_zero(self):
        qs, a, b = _bootstrap_channel_se([0, 0, 0], [210_000] * 3, 210_000, rng=_rng())
        assert qs == [0, 0, 0, 0, 0]

    def test_retorna_5_quantis(self):
        qs, a, b = _bootstrap_channel_se([100, 120, 90], [210_000] * 3, 210_000, rng=_rng())
        assert len(qs) == len(QUANTILES) == 5

    def test_reprodutibilidade_mesma_seed(self):
        """Mesma seed -> mesmo resultado (determinismo, sem o qual paridade não seria testável)."""
        args = ([4057, 4091, 4056], [210_000] * 3, 210_000)
        qs1, _, _ = _bootstrap_channel_se(*args, rng=_rng(7))
        qs2, _, _ = _bootstrap_channel_se(*args, rng=_rng(7))
        assert qs1 == qs2

    def test_mais_exposicao_no_ano_monitorado_aumenta_percentis_proporcionalmente(self):
        """Se e_mon dobra (ex.: população cresceu), os percentis devem escalar
        aproximadamente proporcional — sanity check da separação exposição-de-ajuste vs.
        exposição-de-predição."""
        cases_train = [100, 120, 90]
        exp_train = [100_000] * 3
        qs_normal, _, _ = _bootstrap_channel_se(cases_train, exp_train, 100_000, rng=_rng(5))
        qs_dobro, _, _ = _bootstrap_channel_se(cases_train, exp_train, 200_000, rng=_rng(5))
        assert qs_dobro[4] > qs_normal[4] * 1.5, (
            f"P90 não escalou com e_mon: normal={qs_normal[4]}, dobro={qs_dobro[4]}"
        )

    def test_alta_variancia_produz_banda_mais_larga_que_baixa_variancia(self):
        """Propriedade básica de sanidade estatística: mais variabilidade histórica ->
        banda mais larga."""
        pop = 210_000
        qs_baixa, _, _ = _bootstrap_channel_se([4057, 4091, 4056], [pop] * 3, pop, rng=_rng())
        qs_alta, _, _ = _bootstrap_channel_se([2000, 6000, 4000], [pop] * 3, pop, rng=_rng())
        largura_baixa = qs_baixa[4] - qs_baixa[0]
        largura_alta = qs_alta[4] - qs_alta[0]
        assert largura_alta > largura_baixa
