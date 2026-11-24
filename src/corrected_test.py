"""Teste t pareado corrigido para partições de validação cruzada repetida.

Dois modelos avaliados nas mesmas J partições geram diferenças
d_j = E_CV(a, j) - E_CV(b, j). As partições compartilham boa parte do treino
(com 5 dobras, 3200 de 4000 linhas), então essas diferenças não são
independentes e o teste t comum, com variância s²/J, subestima a incerteza.
A correção de Nadeau e Bengio (2003) usa

    t = média(d) / sqrt((1/J + n_validação/n_treino) * s²),

comparado a uma t de Student com J - 1 graus de liberdade. Com J = 25 e
n_validação/n_treino = 1/4, o erro padrão fica cerca de 2,7 vezes maior que no
teste comum. A correção é aproximada e não protege contra o otimismo de
escolher a melhor entre muitas variantes. Explicação completa e exemplo passo a
passo: notebooks/10_testes_finais.ipynb, seção 3.

Executar este módulo recalcula, a partir dos CSVs por partição em
results/rodada_atual, os testes da comparação final de onze atributos.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "rodada_atual"
TEST_TRAIN_RATIO = 1 / 4  # 5 folds: 800 linhas de validação para 3200 de treino


def corrected_paired_ttest(a, b, test_train_ratio=TEST_TRAIN_RATIO, level=0.95):
    """Compara os RMSE por partição de dois modelos avaliados nas mesmas partições.

    Retorna a diferença média (a - b), o intervalo de confiança e o p-valor bilateral.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = len(d)
    se = np.sqrt((1 / j + test_train_ratio) * d.var(ddof=1))
    t = d.mean() / se
    half = stats.t.ppf((1 + level) / 2, j - 1) * se
    return {
        "diferença média": d.mean(),
        "IC inferior": d.mean() - half,
        "IC superior": d.mean() + half,
        "t": t,
        "p-valor": 2 * stats.t.sf(abs(t), j - 1),
        "vitórias de a": int((d < 0).sum()),
        "partições": j,
    }


def by_partition(path, model):
    frame = pd.read_csv(path)
    return frame[frame["modelo"] == model].sort_values("partição")["E_CV"].to_numpy()


def main():
    eleven = RESULTS / "testes_contagens_por_particao.csv"
    nine = RESULTS / "comparacao_final_por_particao.csv"
    comparisons = {
        "Boosting 11 atributos - regressão 11 atributos": (
            by_partition(eleven, "Boosting"), by_partition(eleven, "Regressão comum")),
        "Boosting 11 atributos - boosting 9 atributos": (
            by_partition(eleven, "Boosting"), by_partition(nine, "Boosting")),
    }
    result = pd.DataFrame(
        [{"comparação": name, **corrected_paired_ttest(a, b)} for name, (a, b) in comparisons.items()]
    )
    result.to_csv(RESULTS / "testes_corrigidos.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
