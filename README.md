# Prevendo avaliações de filmes

Este repositório tenta prever **quantas estrelas (de 0 a 5) um usuário dá a um filme** numa plataforma de streaming fictícia. A previsão usa apenas estatísticas-resumo do usuário e do filme. Não há identificadores nem histórico completo.

O trabalho foi desenvolvido como projeto da disciplina MAE0501 – Aprendizagem Estatística (IME–USP, 2026). A ideia central é usar o problema para estudar, na prática, **generalização, validação e o equilíbrio entre viés e variância**.

## O problema

Cada linha é um par (usuário, filme) descrito por dez atributos:

| Grupo | Atributo | Descrição |
|---|---|---|
| Usuário | `user_avg_stars` | média das notas que o usuário deu a outros filmes |
| | `user_n_ratings` | quantas notas essa média resume |
| | `age` | idade (16–80) |
| | `share_action`, `share_comedy`, `share_drama` | proporção de cada gênero no histórico do usuário (somam 1) |
| Filme | `movie_avg_stars` | média das notas que outros usuários deram ao filme |
| | `movie_n_ratings` | quantas notas essa média resume |
| | `genre` | `action`, `comedy` ou `drama` |
| | `release_year` | ano de lançamento (1970–2024) |
| **Alvo** | `stars` | estrelas dadas pelo usuário ao filme (0–5) |

Os dados são sintéticos: 4000 pares de treino e 2000 de teste, sem valores faltantes.

### Métrica e referências

O desempenho é medido pela **raiz do erro quadrático médio (RMSE)**. As previsões são números reais, cortados no intervalo [0, 5].

| Modelo | RMSE |
|---|---|
| Previsão constante (média de treino) | ≈ 1,19 |
| Regressão linear nos atributos brutos | ≈ 0,84 |
| Oráculo (conhece o mecanismo gerador; piso teórico) | ≈ 0,39 |

Esses valores são referências históricas do problema, não métricas da rodada atual. A diferença até o oráculo não é uma promessa de ganho recuperável com os atributos disponíveis; o oráculo conhece o mecanismo gerador.

## Linha investigativa

Partimos de regressões simples para testar uma mudança por vez: duas médias, afinidade, curvatura e gênero. Depois comparamos procedimentos usando os mesmos seis atributos: regressão comum, Ridge, Bayesian Ridge e boosting. A investigação é guiada pela EDA e pelos resíduos; seus resultados são exploratórios, não confirmação independente.

A rodada atual foi concluída com testes de idade, ano e das duas contagens. Os dados têm dez atributos brutos; depois das transformações, eles são representados por onze colunas numéricas. O boosting com essa representação completa foi escolhido como modelo final.

## Estrutura do repositório

```
data/          dados (não versionados; ver data/README.md)
notebooks/     análises exploratórias e experimentos
src/           código reutilizável: validação, atributos, receitas e experimentos
results/       resultados em JSON das execuções completas
submissions/   previsões geradas para o conjunto de teste
report/        relatório final
docs/          documentos de organização do projeto
```

## Como reproduzir

Requer Python 3.9 ou superior.

```bash
git clone https://github.com/Robootic/movie-rating-prediction.git
cd movie-rating-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Coloque os arquivos `train.csv` e `test.csv` em `data/` e abra os notebooks com `jupyter lab`.

Para reproduzir a análise de sensibilidade, a submissão final e o relatório:

```bash
python -m src.ridge_sensitivity
python -m src.final_feature_tests
python -m src.count_feature_test
python -m src.final_submission
python -m src.reference_linear
python -m src.corrected_test
python -m src.nostalgia_test
python report/build_report.py
```

## Resultados da rodada atual

5 folds × 5 repetições, semente 2026; RMSE sobre todas as previsões OOF separadamente, com corte em [0, 5]. Valores em estrelas. O teste não foi usado.

| Modelo | RMSE OOF repetido |
|---|---:|
| Constante | 1,165913 |
| Regressão nos dez atributos brutos (referência do enunciado) | 0,824064 |
| Só média do usuário | 1,038440 |
| Só média do filme | 0,995403 |
| Duas médias | 0,839575 |
| Duas médias + afinidade | 0,660065 |
| + afinidade² | 0,628989 |
| + gênero | 0,625036 |
| Ridge com os mesmos seis atributos | 0,625036 |
| Bayesian Ridge com os mesmos seis atributos | 0,625074 |
| Boosting com os mesmos seis atributos | 0,619248 |
| Regressão + idade + ano linear e quadrático | 0,616080 |
| Ridge + idade + ano linear e quadrático | 0,616081 |
| **Boosting + idade + ano linear e quadrático** | **0,564180** |
| Regressão + contagens em log | 0,615210 |
| Ridge + contagens em log | 0,615210 |
| **Boosting + contagens em log** | **0,544851** |

A construção de atributos trouxe os maiores ganhos. “Seis” e “nove” contam colunas construídas, enquanto “dez” conta os atributos brutos do projeto. A receita final usa todos eles em onze colunas: o ano gera termos linear e quadrático, e as duas contagens entram como `log1p`. O boosting chegou a 0,544851, venceu nas 25 partições e nas cinco repetições e superou a regressão em cerca de 0,070 RMSE. Seu gap treino–validação foi 0,113995, contra 0,002581 nos modelos lineares; há maior risco de sobreajuste, mas a vantagem fora do treino foi ampla e estável.

### Hipótese de nostalgia (exploratória)

Restringindo as interações permitidas ao boosting, a que mais reduz o erro é **idade × ano de lançamento**. Usuários dão notas mais altas a filmes lançados quando tinham entre 10 e 20 anos (≈ 4,0 estrelas, contra ≈ 3,2 nas demais faixas). Com validação aninhada, nas mesmas 25 partições:

| Modelo | RMSE OOF repetido | Gap treino–validação |
|---|---:|---:|
| Regressão, 11 colunas | 0,615210 | 0,003 |
| Boosting, 11 colunas | 0,544933 | 0,113 |
| Regressão, 11 colunas + pico de nostalgia | 0,541704 | 0,003 |
| **Boosting, 11 colunas + idade no lançamento** | **0,519532** | 0,079 |

Com a coluna de nostalgia, a regressão empata com o boosting de onze colunas (p = 0,50 no teste t corrigido). A vantagem do boosting vinha, em grande parte, de aprender essa interação. Com a idade no lançamento, o boosting melhora em todas as 25 partições. A hipótese surgiu depois de explorar os mesmos dados; ver [notebook 10](notebooks/10_hipotese_nostalgia.ipynb) e `src/nostalgia_test.py`.

## Modelo e submissão final

O modelo final é o **HistGradientBoostingRegressor com informação dos dez atributos brutos**, representados por onze colunas: duas médias, afinidade, afinidade², dois indicadores de gênero, idade, ano linear, ano quadrático e os logaritmos das duas contagens. O script `python -m src.final_submission` refaz a seleção interna, ajusta a receita nas 4000 linhas e gera `submissions/submission_final.csv` com 2000 previsões válidas e cortadas em [0, 5].

O relatório final de três páginas está em [`output/pdf/relatorio_final.pdf`](output/pdf/relatorio_final.pdf). Sua fonte reproduzível está em `report/build_report.py`.
A versão LaTeX pronta para importar no Overleaf está em [`report/relatorio_overleaf.zip`](report/relatorio_overleaf.zip).

## Roteiro e estado dos notebooks

| Notebook | Papel | Estado |
|---|---|---|
| [01 EDA](notebooks/01_eda.ipynb) | Exploração que motivou hipóteses | Histórico, preservar limites das conclusões |
| [02 Validação](notebooks/02_validacao.ipynb) | Discussão do protocolo anterior 5 × 3 | Histórico, não define a rodada atual |
| [03 Duas médias](notebooks/03_hipotese_duas_medias.ipynb) | Testar informação complementar | Executado, leitura para discussão |
| [04 Afinidade](notebooks/04_hipotese_afinidade.ipynb) | Acrescentar afinidade | Executado, leitura para discussão |
| [05 Curvatura](notebooks/05_hipotese_curvatura.ipynb) | Acrescentar afinidade² | Executado, leitura para discussão |
| [06 Gênero](notebooks/06_hipotese_genero.ipynb) | Testar deslocamentos por gênero | Executado, ganho modesto |
| [07 Boosting](notebooks/07_comparacao_boosting.ipynb) | Comparar flexibilidade com informação igual | Executado, busca interna de três folds |
| [08 Regularização](notebooks/08_hipotese_regularizacao_bayesiana.ipynb) | Comparar regressão comum, Ridge e Bayesian Ridge | Executado, sem ganho material |
| [09 Síntese](notebooks/09_sintese_investigacao.ipynb) | Consolidar e diagnosticar erros restantes | Executado, duas hipóteses propostas |
| [10 Nostalgia](notebooks/10_hipotese_nostalgia.ipynb) | Explicar a vantagem do boosting pela interação idade × ano | Executado, exploratório |

As 25 partições externas são iguais para todos os modelos. A média das cinco previsões por observação serve apenas aos gráficos; não é a previsão pontuada. Ridge e boosting selecionam hiperparâmetros em três folds internos. Bayesian Ridge estima as precisões no treino com hiperparâmetros dos priors fixados.

Os notebooks 03–08 são independentes e gravam previsões por repetição, erros, configurações e identificadores de dados/código via `src/current_results.py`. A síntese 09 verifica compatibilidade e reprodução das referências entre notebooks antes de comparar. Execute 03–08 antes de 09. Os resultados estão em `results/rodada_atual`; não se misturam aos JSONs antigos.

As partições compartilham treino; variação e contagens de vitórias são descritivas. A busca aninhada protege a escolha de hiperparâmetros dentro de cada procedimento, mas a seleção entre famílias e a exploração de atributos continuam sujeitas a otimismo.

Idade, ano e contagens foram testados em alterações controladas. A receita vencedora usa informação dos dez atributos brutos em onze colunas numéricas. A exploração foi encerrada e a receita final foi congelada antes da submissão.

Os notebooks em `notebooks/antigos`, `src/experiments.py`, `src/validation.py` e os resultados fora de `results/rodada_atual` preservam o roteiro antigo. O dashboard e o notebook 11 são materiais auxiliares, não definem o protocolo nem substituem a síntese atual.

## Materiais de estudo e próximos passos

- [Apresentação visual dos modelos](docs/modelos-bauhaus.html): slides em HTML com exemplos didáticos animados e controles interativos. Abra o arquivo no navegador.
- [Próximos passos e plano de estudo](docs/proximos-passos.md): sequência dos experimentos, conceitos prioritários e revisão dos notebooks com a voz e a compreensão do grupo.

## Referências

- Y. Abu-Mostafa, M. Magdon-Ismail e H.-T. Lin. *Learning from Data*. AMLBook, 2012.
- G. James, D. Witten, T. Hastie e R. Tibshirani. *An Introduction to Statistical Learning*, 2ª ed., 2023.
