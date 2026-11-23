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

**Notação.** Na validação cruzada, $E_{in}$ é o RMSE no próprio treino de cada partição e $E_{CV}$ é o RMSE na dobra de validação; ambos são médias nas 25 partições. $E_{CV}$ estima o erro fora da amostra, $E_{out}$. O **RMSE OOF repetido** junta todas as previsões fora da dobra (cinco por linha) num único RMSE e é o número principal das tabelas. O gap $E_{CV} - E_{in}$ indica quanto o modelo se adapta ao treino. Para comparar dois modelos nas mesmas partições, usamos o **teste t pareado corrigido** de Nadeau e Bengio, que leva em conta que as partições compartilham dados de treino (explicação e exemplo no notebook 10, seção 3; código em `src/corrected_test.py`).

| Modelo | RMSE |
|---|---|
| Previsão constante (média de treino) | ≈ 1,19 |
| Regressão linear nos atributos brutos | ≈ 0,84 |
| Oráculo (conhece o mecanismo gerador; piso teórico) | ≈ 0,39 |

Esses valores são referências históricas do problema, não métricas da rodada atual. A diferença até o oráculo não é uma promessa de ganho recuperável com os atributos disponíveis; o oráculo conhece o mecanismo gerador.

## Linha investigativa

Partimos de regressões simples para testar uma mudança por vez: duas médias, afinidade, curvatura e gênero. Depois comparamos procedimentos usando os mesmos seis atributos: regressão comum, Ridge, Bayesian Ridge e boosting. A investigação é guiada pela EDA e pelos resíduos; seus resultados são exploratórios, não confirmação independente.

Em seguida testamos idade, ano e as duas contagens, chegando a onze atributos construídos a partir dos dez atributos brutos. Por fim, investigamos por que o boosting supera a regressão: a interação idade × ano de lançamento (hipótese de nostalgia) explica boa parte da vantagem. O boosting com os onze atributos mais a **idade no lançamento** é o modelo final.

## Estrutura do repositório

```
data/          dados (não versionados; ver data/README.md)
notebooks/     análises exploratórias e experimentos
src/           código reutilizável: validação, atributos, testes e submissão
results/       resultados em JSON das execuções completas
submissions/   previsões geradas para o conjunto de teste
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

Para reproduzir os testes finais, a submissão e as análises complementares:

```bash
python -m src.final_feature_tests
python -m src.count_feature_test
python -m src.ridge_sensitivity
python -m src.reference_linear
python -m src.corrected_test
python -m src.nostalgia_test
python -m src.nostalgia_checks
python -m src.final_submission
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
| **Boosting + idade + ano linear e quadrático** | **0,565157** |
| Regressão + contagens em log | 0,615210 |
| Ridge + contagens em log | 0,615210 |
| **Boosting + contagens em log** | **0,544933** |

A construção de atributos trouxe os maiores ganhos. “Seis” e “nove” contam atributos construídos, enquanto “dez” conta os atributos brutos do projeto. A receita de onze atributos usa a informação de todos eles: o ano gera termos linear e quadrático, e as duas contagens entram como `log1p`. O boosting chegou a 0,544933, venceu nas 25 partições e nas cinco repetições e superou a regressão em cerca de 0,070 RMSE. Seu gap $E_{CV} - E_{in}$ foi 0,113212, contra 0,002581 nos modelos lineares; há maior risco de sobreajuste, mas a vantagem fora do treino foi ampla e estável.

### Hipótese de nostalgia e modelo final

Restringindo as interações permitidas ao boosting, a que mais reduz o erro é **idade × ano de lançamento**. Usuários dão notas mais altas a filmes lançados quando tinham entre 10 e 20 anos (≈ 4,0 estrelas, contra ≈ 3,2 nas demais faixas). A idade no lançamento é `age − (2024 − release_year)`: **tomamos como pressuposto que as idades são relativas a 2024**, o último ano de lançamento dos dados, porque o enunciado não informa o ano da medição. Com validação aninhada, nas mesmas 25 partições:

| Modelo | RMSE OOF repetido | Gap $E_{CV} - E_{in}$ |
|---|---:|---:|
| Regressão, 11 atributos | 0,615210 | 0,003 |
| Boosting, 11 atributos | 0,544933 | 0,113 |
| Regressão, 11 atributos + pico de nostalgia | 0,541704 | 0,003 |
| **Boosting, 11 atributos + idade no lançamento (modelo final)** | **0,519532** | 0,079 |

Com o atributo de nostalgia, a regressão empata com o boosting de onze atributos (p = 0,50 no teste t corrigido): a vantagem do boosting vinha, em grande parte, de aprender essa interação. Com a idade no lançamento, o boosting vence os outros três modelos nas 25 partições (p ≈ 3 × 10⁻⁷ contra o boosting de onze atributos e p ≈ 2 × 10⁻⁵ contra a regressão com o pico).

Antes de adotá-lo, verificamos que o resultado não depende do pressuposto de 2024 (com 2026 ou 2000, as previsões são idênticas) nem da borda da grade de hiperparâmetros (com árvores menores disponíveis, as escolhas não mudam). A hipótese surgiu depois de explorar os mesmos dados, o que é registrado como limitação. Ver [notebook 11](notebooks/11_hipotese_nostalgia.ipynb), `src/nostalgia_test.py` e `src/nostalgia_checks.py`.

## Modelo e submissão final

O modelo final é o **HistGradientBoostingRegressor com doze atributos construídos**: duas médias, afinidade, afinidade², dois indicadores de gênero, idade, ano linear, ano quadrático, os logaritmos das duas contagens e a idade no lançamento. Seu RMSE OOF repetido é **0,519532**. O script `python -m src.final_submission` refaz a seleção interna, ajusta a receita nas 4000 linhas e gera `submissions/submission_final.csv` com 2000 previsões válidas e cortadas em [0, 5].

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
| [10 Testes finais](notebooks/10_testes_finais.ipynb) | Idade, ano e contagens; teste de significância | Executado, receita de onze atributos |
| [11 Nostalgia](notebooks/11_hipotese_nostalgia.ipynb) | Explicar a vantagem do boosting; verificações; modelo final e submissão | Executado, modelo final |

As 25 partições externas são iguais para todos os modelos. A média das cinco previsões por observação serve apenas aos gráficos; não é a previsão pontuada. Ridge e boosting selecionam hiperparâmetros em três folds internos. Bayesian Ridge estima as precisões no treino com hiperparâmetros dos priors fixados.

Os notebooks 03–08 são independentes e gravam previsões por repetição, erros, configurações e identificadores de dados/código via `src/current_results.py`. A síntese 09 verifica compatibilidade e reprodução das referências entre notebooks antes de comparar. Execute 03–08 antes de 09. Os resultados estão em `results/rodada_atual`.

As partições compartilham treino; variação e contagens de vitórias são descritivas. A busca aninhada protege a escolha de hiperparâmetros dentro de cada procedimento, mas a seleção entre famílias e a exploração de atributos continuam sujeitas a otimismo.

Idade, ano e contagens foram testados em alterações controladas. A receita vencedora usa informação dos dez atributos brutos em onze atributos construídos. Essa receita foi congelada e gerou a primeira submissão. A análise da nostalgia (notebook 11) veio depois, para explicar a vantagem do boosting, e levou ao modelo final atual.

Os notebooks 01 e 02 preservam a exploração inicial e o protocolo anterior (5 × 3); `src/validation.py` é usado apenas pelo notebook 02.

## Materiais de estudo e próximos passos

- [Apresentação visual dos modelos](docs/modelos-bauhaus.html): slides em HTML com exemplos didáticos animados e controles interativos. Abra o arquivo no navegador.
- [Próximos passos e plano de estudo](docs/proximos-passos.md): sequência dos experimentos, conceitos prioritários e revisão dos notebooks com a voz e a compreensão do grupo.
- [Rascunho da apresentação](docs/apresentacao-cinemateca.html): ainda usa o protocolo antigo e tem slides de resultados com valores fictícios; precisa ser atualizado antes da apresentação.

## Referências

- Y. Abu-Mostafa, M. Magdon-Ismail e H.-T. Lin. *Learning from Data*. AMLBook, 2012.
- G. James, D. Witten, T. Hastie e R. Tibshirani. *An Introduction to Statistical Learning*, 2ª ed., 2023.
