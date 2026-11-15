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

A rodada atual foi executada. Não adicionamos automaticamente redes, vizinhos ou outras árvores: os resíduos agora motivam discutir idade e ano de lançamento antes de abrir novas famílias.

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

## Resultados da rodada atual

5 folds × 5 repetições, semente 2026; RMSE sobre todas as previsões OOF separadamente, com corte em [0, 5]. Valores em estrelas. O teste não foi usado.

| Modelo | RMSE OOF repetido |
|---|---:|
| Constante | 1,165913 |
| Só média do usuário | 1,038440 |
| Só média do filme | 0,995403 |
| Duas médias | 0,839575 |
| Duas médias + afinidade | 0,660065 |
| + afinidade² | 0,628989 |
| + gênero | 0,625036 |
| Ridge com os mesmos seis atributos | 0,625036 |
| Bayesian Ridge com os mesmos seis atributos | 0,625074 |
| Boosting com os mesmos seis atributos | 0,619248 |

A construção de atributos trouxe os maiores ganhos. Boosting melhorou em 20/25 partições e nas cinco repetições, mas a redução sobre regressão com gênero foi modesta (0,005788). Seu RMSE médio de treino foi 0,565527, contra 0,623377 da regressão: existe maior adaptação ao treino, não prova de ausência de sobreajuste. Regularização não trouxe ganho material. Nenhum modelo final foi escolhido.

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

As 25 partições externas são iguais para todos os modelos. A média das cinco previsões por observação serve apenas aos gráficos; não é a previsão pontuada. Ridge e boosting selecionam hiperparâmetros em três folds internos. Bayesian Ridge estima as precisões no treino com hiperparâmetros dos priors fixados.

Os notebooks 03–08 são independentes e gravam previsões por repetição, erros, configurações e identificadores de dados/código via `src/current_results.py`. A síntese 09 verifica compatibilidade e reprodução das referências entre notebooks antes de comparar. Execute 03–08 antes de 09. Os resultados estão em `results/rodada_atual`; não se misturam aos JSONs antigos.

As partições compartilham treino; variação e contagens de vitórias são descritivas. A busca aninhada protege a escolha de hiperparâmetros dentro de cada procedimento, mas a seleção entre famílias e a exploração de atributos continuam sujeitas a otimismo.

**Próxima decisão:** discutir testes separados de idade e ano de lançamento, pois ambos exibem padrões nos resíduos de regressão e boosting. Ainda não implementados. Se alterarem os atributos escolhidos, refazer a comparação entre procedimentos com a mesma informação. Redes, vizinhos e outras árvores permanecem opcionais, não uma lista obrigatória.

Os notebooks em `notebooks/antigos`, `src/experiments.py`, `src/validation.py` e os resultados fora de `results/rodada_atual` preservam o roteiro antigo. O dashboard e o notebook 11 são materiais auxiliares, não definem o protocolo nem substituem a síntese atual.

## Materiais de estudo e próximos passos

- [Apresentação visual dos modelos](docs/modelos-bauhaus.html): slides em HTML com exemplos didáticos animados e controles interativos. Abra o arquivo no navegador.
- [Próximos passos e plano de estudo](docs/proximos-passos.md): sequência dos experimentos, conceitos prioritários e revisão dos notebooks com a voz e a compreensão do grupo.

## Referências

- Y. Abu-Mostafa, M. Magdon-Ismail e H.-T. Lin. *Learning from Data*. AMLBook, 2012.
- G. James, D. Witten, T. Hastie e R. Tibshirani. *An Introduction to Statistical Learning*, 2ª ed., 2023.
