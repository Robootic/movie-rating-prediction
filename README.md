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

A distância entre a regressão linear e o oráculo mostra o quanto ainda há de estrutura a ser capturada. O que fica abaixo de 0,39 é ruído irredutível.

## Abordagem

1. **Análise exploratória:** investigar relações não lineares e interações. Por exemplo, a afinidade entre os gêneros que o usuário costuma assistir e o gênero do filme, e o quanto confiar numa média que resume poucas notas.
2. **Protocolo de validação:** usar validação cruzada k-fold com as mesmas dobras para todos os modelos. O conjunto de teste é usado só na previsão final.
3. **Modelos lineares com engenharia de atributos:** codificação de gênero, encolhimento (*shrinkage*) das médias, interações e splines, com regularização Ridge e Lasso.
4. **Modelos não lineares:** k-vizinhos mais próximos, árvores e seus ensembles, SVM e redes neurais. Também uma formulação ordinal que prevê o valor esperado das estrelas.
5. **Comparação e escolha do modelo:** decidir com base no erro de validação e interpretar o resultado à luz da teoria da generalização.

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

## Resultados

*Em andamento.* A tabela com o RMSE de validação de cada modelo será adicionada aqui.

## Andamento do projeto

Cada etapa tem um notebook que explica o que foi feito, por que e o que foi encontrado. Esta tabela resume as etapas e aponta para cada notebook.

| Etapa | Status | Notebook | O que foi feito / principais achados |
|---|---|---|---|
| 1. Análise exploratória | ✅ Concluída | [01_eda](notebooks/01_eda.ipynb) | Reproduzimos a referência linear (RMSE de CV ≈ 0,83). O achado principal é a **afinidade de gênero**: a proporção do gênero do filme no histórico do usuário é uma interação que a linear bruta não enxerga, e só ela leva o RMSE a ≈ 0,66. Também encontramos curvatura nessa relação, efeito de teto/piso do alvo em [0, 5] e colinearidade perfeita entre as `share_*`. |
| 2. Protocolo de validação | ✅ Concluída | [02_validacao](notebooks/02_validacao.ipynb) | Protocolo fixo: **5 dobras × 3 repetições** (semente 2026), as mesmas para todos os modelos, implementado em [`src/validation.py`](src/validation.py). Um holdout único varia ±0,02 só pela divisão, contra ±0,0003 do protocolo. Os baselines do enunciado foram reproduzidos (constante ≈ 1,17, linear ≈ 0,82). A comparação pareada confirma o ganho da afinidade (≈ 0,65, melhor em 15 de 15 dobras). A curva do kNN ilustra o equilíbrio viés–variância, e a validação aninhada mede o otimismo da escolha de hiperparâmetros. |
| 3. Modelos lineares com engenharia de atributos | 🧪 Preparado para execução | [03_lineares](notebooks/03_lineares.ipynb), [04_splines](notebooks/04_splines.ipynb) | Ridge, Lasso, Elastic Net, encolhimento, interações e splines. |
| 4. Modelos não lineares | 🧪 Preparado para execução | [05_knn](notebooks/05_knn.ipynb), [06_arvores](notebooks/06_arvores.ipynb), [07_boosting](notebooks/07_boosting.ipynb), [08_svr](notebooks/08_svr.ipynb), [09_mlp](notebooks/09_mlp.ipynb) | Busca interna de hiperparâmetros e avaliação externa comum. Abordagem ordinal fica como extensão. |
| 5. Modelo final e submissão | 🧪 Preparado para execução | [10_comparacao_final](notebooks/10_comparacao_final.ipynb) | Carrega os resultados, compara, reajusta e exporta quando `test.csv` estiver disponível. |
| 6. Relatório | ⏳ A fazer | — | |

## Executar os experimentos por família

Abra os notebooks 03–09 a partir da raiz do projeto ou da pasta `notebooks` e execute as células em ordem. Eles são independentes entre si e compartilham as receitas de `src/experiments.py`. Não é necessário executar novamente a EDA ou o notebook de validação.

- `SMOKE_TEST = False` (padrão): avaliação completa nas mesmas 5 dobras × 3 repetições, com busca interna de 3 dobras em cada partição externa. Pode levar vários minutos por família.
- `SMOKE_TEST = True`: verifica os pipelines com uma amostra e uma configuração por receita. Não salva resultados e não serve para comparar desempenho.
- As execuções completas salvam `results/<família>.json`: erros por dobra, configurações escolhidas, previsões externas e identificação dos dados, código e protocolo. Reexecute as famílias se esses elementos mudarem.
- O notebook 10 lê esses arquivos sem repetir os experimentos. Informa famílias pendentes e faz uma nova busca somente para ajustar o modelo escolhido em todo o treino.
- Para gerar `submissions/submission_final.csv`, adicione `data/test.csv`. O arquivo ainda não está presente nesta cópia do projeto.

A avaliação aninhada protege a busca de hiperparâmetros dentro de cada receita. A escolha final entre famílias ainda pode tornar o menor erro observado otimista; a comparação final explicita essa limitação. As conclusões dos notebooks devem ser preenchidas após as execuções completas.

## Materiais de estudo e próximos passos

- [Apresentação visual dos modelos](docs/modelos-bauhaus.html): slides em HTML com exemplos didáticos animados e controles interativos. Abra o arquivo no navegador.
- [Próximos passos e plano de estudo](docs/proximos-passos.md): sequência dos experimentos, conceitos prioritários e revisão dos notebooks com a voz e a compreensão do grupo.

## Referências

- Y. Abu-Mostafa, M. Magdon-Ismail e H.-T. Lin. *Learning from Data*. AMLBook, 2012.
- G. James, D. Witten, T. Hastie e R. Tibshirani. *An Introduction to Statistical Learning*, 2ª ed., 2023.
