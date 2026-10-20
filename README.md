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
src/           código reutilizável: validação, atributos, modelos
submissions/   previsões geradas para o conjunto de teste
report/        relatório final
docs/          documentos de organização do projeto
```

## Como reproduzir

Requer Python 3.10 ou superior.

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

## Referências

- Y. Abu-Mostafa, M. Magdon-Ismail e H.-T. Lin. *Learning from Data*. AMLBook, 2012.
- G. James, D. Witten, T. Hastie e R. Tibshirani. *An Introduction to Statistical Learning*, 2ª ed., 2023.
