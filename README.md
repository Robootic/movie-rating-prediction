# Prevendo avaliações de filmes — Projeto MAE0501

Projeto da disciplina **MAE0501 – Aprendizagem Estatística** (IME–USP, 2º semestre de 2026, Prof. Morgan André).

**Tarefa:** dado um par (usuário, filme) descrito por 10 atributos, prever quantas estrelas (0 a 5) o usuário dá ao filme.
**Métrica:** RMSE no conjunto de teste (previsões reais, cortadas em [0, 5]).

| Referência | RMSE |
|---|---|
| Constante (média de treino) | ~1,19 |
| Regressão linear nos 10 atributos brutos | ~0,84 ← meta a bater |
| Oráculo (piso teórico, ruído irredutível) | ~0,39 |

**Entregáveis:** `submissions/` (arquivo `pair_id,stars`), o código deste repositório e um relatório de no máximo 4 páginas em `report/`.
**Avaliação:** qualidade da **validação** e capacidade de explicar, nos termos do curso, **por que** o modelo final supera (ou não) a regressão linear.

---

## Estrutura do repositório

```
data/          train.csv, test.csv, sample_submission.csv (fornecidos pelo professor)
notebooks/     análises exploratórias e experimentos (um notebook por etapa)
src/           código reutilizável: validação, features, modelos
submissions/   arquivos de submissão gerados
report/        relatório final (≤ 4 páginas)
```

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

---

## Plano de trabalho

### Etapa 0 — Preparação
- Colocar os CSVs em `data/`, configurar o ambiente, todos clonarem o repositório.

### Etapa 1 — Análise exploratória (EDA)
Perguntas que guiam a análise:
- Como `stars` se relaciona com `user_avg_stars` e `movie_avg_stars`? A relação é linear?
- **Confiabilidade das médias:** uma média calculada com 3 notas é muito mais ruidosa do que uma com 500. O efeito de `user_avg_stars` depende de `user_n_ratings`? (mesma pergunta para o filme)
- **Afinidade de gênero:** o usuário que assiste muito drama gosta mais de filmes de drama? Isso é uma *interação* entre `genre` e `share_*` que a regressão linear bruta não enxerga.
- Efeitos não lineares de `age` e `release_year`?

### Etapa 2 — Protocolo de validação (a parte mais importante da nota)
- **Validação cruzada k-fold** (k = 5, repetida com algumas sementes) sobre as 4000 linhas de treino, com **as mesmas dobras para todos os modelos** → comparação justa.
- Reportar RMSE médio ± desvio entre dobras.
- Toda escolha (features, hiperparâmetros) é feita **só** com a validação; o teste nunca é usado para decidir nada.
- Reproduzir os dois baselines (constante ≈ 1,19 e linear ≈ 0,84) como sanidade do protocolo.
- Ligação com a teoria: E_in (erro de treino) vs. E_out (erro fora da amostra), sobreajuste, desigualdade de Hoeffding (por que o erro de validação é um bom estimador de E_out).

### Etapa 3 — Melhorar o modelo linear com boas features
- Codificar `genre` (one-hot).
- **Afinidade:** `share` do gênero do filme (ex.: se o filme é de ação, usar `share_action`).
- **Encolhimento (shrinkage) das médias:** `(n · média + k · média_global) / (n + k)` — puxa médias baseadas em poucas notas para a média global.
- Transformações (`log(n)`), interações, termos polinomiais / splines.
- **Regularização** (Ridge / Lasso) para controlar a complexidade → trade-off viés–variância.

### Etapa 4 — Modelos não lineares
- k-vizinhos mais próximos, árvores, florestas aleatórias / gradient boosting (de acordo com o que for visto no curso).
- Aproveitar a natureza **ordinal**: modelar P(stars = k) e prever o **valor esperado** E[stars] (não a classe mais provável).
- (Opcional) variante de classificação "gostou" (stars ≥ 4).

### Etapa 5 — Modelo final e submissão
- Escolher o modelo pelo RMSE de validação (preferindo o mais simples entre os empatados — Navalha de Ockham).
- Reajustar em todo o treino, prever o teste, cortar em [0, 5], gerar `submissions/submission.csv`.

### Etapa 6 — Relatório (≤ 4 páginas) e apresentação
1. Dados e EDA (principais achados)
2. Protocolo de validação
3. Modelos testados e tabela de RMSE de validação
4. **Por que o modelo final supera a regressão linear** — viés (a linear bruta não representa interações/não linearidades) vs. variância (regularização, dimensão VC / número de parâmetros), distância até o piso do oráculo (0,39)
5. Conclusão

---

## Divisão de tarefas (sugestão)

| Integrante | Responsabilidade principal |
|---|---|
| A | EDA + engenharia de features (Etapas 1 e 3) |
| B | Protocolo de validação + baselines + modelos lineares regularizados (Etapas 2 e 3) |
| C | Modelos não lineares / ordinais + submissão (Etapas 4 e 5) |

O relatório é escrito por todos, e **todos** devem saber explicar por que o modelo final foi escolhido.

## Fluxo de trabalho no Git

1. `git pull` antes de começar.
2. Crie uma branch para sua tarefa: `git checkout -b eda-afinidade`.
3. Commits pequenos com mensagens claras.
4. Abra um Pull Request e peça para um colega revisar antes de fazer merge na `main`.
