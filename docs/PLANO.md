# Plano de trabalho do grupo

Documento interno de organização. A descrição pública do projeto está no [README](../README.md).

## Lembretes do enunciado

- Referências de RMSE: constante ≈ 1,19 · linear bruta ≈ 0,84 (meta) · oráculo ≈ 0,39 (piso).
- A nota vem da **qualidade da validação** e da **explicação, nos termos do curso**, de por que o modelo final supera (ou não) a regressão linear.
- O relatório tem no máximo 4 páginas e é entregue antes da apresentação. Na apresentação, o professor faz perguntas a **cada integrante**.

## Etapas

### Etapa 0: Preparação
- Colocar os CSVs em `data/` (eles não são versionados), configurar o ambiente e todos clonarem o repositório.

### Etapa 1: Análise exploratória (EDA)
- Como `stars` se relaciona com `user_avg_stars` e `movie_avg_stars`? A relação é linear?
- **Confiabilidade das médias:** o efeito de `user_avg_stars` depende de `user_n_ratings`? E o de `movie_avg_stars` depende de `movie_n_ratings`?
- **Afinidade de gênero:** quem assiste muito drama gosta mais de filmes de drama? Isso é uma interação entre `genre` e `share_*`.
- `age` e `release_year` têm efeitos não lineares?

### Etapa 2: Protocolo de validação
- Validação cruzada com k = 5, repetida com algumas sementes, usando **as mesmas dobras para todos os modelos**.
- Reportar o RMSE médio ± o desvio entre as dobras.
- O teste nunca é usado para decidir nada.
- Reproduzir os baselines (≈ 1,19 e ≈ 0,84) para conferir o protocolo.
- Teoria: E_in vs. E_out, sobreajuste, Hoeffding (por que o erro de validação estima bem o E_out).

### Etapa 3: Modelo linear com boas features
- Codificar `genre` em one-hot. Criar a feature de afinidade (o `share` do gênero do filme).
- Encolher as médias: `(n · média + k · média_global) / (n + k)`.
- `log(n)`, interações, splines (tópico 5 da ementa).
- Ridge e Lasso, com seleção de variáveis (tópico 8).

### Etapa 4: Modelos não lineares (tópicos da ementa)
- k-vizinhos mais próximos e árvores, florestas aleatórias e boosting (tópico 3).
- SVM / SVR (tópico 10).
- Redes neurais (tópico 12).
- Abordagem ordinal: modelar P(stars = k) e prever E[stars].
- (Opcional) variante de classificação "gostou" (stars ≥ 4).

### Etapa 5: Modelo final e submissão
- Escolher pelo RMSE de validação. Entre modelos empatados, ficar com o mais simples (Navalha de Ockham).
- Reajustar em todo o treino, cortar as previsões em [0, 5] e gerar `submissions/submission.csv`.

### Etapa 6: Relatório e apresentação
1. Dados e EDA
2. Protocolo de validação
3. Modelos e tabela de RMSE
4. Por que o modelo final supera a linear: viés vs. variância, dimensão VC, distância até o oráculo
5. Conclusão

## Divisão de tarefas (sugestão)

| Integrante | Responsabilidade principal |
|---|---|
| A | EDA e engenharia de features (Etapas 1 e 3) |
| B | Validação, baselines e modelos lineares regularizados (Etapas 2 e 3) |
| C | Modelos não lineares e ordinais, submissão (Etapas 4 e 5) |

Todos escrevem o relatório e devem saber explicar a escolha do modelo final.

## Fluxo de trabalho no Git

1. `git pull` antes de começar.
2. Criar uma branch por tarefa: `git checkout -b eda-afinidade`.
3. Fazer commits pequenos, com mensagens claras.
4. Abrir um Pull Request e pedir a revisão de um colega antes do merge na `main`.
