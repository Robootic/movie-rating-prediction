# Próximos passos após a rodada atual

Atualizado em 15 de novembro de 2026. Este é o roteiro vigente; o documento `plano_conclusao_projeto.docx` registra a proposta anterior à execução e deve ser lido como histórico de planejamento.

## O que foi concluído

Executamos os notebooks 03–08 com 5 folds × 5 repetições, semente 2026. A síntese 09 confere dados, código, partições e versões antes de comparar resultados. Os resultados atuais ficam em `results/rodada_atual`; os demais resultados e as receitas antigas não definem esta rodada.

A sequência duas médias → afinidade → curvatura → gênero reduziu o RMSE de 0,839575 para 0,625036. Com os mesmos seis atributos, boosting chegou a 0,619248, enquanto Ridge e Bayesian Ridge ficaram praticamente empatados com a regressão. O ganho do boosting foi modesto e acompanhado de maior distância entre treino e validação.

## Linha investigativa e decisões pendentes

Nossa investigação distingue informação e forma funcional de mudança de algoritmo. Os maiores ganhos vieram da afinidade e de sua curvatura. A regularização não trouxe ganho material neste conjunto de atributos. Não escolhemos um vencedor definitivo nem demonstramos ausência de sobreajuste.

O diagnóstico OOF da síntese mostra padrões de idade e lançamento em regressão e boosting. A prioridade proposta é discutir e testar essas duas hipóteses, uma alteração por vez, antes de abrir novas famílias:

1. Acrescentar idade à referência, começando por um termo linear, e verificar ganho e resíduos por idade.
2. Investigar lançamento em uma comparação separada; o arco nos resíduos motiva testar termo linear e depois quadrático, mantendo o restante fixo.

Esses testes ainda não foram implementados ou executados. Se mudarem os atributos escolhidos, repetir a comparação regressão × boosting com informação igual. Não ampliar grades por tentativa e erro sem registrar a exploração.

## Caminho até a entrega

- [ ] O grupo deve revisar as leituras numéricas nos notebooks 03–09 e escrever suas interpretações.
- [ ] Confirmar enunciado, prazo, critérios, regras de IA e construção das médias. Obter `test.csv`, ainda ausente.
- [ ] Decidir se os dois testes de atributos cabem no prazo e congelar suas configurações antes de rodar.
- [ ] Consolidar uma tabela comparável após os testes escolhidos, incluindo treino, validação, estabilidade e custo.
- [ ] Considerar kNN ou uma MLP pequena somente se houver uma pergunta adicional e tempo para seleção interna e diagnóstico. Árvore simples pode ser didática; Random Forest não é uma obrigação.
- [ ] Encerrar a exploração quando as hipóteses acordadas estiverem respondidas e testes adicionais não se justificarem pelo prazo e benefício esperado.
- [ ] Escolher uma receita, reconhecendo o otimismo da seleção entre famílias. Não declarar pequenas diferenças como significativas com base em dobras dependentes.
- [ ] Escrever relatório: problema → evidências → hipóteses → protocolo → resultados → decisão → limitações. Distinguir associação, hipótese e resultado.
- [ ] Revisar documentação e apresentação para remover números e estados desatualizados. Todos devem conseguir explicar as decisões.
- [ ] Refazer a seleção interna definida no treino completo, quando aplicável, ajustar a receita final e gerar a submissão no formato exigido, sem usar teste para escolher modelo.
- [ ] Conferir execução limpa, versões, ordem e identificadores, valores finitos, previsões em [0, 5], referências e arquivos finais.

## Responsabilidades sugeridas

Distribuir execução e revisão entre dados e protocolo, regressões e resíduos, comparação de procedimentos e relatório/submissão. Definir nomes e prazos no grupo. Cada frente deve ter um responsável e um revisor; a compreensão do modelo final é coletiva.

## O que precisamos estudar de cabo a rabo

| Prioridade | Tema | O que precisamos conseguir explicar |
|---|---|---|
| Essencial | Problema e dados | O que uma linha representa, quais informações existem na previsão e por que `pair_id` fica fora. |
| Essencial | Alvo, perda e RMSE | Por que prever valores reais, por que não arredondar e como o corte em [0, 5] afeta a perda quadrática. |
| Essencial | Generalização | Diferença entre erro de treino e validação, viés, variância e sinais de sobreajuste. |
| Essencial | Vazamento e pipelines | Onde cada transformação aprende e como evitar que dados da validação influenciem o ajuste. |
| Essencial | Validação e seleção | Função de cada laço da validação aninhada, comparação pareada e otimismo da seleção entre famílias. |
| Essencial | Afinidade e interações | Por que a combinação de gênero e proporção importa e o que significa ser linear nos coeficientes. |
| Alta | Ridge, Lasso e Elastic Net | Função da penalização, padronização, coeficientes zerados e comportamento com atributos correlacionados. |
| Alta | Encolhimento e splines | Como a contagem controla a confiança numa média e como bases locais representam curvas. |
| Alta | Árvores e ensembles | Critério de corte, previsão nas folhas, bootstrap, sorteio de atributos e limiares, média de árvores e correções sequenciais. |
| Alta | Diagnóstico e interpretação | Como identificar estrutura nos resíduos e por que associação e importância preditiva não estabelecem causalidade. |
| Por modelo | kNN | Distância, padronização, escolha de k e pesos por distância. |
| Por modelo | SVR | Tolerância epsilon, papel de C e gamma, kernel RBF e compromisso entre complexidade e desvios. |
| Por modelo | MLP | Propagação direta, ReLU, saída de regressão, retropropagação, otimização e parada antecipada. |

Todos precisam dominar os temas essenciais. Podemos dividir o aprofundamento inicial dos modelos, mas cada responsável deve explicar sua família ao restante do grupo. O modelo final exige compreensão coletiva completa.

## Exercícios curtos para conferir a compreensão

- Calcular a melhor previsão constante e o RMSE de quatro notas à mão.
- Mostrar que, para uma nota real em [0, 5], projetar uma previsão nesse intervalo não aumenta seu erro quadrático.
- Construir dois pares com o mesmo usuário e gêneros diferentes, calcular suas afinidades e explicar a interação.
- Encolher uma média baseada em poucas avaliações e outra baseada em muitas, usando a mesma referência e o mesmo parâmetro.
- Desenhar um pipeline dentro de uma dobra e indicar onde aprendemos escalas, nós dos splines, referências e hiperparâmetros.
- Fazer um corte de árvore numa amostra pequena e calcular as médias das folhas e a redução do erro quadrático.
- Calcular uma etapa de boosting: previsão inicial, resíduos, correção e taxa de aprendizado.
- Explicar o que mudaria se escolhêssemos o melhor hiperparâmetro olhando a própria dobra externa.

## Como saber se estamos prontos

Uma boa revisão final é escolher uma célula ao acaso e pedir que alguém explique: o que entra, o que sai, o que foi aprendido, quais dados foram usados e por que essa célula existe. Depois, escolher uma conclusão e pedir a evidência que a sustenta.

Se conseguirmos fazer isso nos notebooks inteiros, com resultados reproduzidos e escrita própria, teremos um trabalho que podemos apresentar e defender com segurança.
