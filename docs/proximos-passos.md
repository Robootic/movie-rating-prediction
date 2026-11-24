# Próximos passos após a rodada atual

Atualizado em 21 de novembro de 2026. O documento `plano_conclusao_projeto.docx` registra a proposta anterior à execução e deve ser lido como histórico de planejamento.

## O que foi concluído

Executamos os notebooks 03–08 com 5 folds × 5 repetições, semente 2026. A síntese 09 confere dados, código, partições e versões antes de comparar resultados. Os resultados atuais ficam em `results/rodada_atual`; os demais resultados e as receitas antigas não definem esta rodada.

A sequência duas médias → afinidade → curvatura → gênero reduziu o RMSE de 0,839575 para 0,625036. Idade e ano levaram o boosting a 0,565157; os logaritmos das duas contagens reduziram o resultado final a 0,544933. O boosting venceu nas 25 partições e nas cinco repetições. Ridge permaneceu praticamente idêntico à regressão.

## Decisão final

Nossa investigação distingue atributos brutos de atributos construídos. Os dados possuem dez atributos brutos; a receita final os representa em onze atributos construídos, pois o ano é desdobrado em dois termos e gênero/proporções são condensados. Escolhemos o boosting com essa representação completa porque sua vantagem permaneceu ampla e estável, mesmo reconhecendo o gap $E_{CV} - E_{in}$ de 0,113212, contra 0,002581 nos modelos lineares.

O diagnóstico OOF da síntese mostrou padrões de idade e lançamento. As duas hipóteses foram executadas:

1. Acrescentar idade à referência, começando por um termo linear, e verificar ganho e resíduos por idade.
2. Investigar lançamento em uma comparação separada; o arco nos resíduos motiva testar termo linear e depois quadrático, mantendo o restante fixo.

Idade, ano linear, ano quadrático e as duas contagens em log foram avaliados. A versão completa usa informação dos dez atributos brutos e obteve o menor RMSE. Não haverá novas explorações nesta entrega.

## Caminho até a entrega

- [x] Revisar as leituras numéricas dos notebooks 03–09 e consolidar suas interpretações.
- [ ] Confirmar com o professor prazo, critérios, regras de IA e construção das médias.
- [x] Executar os testes finais de idade e ano e congelar a receita vencedora.
- [x] Consolidar uma tabela comparável de treino, validação e estabilidade.
- [x] Avaliar kNN apenas como verificação motivada pela aula; ele não melhorou o resultado.
- [x] Encerrar a exploração planejada.
- [x] Escolher o boosting com idade, ano e contagens, reconhecendo o otimismo da seleção entre famílias e seu maior gap $E_{CV} - E_{in}$.
- [x] Escrever relatório: problema → evidências → hipóteses → protocolo → resultados → decisão → limitações. Distinguir associação, hipótese e resultado.
- [x] Revisar a documentação para remover números e estados desatualizados.
- [x] Ajustar a receita final no treino completo e gerar a submissão sem usar o teste para escolher o modelo.
- [x] Conferir ordem e identificadores, valores finitos e previsões em [0, 5].

## Atualizações de 22 e 23 de novembro de 2026

- **Diário completo.** O notebook 10 documenta os testes de idade, ano e contagens, que antes só existiam como scripts, e explica em detalhes o teste t corrigido, com um exemplo passo a passo.
- **Hipótese de nostalgia.** O notebook 11 mostra que boa parte da vantagem do boosting vem da interação idade × ano (notas mais altas para filmes lançados quando o usuário tinha 10 a 20 anos, supondo idades relativas a 2024). Com esse atributo, a regressão empata com o boosting de onze atributos, e o boosting com a idade no lançamento chega a 0,519532, vencendo os demais candidatos nas 25 partições.
- **Novo modelo final.** Depois de verificar que o resultado não depende do ano de referência nem da borda da grade (`src/nostalgia_checks.py`), o boosting com os onze atributos mais a idade no lançamento passou a ser o modelo final, e a submissão foi regenerada. A submissão anterior (boosting com onze atributos) fica no histórico do Git.
- **Reprodutibilidade.** Os testes corrigidos têm código (`src/corrected_test.py`). O boosting varia na quarta casa decimal entre computadores (0,544851 na primeira execução, 0,544933 na atual), porque algumas configurações da grade empatam quase exatamente; a regressão é idêntica. Os resultados em `results/rodada_atual` foram regenerados na execução atual.
- **Diagnóstico do modelo final.** O notebook 12 separa os erros por subgrupo, testa a calibração e analisa as notas extremas. O erro restante é, em boa parte, ruído; resta um viés pequeno no pico da nostalgia.
- **Organização.** Os notebooks do roteiro antigo, o gerador do relatório e o material auxiliar foram removidos. Os termos foram padronizados: atributo, $E_{in}$ e $E_{CV}$.

Pendências:

- [x] Decidir se o modelo final passa a usar a idade no lançamento.
- [ ] Revisar o relatório para o novo modelo final.
- [ ] Atualizar `apresentacao-cinemateca.html`: os slides de resultados ainda têm valores fictícios e o protocolo antigo.

## Responsabilidades sugeridas

Distribuir execução e revisão entre dados e protocolo, regressões e resíduos, comparação de procedimentos e relatório/submissão. Definir nomes e prazos no grupo. Cada frente deve ter um responsável e um revisor; a compreensão do modelo final é coletiva.

## O que precisamos estudar de cabo a rabo

| Prioridade | Tema | O que precisamos conseguir explicar |
|---|---|---|
| Essencial | Problema e dados | O que uma linha representa, quais informações existem na previsão e por que `pair_id` fica fora. |
| Essencial | Alvo, perda e RMSE | Por que prever valores reais, por que não arredondar e como o corte em [0, 5] afeta a perda quadrática. |
| Essencial | Generalização | Diferença entre $E_{in}$ e $E_{CV}$, viés, variância e sinais de sobreajuste. |
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
