# Próximos passos e plano de estudo do trabalho

O objetivo é concluir os experimentos, escolher um modelo com uma comparação defensável e entregar um trabalho que o grupo consiga explicar de cabo a rabo. Isso inclui reler o enunciado, o código e todas as explicações, conferir os resultados e reescrever os notebooks com a nossa forma de falar.

Os notebooks 03 a 09 estão preparados para execução. Os testes rápidos de funcionamento não substituem os experimentos completos. A apresentação `modelos-bauhaus.html` ajuda a entender os mecanismos, mas suas figuras são didáticas e não mostram o desempenho dos modelos na base do trabalho.

## Ordem de trabalho

### 1. Reler o enunciado e definir o que precisamos entregar

- [ ] Reler o PDF do projeto e anotar as exigências de entrega, a métrica, as restrições e os critérios de avaliação.
- [ ] Confirmar como os dados foram construídos e o significado de cada atributo. Em particular, entender se as médias fornecidas excluem a nota que queremos prever.
- [ ] Confirmar o formato da submissão e conseguir o arquivo de teste. Nesta cópia do projeto, `data/test.csv` ainda está ausente.
- [ ] Conferir as regras da disciplina sobre ferramentas de IA e registrar seu uso conforme essas regras.
- [ ] Conferir se todas as afirmações do README e dos notebooks correspondem ao que o enunciado e os resultados permitem afirmar.

**Ao terminar:** precisamos conseguir explicar o problema, a informação disponível na hora da previsão e o que conta como uma boa solução, sem consultar um texto pronto.

### 2. Entender e revisar os notebooks 01 e 02

Esses notebooks sustentam todas as decisões seguintes. Vale dedicar tempo a eles antes de rodar mais modelos.

- [ ] Executar a EDA do início ao fim e conferir se cada interpretação corresponde à figura ou à tabela imediatamente anterior.
- [ ] Explicar como a afinidade combina o gênero do filme com o histórico do usuário e por que a regressão bruta perde essa interação.
- [ ] Entender por que as três proporções de gênero, juntas com o intercepto, criam colinearidade perfeita.
- [ ] Reproduzir os baselines e conferir os números, incluindo o corte das previsões em [0, 5].
- [ ] Desenhar no papel as dobras externas e internas. Marcar exatamente quais linhas cada transformação e cada busca podem enxergar.
- [ ] Revisar os trechos sobre incerteza. O desvio entre dobras e o erro padrão aproximado de diferenças dependentes não justificam, sozinhos, um teste formal de significância.
- [ ] Conferir as condições dos argumentos teóricos. Por exemplo, a perda quadrática de previsões sem limite não satisfaz automaticamente as hipóteses de uma versão de Hoeffding para perdas limitadas.

**Ao terminar:** uma pessoa deve conseguir mostrar uma linha de validação e explicar por que nem sua nota nem suas estatísticas de validação participaram do ajuste daquele pipeline.

### 3. Rodar os experimentos completos por família

Começar por `03_lineares.ipynb` e `04_splines.ipynb`, que aproveitam diretamente os achados da EDA. Depois executar kNN, árvores, boosting, SVR e MLP. Os notebooks são independentes entre si.

- [ ] Usar `SMOKE_TEST = False` para resultados de desempenho.
- [ ] Executar cada notebook do começo ao fim no mesmo ambiente de dependências.
- [ ] Investigar avisos de convergência e problemas numéricos. Previsões finitas são necessárias, mas não bastam para comprovar que o ajuste foi adequado.
- [ ] Conferir a tabela de treino e validação, as configurações escolhidas por dobra e os resíduos fora da amostra.
- [ ] Confirmar a gravação de `results/<família>.json`.
- [ ] Preencher a conclusão de cada notebook com os resultados efetivamente observados.

As buscas têm grades pequenas para começar. Se o melhor valor ficar numa borda da grade, vale considerar uma ampliação com justificativa. Registrar essa decisão, pois explorar muitas variantes olhando os mesmos resultados também aumenta o otimismo da seleção.

Os JSONs identificam dados, código e protocolo. Se os arquivos compartilhados de código mudarem, será necessário executar novamente as famílias para manter a comparação compatível. Evitar alterar as receitas no meio de uma rodada de comparação.

**Ao terminar:** teremos uma execução completa de cada família escolhida, com resultados salvos e uma conclusão que diga o que aprendemos.

### 4. Comparar os modelos e investigar os ganhos

- [ ] Executar `10_comparacao_final.ipynb` e conferir se há famílias pendentes.
- [ ] Comparar com a constante e com a linear com afinidade, usando as mesmas dobras.
- [ ] Ver se os ganhos são consistentes por dobra e se o tamanho do ganho compensa a complexidade e o custo.
- [ ] Examinar os resíduos por afinidade, nota prevista e gênero. Se for útil, acrescentar diagnósticos por contagem de avaliações e faixa de nota real.
- [ ] Distinguir o ganho do algoritmo do ganho da engenharia de atributos. Os modelos não lineares atuais também recebem atributos construídos.
- [ ] Fazer poucas comparações controladas para responder dúvidas concretas: mesmo modelo com e sem afinidade, com e sem encolhimento, por exemplo.
- [ ] Registrar a limitação da escolha entre famílias. A avaliação aninhada atual protege a busca dentro de cada receita, mas a escolha da família com menor erro externo ainda usa essa avaliação.

Se precisarmos estimar o erro de todo o procedimento de escolha, o laço externo deve conter também a seleção da família, ou precisamos de uma avaliação independente. Decidir a necessidade disso à luz do enunciado, do custo e do que pretendemos afirmar no relatório.

**Ao terminar:** devemos conseguir defender a escolha final e explicar por que um número um pouco menor, sozinho, não resolve toda a comparação.

### 5. Reler e reescrever o trabalho inteiro com a nossa voz

Precisamos reler todos os notebooks, do 01 ao 10, além do README e do relatório. A escrita atual é um ponto de partida. Vamos reescrever as explicações com nossas palavras, para tirar a cara de texto genérico de IA e garantir que entendemos tudo que estamos apresentando.

A revisão precisa ir além de trocar palavras. Cada integrante deve reconstruir o raciocínio e conferir se o código e os resultados sustentam a explicação.

Para cada seção:

1. Ler e executar o código, acompanhando as entradas, as transformações e as saídas.
2. Fechar a explicação existente e dizer, em voz alta, o que fizemos, por que fizemos e o que o resultado permite concluir.
3. Escrever novamente do zero, com o vocabulário que usaríamos para explicar a um colega da turma.
4. Voltar à figura ou à tabela e conferir cada afirmação. Remover conclusões que o experimento não sustentou.
5. Manter os termos técnicos necessários e explicar seu significado. Associar fórmulas às variáveis usadas no código.
6. Pedir para outro integrante ler e fazer perguntas. Se não soubermos responder, voltar ao conceito ou refazer um exemplo pequeno.

Algumas perguntas ajudam a substituir texto genérico por raciocínio próprio:

- Qual dúvida concreta levou a este experimento?
- Qual foi a única coisa que mudamos nesta comparação? Se mudamos várias, o que ainda conseguimos concluir?
- Qual número ou padrão na figura sustenta esta frase?
- O resultado confirmou nossa hipótese, contrariou-a ou ficou inconclusivo?
- Qual limitação precisa aparecer junto da conclusão?

Não precisamos inserir erros, informalidade artificial ou sinônimos para parecer humanos. O texto deve refletir o que de fato entendemos. A declaração de uso de IA, quando exigida pela disciplina, deve corresponder ao processo real.

**Ao terminar:** qualquer integrante deve conseguir explicar uma seção escolhida ao acaso e responder perguntas sobre suas decisões, sem depender do texto gerado.

### 6. Fechar o relatório e produzir a submissão

- [ ] Atualizar a tabela de resultados e o andamento no README.
- [ ] Escrever o relatório em torno do problema, hipóteses, protocolo, resultados e limitações.
- [ ] Separar claramente fatos observados, interpretações e hipóteses não testadas.
- [ ] Conferir a interpretação do RMSE e de qualquer piso de erro citado no enunciado. Não prometer atingir esse piso nem afirmar que toda diferença até ele é recuperável com os atributos disponíveis.
- [ ] Registrar o ambiente e as versões usadas na execução final, além das sementes e configurações.
- [ ] Reiniciar o kernel e executar os notebooks na ordem das células para identificar dependências de estado oculto.
- [ ] Ajustar a receita escolhida em todo o treino, sem consultar as notas de teste.
- [ ] Gerar a submissão no formato exigido, preservando `pair_id`, ordem, quantidade de linhas e valores reais em [0, 5].
- [ ] Conferir nomes de arquivos, referências, fórmulas, figuras e exigências de entrega do enunciado.

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
