"""Gera o relatório acadêmico final em até quatro páginas."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output" / "pdf" / "relatorio_final.pdf"


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#777777"))
    canvas.setLineWidth(0.35)
    canvas.line(1.8 * cm, 1.25 * cm, 19.2 * cm, 1.25 * cm)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawString(1.8 * cm, 0.88 * cm, "MAE0501 - Aprendizagem Estatística")
    canvas.drawRightString(19.2 * cm, 0.88 * cm, str(document.page))
    canvas.restoreState()


def academic_table(data, widths, alignments=None, font_size=8.2):
    table = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.1),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for column, alignment in enumerate(alignments or []):
        style.append(("ALIGN", (column, 0), (column, -1), alignment))
    table.setStyle(TableStyle(style))
    return table


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=1.85 * cm, rightMargin=1.85 * cm,
        topMargin=1.55 * cm, bottomMargin=1.65 * cm,
        title="Prevendo avaliações de filmes",
        author=("Lucas Maia Fontes Pineiro; Matheus Ramos de Oliveira Queiroz; "
                "Thais Martins de Sousa"),
    )
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "AcademicTitle", parent=base["Title"], fontName="Times-Bold",
        fontSize=18, leading=21, alignment=TA_CENTER, spaceAfter=5,
    )
    authors = ParagraphStyle(
        "Authors", parent=base["Normal"], fontName="Times-Roman",
        fontSize=9, leading=11.5, alignment=TA_CENTER, spaceAfter=8,
    )
    abstract = ParagraphStyle(
        "Abstract", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=8.8, leading=11.2, alignment=TA_JUSTIFY,
        leftIndent=0.7 * cm, rightIndent=0.7 * cm, spaceAfter=7,
    )
    heading = ParagraphStyle(
        "Section", parent=base["Heading2"], fontName="Times-Bold",
        fontSize=11.3, leading=13.5, spaceBefore=7, spaceAfter=3,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=9.2, leading=11.7, alignment=TA_JUSTIFY, spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=7.9, leading=9.7,
        textColor=colors.HexColor("#333333"),
    )

    story = [
        Spacer(1, 0.15 * cm),
        Paragraph("Prevendo avaliações de filmes", title),
        Paragraph(
            "Lucas Maia Fontes Pineiro (14591243) &nbsp;&nbsp;•&nbsp;&nbsp; "
            "Matheus Ramos de Oliveira Queiroz (15479562) &nbsp;&nbsp;•&nbsp;&nbsp; "
            "Thais Martins de Sousa (14608786)<br/>"
            "MAE0501 - Aprendizagem Estatística, IME-USP, 2026", authors),
        Paragraph("<b>Resumo.</b> Investigamos a previsão da nota, entre 0 e 5 estrelas, "
                  "que um usuário daria a um filme. A análise foi organizada como uma "
                  "sequência de hipóteses sobre informação, forma funcional e flexibilidade. "
                  "A afinidade entre usuário e gênero produziu o maior ganho inicial. Testes "
                  "finais incluíram idade, ano e contagens. Com os dez atributos brutos "
                  "representados em onze colunas, o boosting obteve RMSE OOF de 0,544851, venceu "
                  "nas 25 partições externas e foi "
                  "adotado como modelo final, com a maior tendência ao sobreajuste registrada "
                  "como limitação. Uma análise posterior mostra que boa parte dessa vantagem vem "
                  "de uma interação entre idade e ano de lançamento (nostalgia): com essa coluna, "
                  "a regressão linear empata com o boosting.", abstract),
        Paragraph("1. Problema e dados", heading),
        Paragraph("O conjunto contém 4000 pares usuário-filme para treino e 2000 para teste. "
                  "Os dez atributos descrevem médias e contagens de avaliações, idade, "
                  "preferências de gênero, gênero e ano de lançamento. O identificador "
                  "<i>pair_id</i> não entra no modelo. A resposta <i>stars</i> varia de 0 a 5; "
                  "as previsões são reais e cortadas nesse intervalo.", body),
        Paragraph("O desempenho é medido pela raiz do erro quadrático médio, "
                  "RMSE = raiz da média de (observado - previsto)². A métrica permanece em "
                  "estrelas e penaliza erros grandes. A previsão constante apresenta RMSE "
                  "próximo de 1,19 e a regressão nos dez atributos brutos, aproximadamente 0,84, "
                  "segundo o enunciado. No nosso protocolo (seção 3), essas referências "
                  "obtiveram 1,165913 e 0,824064.", body),
        Paragraph("2. Linha investigativa", heading),
        Paragraph("Partimos das médias do usuário e do filme. Em seguida definimos afinidade "
                  "como a proporção do gênero do filme no histórico do usuário; testamos sua "
                  "curvatura e indicadores de comédia e drama. Os resíduos ainda mostravam "
                  "estrutura em idade e ano de lançamento, testados ao final como idade linear, "
                  "ano linear e ano quadrático. Essa ordem separa ganho de informação de mudança "
                  "de algoritmo.", body),
        Paragraph("Os dados possuem dez atributos brutos. A referência de seis colunas combina "
                  "duas médias, afinidade, afinidade² e dois indicadores de gênero. Idade e os "
                  "dois termos do ano elevam o total a nove; os logaritmos das duas contagens "
                  "levam a receita final a onze colunas numéricas. Não há um 11º atributo bruto: "
                  "gênero e proporções foram condensados, enquanto o ano foi desdobrado.", body),
        Paragraph("3. Protocolo de validação", heading),
        Paragraph("Usamos validação cruzada repetida com 5 folds, 5 repetições e semente 2026. "
                  "Em cada repetição, todas as observações recebem uma previsão fora do treino "
                  "(OOF). O RMSE OOF repetido considera as cinco previsões de cada linha "
                  "separadamente. Todos os modelos compartilham as mesmas 25 partições.", body),
        Paragraph("Ridge e boosting selecionam hiperparâmetros em três folds internos de cada "
                  "treino externo. Transformações e seleção permanecem dentro do ajuste. O teste "
                  "não foi consultado para escolher atributos, modelos ou configurações.", body),
        academic_table([
            ["Procedimento", "Função", "Seleção"],
            ["Regressão comum", "Referência linear interpretável", "Sem hiperparâmetro"],
            ["Ridge", "Regularização L2", "alpha em 3 folds internos"],
            ["Bayesian Ridge", "Regularização por Bayes empírico", "Precisões no treino"],
            ["Boosting", "Relações e interações não lineares", "Grade em 3 folds internos"],
            ["kNN", "Verificação motivada pela aula", "Avaliação separada"],
        ], [3.2 * cm, 7.5 * cm, 5.1 * cm], font_size=8.1),
        Paragraph("4. Testes de idade e ano", heading),
        Paragraph("Os testes foram definidos antes da rodada final: acrescentar idade; ano "
                  "linear; ano linear e quadrático; e as combinações com idade. A tabela mostra "
                  "regressões sem penalização no mesmo protocolo.", body),
        academic_table([
            ["Conjunto de atributos", "RMSE OOF"],
            ["Referência: seis atributos", "0,625036"],
            ["+ idade", "0,620722"],
            ["+ ano linear", "0,622705"],
            ["+ ano linear e quadrático", "0,620512"],
            ["+ idade e ano linear", "0,618458"],
            ["+ idade, ano linear e quadrático", "0,616080"],
        ], [12.8 * cm, 3.0 * cm], ["LEFT", "RIGHT"], 8.2),
        Paragraph("Idade e ano fornecem informação complementar. O termo quadrático melhora o "
                  "ano linear quando usado com idade, coerente com o arco observado nos resíduos. "
                  "O conjunto completo foi congelado para a comparação final de procedimentos.", body),
        Paragraph("As partições compartilham treinos e as hipóteses nasceram da exploração dos "
                  "mesmos dados. Portanto, a rodada mede desempenho preditivo interno, não uma "
                  "confirmação independente de cada hipótese.", small),
        Paragraph("5. Resultados", heading),
        academic_table([
            ["Etapa ou modelo", "RMSE OOF"],
            ["Constante", "1,165913"],
            ["Regressão nos dez atributos brutos (referência do enunciado)", "0,824064"],
            ["Duas médias", "0,839575"],
            ["Duas médias + afinidade", "0,660065"],
            ["+ afinidade ao quadrado", "0,628989"],
            ["Regressão com gênero", "0,625036"],
            ["Boosting com seis atributos", "0,619248"],
            ["Regressão com nove atributos", "0,616080"],
            ["Ridge com nove atributos", "0,616081"],
            ["Boosting com nove atributos", "0,564180"],
            ["Regressão com onze colunas", "0,615210"],
            ["Ridge com onze colunas", "0,615210"],
            ["Boosting com onze colunas", "0,544851"],
        ], [12.8 * cm, 3.0 * cm], ["LEFT", "RIGHT"], 8.3),
        Paragraph("O maior ganho inicial veio de afinidade. Idade e ano trouxeram uma melhora "
                  "moderada à regressão, mas uma melhora substancial ao boosting, indicando "
                  "relações ou interações que a forma linear não representa adequadamente.", body),
        Paragraph("5.1 Comparação final", heading),
        academic_table([
            ["Modelo", "RMSE treino", "RMSE validação", "Gap"],
            ["Regressão", "0,612450", "0,615032", "0,002581"],
            ["Ridge", "0,612451", "0,615032", "0,002581"],
            ["Boosting", "0,430704", "0,544699", "0,113995"],
        ], [6.0 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm],
           ["LEFT", "RIGHT", "RIGHT", "RIGHT"], 8.3),
        Paragraph("O boosting venceu nas 25 partições e nas cinco repetições. A diferença média "
                  "Boosting - regressão foi -0,070333. No teste corrigido, o intervalo de 95% "
                  "foi [-0,0808; -0,0599] e p < 10⁻¹².", body),
        Paragraph("O gap treino-validação do boosting é maior, sinal claro de maior adaptação ao "
                  "treino. Seu gap de 0,113995 é cerca de 44 vezes o gap linear de 0,002581. "
                  "Isso exige cautela, mas o RMSE de validação ficou cerca de 0,070 estrela "
                  "abaixo dos modelos lineares, de forma estável.", body),
        Paragraph("5.2 Regularização", heading),
        academic_table([
            ["alpha", "0", "0,01", "0,1", "1", "10", "100"],
            ["RMSE", "0,616080", "0,616081", "0,616087", "0,616153", "0,617612", "0,640695"],
        ], [2.2 * cm] + [2.25 * cm] * 6, ["LEFT"] + ["RIGHT"] * 6, 7.7),
        Paragraph("Ridge escolheu alpha = 0,01 nas 25 partições e permaneceu praticamente "
                  "idêntico à regressão. Penalizações maiores aumentaram o erro; não houve "
                  "evidência de benefício da regularização linear.", body),
        Paragraph("5.3 Por que o boosting supera a regressão", heading),
        Paragraph("Para identificar o que o boosting aprende, restringimos as interações "
                  "permitidas entre colunas (<i>interaction_cst</i>). Sem nenhuma interação, o "
                  "boosting fica em 0,639. Liberar apenas idade × ano o leva a 0,601, enquanto os "
                  "demais pares testados (gênero × idade, gênero × ano, afinidade × gênero, "
                  "médias × afinidade) ficam entre 0,630 e 0,636. A interação tem leitura direta: "
                  "usuários dão notas mais altas a filmes lançados quando tinham entre 10 e 20 anos "
                  "(cerca de 4,0 estrelas, contra 3,2 nas demais faixas).", body),
        Paragraph("Definimos a idade no lançamento, x = idade - (2024 - ano), e um pico "
                  "exp(-(x - c)² / 2w²), com centro c e largura w escolhidos em três folds "
                  "internos (c = 15 e w = 6 nas 25 partições). Nas mesmas partições:", body),
        academic_table([
            ["Modelo", "RMSE OOF", "Gap treino-validação"],
            ["Regressão, onze colunas", "0,615210", "0,0026"],
            ["Boosting, onze colunas (recalculado)", "0,544933", "0,1132"],
            ["Regressão, onze colunas + pico de nostalgia", "0,541704", "0,0025"],
            ["Boosting, onze colunas + idade no lançamento", "0,519532", "0,0788"],
        ], [9.0 * cm, 3.0 * cm, 3.8 * cm], ["LEFT", "RIGHT", "RIGHT"], 8.2),
        Paragraph("Com uma única coluna a mais, a regressão empata com o boosting de onze colunas "
                  "(diferença -0,003; IC de 95% [-0,013; 0,006]; p = 0,50 no teste corrigido). "
                  "Em termos de viés e variância: a regressão de referência tem viés porque seu "
                  "conjunto de hipóteses não contém a interação idade × ano. O boosting reduz esse "
                  "viés aproximando a faixa diagonal do plano idade × ano com muitos cortes "
                  "paralelos aos eixos, o que eleva a variância (gap de 0,113). Construir o "
                  "atributo reduz o mesmo viés mantendo um modelo de baixa complexidade (gap de "
                  "0,003). Com a idade no lançamento, o boosting também melhora, nas 25 partições.",
                  body),
        Paragraph("Esta análise foi feita depois de congelar a receita e é exploratória: a "
                  "validação aninhada protege a escolha de c, w e dos hiperparâmetros, mas não a "
                  "decisão de testar a nostalgia. O modelo final não foi alterado.", small),
        Paragraph("6. Modelo final e submissão", heading),
        Paragraph("O modelo final é o <b>HistGradientBoostingRegressor</b> com onze colunas "
                  "derivadas dos dez atributos brutos. "
                  "No treino completo, a busca interna selecionou 15 folhas máximas, mínimo de "
                  "60 observações por folha, sem regularização L2, taxa de aprendizado "
                  "0,05 e 200 iterações. A escolha aceita maior complexidade porque a vantagem "
                  "OOF foi ampla, consistente e permaneceu significativa após a correção.", body),
        Paragraph("Após congelar a receita, reajustamos a busca interna e o modelo nas 4000 "
                  "linhas de treino. A submissão contém 2000 previsões, sem arredondamento, "
                  "alinhadas aos identificadores e cortadas em [0, 5]. Os valores variam de "
                  "0,402621 a 5,000000, com média 3,408512.", body),
        Paragraph("7. Limitações", heading),
        academic_table([
            ["Limitação", "Consequência"],
            ["Hipóteses motivadas pelos mesmos dados", "O ganho pode conter otimismo de seleção."],
            ["Treinos sobrepostos entre folds", "As 25 diferenças não são independentes."],
            ["Gap maior no boosting", "Há maior tendência ao sobreajuste."],
            ["Dados sintéticos e sem IDs de entidades", "A conclusão vale para os atributos fornecidos."],
            ["Oráculo conhece o gerador", "RMSE 0,39 não é necessariamente atingível."],
        ], [6.0 * cm, 9.8 * cm], font_size=8.1),
        Paragraph("A correção estatística reduz o risco de exagerar a evidência, mas não elimina "
                  "o fato de que atributos e famílias foram discutidos após explorar o treino. "
                  "Depois da rodada final, apenas a análise exploratória da seção 5.3 foi feita, "
                  "sem alterar o modelo.", body),
        Paragraph("8. Conclusão", heading),
        Paragraph("A investigação mostrou que representar afinidade foi decisivo para superar a "
                  "regressão bruta. Idade, ano e contagens explicaram estrutura adicional. A receita usa "
                  "informação de todos os dez atributos brutos, e o boosting superou regressão e Ridge de maneira "
                  "ampla e estável, justificando sua escolha apesar do maior gap de generalização. "
                  "Boa parte dessa vantagem vem de aprender a interação entre idade e ano de "
                  "lançamento; representada explicitamente, ela permite que um modelo linear "
                  "alcance o mesmo erro. O projeto termina com uma receita congelada, validação reproduzível e submissão "
                  "verificada.", body),
        Paragraph("<b>Referências.</b> Abu-Mostafa, Magdon-Ismail e Lin, <i>Learning from Data</i> "
                  "(2012). James, Witten, Hastie e Tibshirani, <i>An Introduction to Statistical "
                  "Learning</i>, 2ª ed. (2023). Materiais da disciplina MAE0501 (2026).", small),
    ]

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
