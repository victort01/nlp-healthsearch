"""Copiado para cada repositório: gera corpus e relatório a partir das fontes locais."""
from pathlib import Path
from xml.sax.saxutils import escape
import ast
import json
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE=Path(__file__).resolve().parent
regular=Path('C:/Windows/Fonts/arial.ttf')
bold=Path('C:/Windows/Fonts/arialbd.ttf')
if regular.exists() and bold.exists():
    pdfmetrics.registerFont(TTFont('Corpo',str(regular)))
    pdfmetrics.registerFont(TTFont('CorpoBold',str(bold)))
else:
    regular=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    if regular.exists():
        pdfmetrics.registerFont(TTFont('Corpo',str(regular)))
        pdfmetrics.registerFont(TTFont('CorpoBold',str(bold)))
    else:
        from reportlab.pdfbase.pdfmetrics import Font
        pdfmetrics.registerFont(Font('Corpo','Helvetica','WinAnsiEncoding'))
        pdfmetrics.registerFont(Font('CorpoBold','Helvetica-Bold','WinAnsiEncoding'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TituloDoc',fontName='CorpoBold',fontSize=20,leading=24,spaceAfter=14,textColor=colors.HexColor('#16374B')))
styles.add(ParagraphStyle(name='SecaoDoc',fontName='CorpoBold',fontSize=11.5,leading=15,spaceBefore=11,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='TextoDoc',fontName='Corpo',fontSize=10.2,leading=14.7,spaceAfter=8))
styles.add(ParagraphStyle(name='NotaDoc',fontName='Corpo',fontSize=8.4,leading=11,spaceAfter=8,textColor=colors.HexColor('#44515C')))

def par(texto,tipo='TextoDoc'):
    return Paragraph(escape(texto).replace('\n','<br/>'),styles[tipo])

def rodape(canvas,doc):
    canvas.saveState();canvas.setFont('Corpo',8);canvas.setFillColor(colors.HexColor('#44515C'))
    canvas.drawString(44,25,'UNIPÊ · Tendências em Ciência da Computação · Material acadêmico')
    canvas.drawRightString(A4[0]-44,25,str(doc.page));canvas.restoreState()

def escrever(path,story):
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=44,title=path.stem,author='Paulo Victor').build(story,onFirstPage=rodape,onLaterPages=rodape)

def constante(arquivo,nome):
    for node in ast.parse(arquivo.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==nome for t in node.targets):return ast.literal_eval(node.value)
    raise ValueError(nome)

def corpus():
    (BASE/'corpus').mkdir(exist_ok=True)
    agro=BASE/'agrosearch_app.py';health=BASE/'healthsearch_app.py'
    if agro.exists():
        for i,txt in enumerate(constante(agro,'DOCUMENTOS'),1):
            titulo,_,corpo=txt.partition('\n')
            story=[par(f'Documento {i}','NotaDoc'),par(titulo,'TituloDoc'),par('Texto ampliado com auxílio de IA a partir do tema do enunciado. Uso didático para recuperação de informação.','NotaDoc')]
            story += [par(x.strip()) for x in corpo.split('\n\n') if x.strip()]
            escrever(BASE/'corpus'/f'doc_{i:02d}.pdf',story)
    if health.exists():
        for i,(id_,paginas) in enumerate(constante(health,'PAGINAS_EXPANDIDAS').items(),1):
            story=[]
            for j,txt in enumerate(paginas):
                if j:story.append(PageBreak())
                titulo,_,corpo=txt.partition('\n')
                story+=[par(f'{id_} · Parte {j+1} de 2','NotaDoc'),par(titulo,'TituloDoc'),par('Corpus didático ampliado com IA. Não é diretriz clínica validada nem orientação de tratamento.','NotaDoc')]
                # Textos incorporados mantêm linhas de extração; agrupá-las evita linhas curtas artificiais.
                linhas=corpo.splitlines();buffer=[]
                for linha in linhas:
                    if len(linha)<72 and not linha.endswith(('.',',',':',']')) and (not buffer or buffer[-1].endswith(('.',']'))):
                        if buffer:story.append(par(' '.join(buffer)));buffer=[]
                        story.append(par(linha,'SecaoDoc'))
                    else:buffer.append(linha)
                if buffer:story.append(par(' '.join(buffer)))
            escrever(BASE/'corpus'/f'doc_{i:02d}.pdf',story)

def relatorio():
    config=BASE/'relatorio.json'
    if not config.exists():return
    dados=json.loads(config.read_text(encoding='utf-8'));story=[]
    for i,pagina in enumerate(dados['paginas']):
        if i:story.append(PageBreak())
        story.append(par(pagina['titulo'],'TituloDoc'))
        for bloco in pagina['blocos']:
            if 'imagem' in bloco:
                from PIL import Image as PILImage
                path=BASE/bloco['imagem']
                w,h=PILImage.open(path).size
                width=bloco.get('largura',480)
                story.append(Image(str(path),width=width,height=width*h/w))
            else:story.append(par(bloco['texto'],bloco.get('estilo','TextoDoc')))
    escrever(BASE/dados['arquivo'],story)

if __name__=='__main__':
    if (BASE/'agrosearch_app.py').exists() or (BASE/'healthsearch_app.py').exists():corpus()
    relatorio()
