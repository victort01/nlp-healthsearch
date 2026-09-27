"""Executa buscas reais e salva comparação reprodutível (inclui falhas)."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from healthsearch_app import pesquisar, reranquear, MODELO, RERANKER

CASOS=[
    ('ataque cardíaco','Doc 1'),
    ('CÓD-ECG-12D','Doc 1'),
    ('P2Y12','Doc 2'),
    ('vigilância remota dos batimentos de paciente internado','Doc 6'),
    ('o coração parou de bater e a pessoa não reage','Doc 5'),
    ('pressão arterial muito elevada','Doc 3'),
    ('derrame com início recente dos sintomas','Doc 4'),
    ('antiagregantes plaquetários','Doc 2'),
    ('monitoramento de arritmias por telemetria','Doc 6'),
    ('compressões torácicas durante reanimação','Doc 5'),
    ('trombólise no acidente vascular cerebral','Doc 4'),
    ('tratamento venoso de crise hipertensiva','Doc 3'),
]

def main():
    out=Path(__file__).parent/'resultados';out.mkdir(exist_ok=True)
    linhas=[]
    for consulta,esperado in CASOS:
        df,_=pesquisar(consulta)
        correto=df.set_index('ID').loc[esperado]
        linhas.append({'consulta':consulta,'referencia_didatica':esperado,**{col:int(correto[col]) for col in ['Rank BM25','Rank Semântico','Rank RRF']}})
    tabela=pd.DataFrame(linhas)
    tabela.to_csv(out/'comparacao_ranks.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,3.5))
    for col in ['Rank BM25','Rank Semântico','Rank RRF']:
        ax.plot(range(1,len(tabela)+1),tabela[col],marker='o',label=col)
    ax.set(xticks=range(1,len(tabela)+1),yticks=range(1,7),xlabel='Consulta (ordem da tabela)',ylabel='Posição do documento de referência')
    ax.invert_yaxis();ax.legend();fig.tight_layout();fig.savefig(out/'comparacao_ranks.png',dpi=170)
    df,_=pesquisar('CÓD-ECG-12D')
    bonus=reranquear(df,'CÓD-ECG-12D')
    bonus.to_csv(out/'cross_encoder_top3.csv',index=False)
    resumo={'modelo':MODELO,'cross_encoder':RERANKER,'corpus':'documentos expandidos incorporados',
            'parametros':{'k1':1.2,'b':.75,'alpha':.5,'k_rrf':60},'consultas':len(CASOS),
            'acertos_top1':{col:int((tabela[col]==1).sum()) for col in ['Rank BM25','Rank Semântico','Rank RRF']},
            'nota':'Referências didáticas exploratórias, não validação clínica; a consulta por código também é relevante ao Doc 6. Não é benchmark cego.'}
    (out/'metricas.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumo,ensure_ascii=False))

if __name__=='__main__':main()
