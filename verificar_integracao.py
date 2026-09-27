"""Verificação opcional de modelos e interface reais, fora do CI leve."""
from pathlib import Path
import json
import numpy as np
from streamlit.testing.v1 import AppTest
from healthsearch_app import pesquisar, reranquear, ranks

def main():
    for alpha in [0.,.5,1.]:
        df,_=pesquisar('CÓD-ECG-12D',k1=0,b=0,alpha=alpha)
        assert np.isfinite(df[['BM25','Cosseno','RRF']].to_numpy()).all()
        if alpha==0: assert (df['Rank RRF']==df['Rank Semântico']).all()
        if alpha==1: assert (df['Rank RRF']==df['Rank BM25']).all()
    df,_=pesquisar('ataque cardíaco')
    top=reranquear(df,'ataque cardíaco')
    assert len(top)==3 and set(top.ID)==set(df.sort_values('Rank RRF').head(3).ID)
    base=Path(__file__).parent
    app=AppTest.from_file(str(base/'healthsearch_app.py'),default_timeout=300).run()
    assert not app.exception and not app.error
    app.slider[0].set_value(0.).run()
    app.slider[1].set_value(0.).run()
    app.slider[2].set_value(1.).run()
    app.checkbox[0].check().run()
    assert not app.exception and not app.error
    app.text_input[0].set_value('').run()
    assert not app.exception and len(app.warning)>0
    (base/'resultados'/'integracao.json').write_text(json.dumps({'status':'aprovado','modelos_reais':True,'cross_encoder_top3':True,'interface':'AppTest: sliders, bônus e consulta vazia','extremos_rrf':[0,.5,1]},indent=2),encoding='utf-8')
    print('Integração HealthSearch aprovada.')

if __name__=='__main__':main()
