import math
import numpy as np
import pytest
from healthsearch_app import scores_bm25, ranks, rrf, tokenizar, CORPUS, PAGINAS_EXPANDIDAS


def test_bm25_manual():
    corpus=['soja soja milho','agua','terra']
    esperado=math.log(2.5/1.5)*(2*2.2)/(2+1.2*(.25+.75*3/(5/3)))
    assert scores_bm25(corpus,'soja')[0] == pytest.approx(esperado)
    assert scores_bm25(corpus,'soja',0,1)[0] == pytest.approx(math.log(2.5/1.5))
    assert scores_bm25(corpus,'soja',0,1)[1] == 0


@pytest.mark.parametrize('k1',[0,1.2,3])
@pytest.mark.parametrize('b',[0,.75,1])
def test_limites(k1,b):
    s=scores_bm25(['soja soja','milho','agua'],'soja',k1,b)
    assert np.isfinite(s).all()
    assert not np.any(scores_bm25(['a','de'],'soja',k1,b))


def test_formula_rrf_e_extremos():
    lex=[3,1,2];sem=[.2,.9,.1]
    assert ranks(lex).tolist()==[1,3,2]
    assert rrf(lex,sem,.25)[0] == pytest.approx(.25/61+.75/62)
    np.testing.assert_array_equal(ranks(rrf(lex,sem,1)),ranks(lex))
    np.testing.assert_array_equal(ranks(rrf(lex,sem,0)),ranks(sem))
    assert ranks([0,0,0]).tolist()==[1,2,3]
    with pytest.raises(ValueError):rrf(lex,sem,2)


def test_codigo_e_corpus():
    assert tokenizar('CÓD-ECG-12D, AAS 100mg!')==['cod-ecg-12d','aas','100mg']
    assert len(CORPUS)==6
    for id_,_,frase in CORPUS:
        paginas=PAGINAS_EXPANDIDAS[id_]
        assert len(paginas)==2
        assert ' '.join(frase.split()) in ' '.join(' '.join(paginas).split())
