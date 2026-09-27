"""HealthSearch: arquivo único com corpus obrigatório, BM25, embeddings e RRF."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')
import re
import time
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

MODELO = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANKER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
CORPUS = [
    ('Doc 1', 'Protocolo Emergência ECG', 'Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos.'),
    ('Doc 2', 'Guia de Farmacologia Cardíaca', 'O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio.'),
    ('Doc 3', 'Diretriz de Hipertensão Arterial', 'A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI.'),
    ('Doc 4', 'Manual de AVC Isquêmico', 'O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas.'),
    ('Doc 5', 'Protocolo de Reanimação RCR', 'Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul.'),
    ('Doc 6', 'Procedimentos de UTI Geral', 'Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria.'),
]
# Duas páginas por tema, incorporadas da proposta anterior do próprio projeto.
PAGINAS_EXPANDIDAS = {'Doc 1': ('Protocolo Emergência ECG\nFrase de contexto original\nPacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar\neletrocardiograma CÓD-ECG-12D em até 10 minutos.\nAvaliação inicial da dor precordial\nA dor precordial aguda exige avaliação rápida para identificar situações de risco. Este documento\naborda o papel do eletrocardiograma na investigação de síndrome coronariana aguda, condição\nque inclui apresentações de infarto agudo do miocárdio. A expressão popular ataque cardíaco\ncostuma se referir ao infarto, mas a confirmação depende da avaliação clínica e dos exames.\nO eletrocardiograma de 12 derivações registra a atividade elétrica cardíaca e ajuda a reconhecer\nalterações compatíveis com isquemia miocárdica. A diretriz de dor torácica recomenda obter e\nrevisar o ECG em até 10 minutos da chegada. O prazo orienta a organização do atendimento; não\nrepresenta um período que se deva aguardar antes de iniciar a avaliação. [1]\nIdentificação e qualidade do exame\nNeste corpus, CÓD-ECG-12D é o identificador fornecido para o exame de 12 derivações. Ele não\ndeve ser interpretado como uma codificação clínica universal. A solicitação, o traçado e o registro\ndo atendimento precisam estar associados ao paciente correto.\nO registro pode reunir horário de chegada, início dos sintomas, horário da aquisição, qualidade do\ntraçado e profissional que realizou a leitura. Esses campos permitem reconstruir a sequência do\natendimento e verificar se o exame ficou disponível no momento necessário. Um arquivo sem\nidentificação ou sem horário reduz a utilidade da informação para a equipe que recebe o caso.', 'Protocolo Emergência ECG\nInterpretação no contexto clínico\nO resultado do ECG deve ser analisado junto da história, dos sintomas e da avaliação de risco. Um\ntraçado inicial sem alterações diagnósticas não encerra a investigação quando a suspeita\npermanece. ECGs seriados podem ser necessários. A dosagem de troponina, preferencialmente de\nalta sensibilidade quando disponível, integra os caminhos diagnósticos de dor torácica. [1, 2]\nA avaliação também considera desconforto em outras regiões do tórax, dispneia e sintomas\nassociados. A equipe utiliza um percurso estruturado para decidir observação, exames adicionais e\nencaminhamento. O exame elétrico e os biomarcadores oferecem informações complementares;\nnenhum resultado isolado resume toda a situação do paciente. [2]\nExemplo fictício de registro\nUm adulto chega à emergência às 14h00 com aperto no peito iniciado durante uma caminhada. O\nregistro informa dor precordial e suspeita de síndrome coronariana. O exame CÓD-ECG-12D é\nadquirido às 14h07 e encaminhado para leitura. O exemplo demonstra um intervalo de sete minutos\nentre chegada e aquisição, sem pressupor um diagnóstico a partir desse prazo.\nNa evolução, a equipe descreve a interpretação, os sintomas no momento do exame e o plano de\ninvestigação. Se houver outro traçado, ambos recebem horários próprios. Assim, o documento\nconserva a relação entre queixa, exame solicitado e informação produzida, permitindo comparar\nepisódios sem confundir registros de momentos distintos.\nVocabulário do documento\nDor precordial, dor torácica, aperto no peito, síndrome coronariana aguda, isquemia miocárdica,\ninfarto, ataque cardíaco, eletrocardiograma de 12 derivações e CÓD-ECG-12D aparecem\nrelacionados ao mesmo cenário de avaliação inicial.'), 'Doc 2': ('Guia de Farmacologia Cardíaca\nFrase de contexto original\nO uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto\nagudo do miocárdio.\nAntiagregação no infarto\nEste guia apresenta a função dos antiagregantes plaquetários no cuidado ao paciente com infarto\nagudo do miocárdio. O ácido acetilsalicílico, também chamado AAS ou aspirina, integra essa\nclasse. Portanto, a frase de contexto cita um medicamento e o grupo ao qual ele pertence, e não\nduas categorias independentes.\nAs plaquetas participam da formação de coágulos. Em parte dos eventos coronarianos, um coágulo\ncompromete o fluxo de sangue para o músculo cardíaco. A antiagregação ajuda a limitar a\nformação de trombos e pode integrar o tratamento da síndrome coronariana aguda. O benefício\nprecisa ser avaliado junto ao risco de sangramento e às condições individuais. [1]\nAssociação de medicamentos\nA dupla antiagregação plaquetária combina AAS com um inibidor de P2Y12. A diretriz de 2025\nconsidera essa associação por pelo menos 12 meses como estratégia padrão após síndrome\ncoronariana aguda em pacientes sem alto risco hemorrágico. Outras estratégias podem ser\nescolhidas conforme o risco e a situação clínica. [2]\nO tratamento do infarto também pode envolver procedimentos para restaurar o fluxo coronariano e\noutras classes de medicamentos. A prescrição depende da apresentação do quadro e da estratégia\ndefinida pela equipe. A frase de contexto não fornece dose, via ou duração suficientes para\nconstituir uma prescrição individual. [2]\nInformação farmacológica organizada\nUm registro útil distingue princípio ativo, apresentação, dose prescrita, via, horário e motivo de uso.\nA palavra aspirina pode aparecer na história relatada pelo paciente, enquanto ácido acetilsalicílico\naparece na prescrição; ambas devem ser compreendidas como nomes relacionados ao AAS.', 'Guia de Farmacologia Cardíaca\nSegurança e acompanhamento\nAlergia ou intolerância ao AAS, história de sangramento e uso concomitante de outros\nmedicamentos precisam ser comunicados à equipe. Anti-inflamatórios e outros produtos podem\ninterferir na segurança da terapia. A utilização diária de aspirina deve ser orientada por profissional\nde saúde, pois o balanço de riscos e benefícios varia entre pessoas. [1]\nA decisão de manter ou modificar a dupla antiagregação deve considerar a evolução clínica e o\nrisco hemorrágico. Após a alta, a continuidade do cuidado inclui revisão dos medicamentos e\nencaminhamento para reabilitação cardíaca quando indicada. A diretriz de síndrome coronariana\naguda destaca a importância dessas medidas na recuperação. [2]\nExemplo fictício de conciliação\nDurante uma internação por infarto, o paciente informa que toma aspirina em casa, mas não lembra\na apresentação. A equipe registra o relato como informação ainda a confirmar e consulta a lista de\nmedicamentos disponível. O nome informado pelo paciente é relacionado a ácido acetilsalicílico,\nevitando tratar os dois termos como substâncias diferentes.\nDepois da avaliação, a prescrição e os horários administrados são documentados separadamente\ndo histórico domiciliar. Se o plano mudar, o registro informa quando ocorreu a alteração e qual\norientação foi transmitida. O exemplo ilustra a organização da informação, sem estabelecer uma\ndose ou recomendar uma combinação para um paciente real.\nTerminologia essencial\nInfarto agudo do miocárdio, IAM e ataque cardíaco são expressões que podem surgir no mesmo\natendimento. AAS, aspirina e ácido acetilsalicílico identificam o medicamento central deste guia.\nAntiagregante plaquetário e inibidor de P2Y12 descrevem classes e funções relacionadas, mas não\nsão nomes intercambiáveis de um único produto.'), 'Doc 3': ('Diretriz de Hipertensão Arterial\nFrase de contexto original\nA crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento\ncontínuo da pressão arterial na UTI.\nGravidade e lesão aguda de órgãos\nEste documento desenvolve o cenário de hipertensão arterial grave em atendimento hospitalar. A\nfrase de contexto precisa de uma qualificação: a indicação de terapia intravenosa e cuidados\nintensivos está ligada à emergência hipertensiva com lesão aguda de órgão-alvo. Uma medida\nelevada, isoladamente, não determina essa conduta. [1]\nA avaliação diferencia elevação pressórica sem lesão aguda de situações com comprometimento\nnovo ou em piora de órgãos. Essa distinção orienta a urgência e a intensidade do tratamento. O\nquadro clínico, a qualidade da medida e as condições associadas são parte da decisão; reduzir um\nnúmero no monitor não resume o objetivo do cuidado. [1]\nMedição e avaliação inicial\nA pressão arterial deve ser aferida com técnica apropriada. Tamanho inadequado do manguito,\nposição do braço e condições da medição podem alterar o resultado. Medidas repetidas e\ncorretamente registradas ajudam a interpretar a persistência da elevação e a resposta ao cuidado.\n[2]\nNa avaliação inicial, a equipe reúne sintomas, histórico de hipertensão, medicamentos em uso e\ninformações sobre adesão. O registro deve separar o valor observado da interpretação clínica.\nEscrever apenas pressão alta dificulta compreender se havia um achado transitório, uma condição\ncrônica descompensada ou uma emergência com comprometimento de órgãos.\nContexto da terapia venosa\nNa emergência hipertensiva, o ambiente de cuidado precisa permitir vigilância e ajustes do\ntratamento. A monitorização arterial invasiva pode ser preferida para acompanhar a queda\npressórica durante o uso de anti-hipertensivos intravenosos. A escolha da abordagem depende da\nsituação clínica e dos recursos necessários. [2]', 'Diretriz de Hipertensão Arterial\nTratamento individualizado\nO plano terapêutico deve considerar a presença de lesão aguda, as condições associadas e a\nevolução do paciente. A declaração da AHA sobre pressão elevada no hospital recomenda cautela\ncom medicamentos, especialmente intravenosos, e enfatiza a individualização do cuidado. Não se\ndeve transformar toda elevação pressórica em indicação automática de tratamento intensivo. [1]\nEm adultos não gestantes com hipertensão grave e sem evidência de lesão aguda de órgão-alvo, a\ndiretriz de 2025 orienta avaliação e tratamento oportunos, com início, retomada ou intensificação de\nterapia oral no contexto ambulatorial apropriado. Isso reforça que a via venosa depende da\ncaracterização da emergência. [3]\nExemplo fictício de documentação\nUma pessoa chega ao hospital com uma medida de pressão muito elevada obtida em casa. A\nequipe repete a aferição e registra as condições em que ela foi realizada. O prontuário descreve os\nsintomas investigados, os medicamentos relatados e a avaliação de possíveis lesões agudas, sem\nconcluir pela necessidade de UTI apenas com base na medida inicial.\nCaso o atendimento confirme emergência hipertensiva, a evolução passa a registrar o órgão\ncomprometido, o plano prescrito e as medidas subsequentes com seus horários. Se essa condição\nnão for identificada, o documento registra a avaliação e o acompanhamento definidos. O exemplo\nevidencia por que o mesmo termo pressão alta pode representar cenários diferentes.\nContinuidade do cuidado\nO resumo de transferência ou alta pode reunir a hipótese diagnóstica, a sequência de medidas, os\nmedicamentos revisados e o plano de seguimento. Para manter clareza, expressões como crise\nhipertensiva, emergência hipertensiva, hipertensão grave e pressão elevada devem vir\nacompanhadas da descrição clínica correspondente.'), 'Doc 4': ('Manual de AVC Isquêmico\nFrase de contexto original\nO acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até\nquatro horas e meia do início dos sintomas.\nReconhecimento e avaliação urgente\nO acidente vascular cerebral isquêmico ocorre quando a circulação para uma região do cérebro é\nobstruída. Este manual apresenta o contexto da avaliação urgente e da seleção para tratamento de\nreperfusão. A expressão derrame cerebral é frequente na linguagem cotidiana, mas não distingue o\nAVC isquêmico do hemorrágico. [1]\nO atendimento investiga rapidamente a natureza do evento e as opções de tratamento. Exames de\nimagem ajudam a identificar se há sangramento ou obstrução e orientam as decisões. O tempo de\ninício dos sintomas, ou o último momento em que a pessoa estava bem, é uma informação central\npara a equipe. [1]\nSignificado da janela de quatro horas e meia\nA frase de contexto não representa uma indicação universal. Na diretriz de 2026, a trombólise\nintravenosa com alteplase ou tenecteplase é uma opção para pacientes elegíveis, com déficits\nincapacitantes, dentro da janela de 4,5 horas. A equipe verifica os critérios clínicos e as\ncontraindicações antes de administrar o tratamento. [2]\nO objetivo é tratar o paciente elegível o mais cedo possível. A existência de uma janela não significa\naguardar seu limite. Há também situações selecionadas de início desconhecido ou janela ampliada\nnas quais critérios de imagem podem permitir trombólise. Esses casos exigem avaliação\nespecializada e não podem ser decididos apenas pelo horário. [2]\nRegistro do tempo\nO prontuário deve distinguir horário de descoberta dos sintomas, início conhecido e último momento\nsem alterações. Se o início for incerto, a incerteza precisa permanecer explícita. Substituir um\nhorário desconhecido por uma estimativa não confirmada pode distorcer a interpretação do\natendimento.', 'Manual de AVC Isquêmico\nReperfusão e acompanhamento\nA trombólise utiliza medicamento para dissolver o coágulo. A trombectomia é um procedimento para\nremovê-lo e pode ser indicada em pacientes selecionados com oclusão de grandes vasos. As\nopções são avaliadas conforme exames, características do evento e elegibilidade, dentro da\norganização da rede de atendimento. [1, 2]\nNem todo déficit neurológico dentro de 4,5 horas leva à trombólise. A diretriz diferencia\napresentações incapacitantes de déficits não incapacitantes e aborda escolhas específicas para\nesses grupos. Assim, a frase original funciona como uma síntese do tema, enquanto a decisão\ndepende da avaliação completa. [2]\nExemplo fictício de linha do tempo\nUma pessoa é vista sem alterações às 8h00 e, às 8h20, um familiar percebe dificuldade para falar e\nmovimentar um braço. No atendimento, os dois horários são registrados com a indicação de quem\nforneceu a informação. O caso é encaminhado à avaliação de AVC e à investigação por imagem.\nO registro não presume que a observação do familiar determine, por si só, o tratamento. A equipe\ndocumenta os achados, a elegibilidade e a decisão tomada. Se houver transferência, a linha do\ntempo acompanha o paciente para que o serviço de destino compreenda quais informações foram\nconfirmadas e quais permanecem incertas.\nComunicação entre equipes\nUm resumo útil apresenta o déficit observado, a evolução, os horários e os exames realizados. A\nexpressão AVC isquêmico agudo deve permanecer distinta de suspeita de AVC até a avaliação\ndiagnóstica correspondente. Termos como trombólise venosa, reperfusão e trombectomia precisam\nindicar procedimentos diferentes, embora relacionados ao restabelecimento da circulação cerebral.'), 'Doc 5': ('Protocolo de Reanimação RCR\nFrase de contexto original\nParada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e\ndesfibrilação precoce no código azul.\nReconhecimento e resposta da equipe\nEste protocolo didático aborda a resposta à parada cardiorrespiratória em adultos. A reanimação\ncardiopulmonar, também descrita como ressuscitação cardiopulmonar ou RCP, combina\nintervenções para sustentar a circulação e a oxigenação. O título RCR foi preservado conforme o\ndocumento de origem.\nNo algoritmo para profissionais de saúde, a pessoa é avaliada quanto à resposta, respiração e\npulso. Ausência de respiração normal, incluindo respiração agônica, e ausência de pulso definido\nem até 10 segundos orientam o início da RCP. A equipe aciona a resposta de emergência e\nprovidencia o desfibrilador. [1]\nCompressões de alta qualidade\nAs diretrizes de 2025 indicam frequência de 100 a 120 compressões por minuto em adultos. A\nprofundidade deve ser de pelo menos 5 cm, evitando ultrapassar 6 cm, com retorno completo do\ntórax e interrupções mínimas. A qualidade das compressões é parte essencial da ressuscitação. [2]\nNa frase original, contínuas deve ser entendido como esforço para reduzir pausas evitáveis, sem\neliminar as etapas necessárias do algoritmo. Antes de uma via aérea avançada, profissionais\npodem realizar ciclos de 30 compressões e duas ventilações. A ventilação deve produzir elevação\nvisível do tórax, evitando excesso. [1, 2]\nOrganização do código azul\nNeste cenário, código azul identifica o acionamento da equipe hospitalar de emergência. A divisão\nde funções pode contemplar compressões, ventilação, operação do desfibrilador e registro dos\neventos. A comunicação deve permitir que cada ação seja anunciada e compreendida pelos\nprofissionais envolvidos.', 'Protocolo de Reanimação RCR\nDesfibrilação conforme o ritmo\nA desfibrilação precoce se aplica aos ritmos chocáveis, como fibrilação ventricular e taquicardia\nventricular sem pulso. Assistolia e atividade elétrica sem pulso não são tratadas com choque. Por\nisso, a frase de contexto não deve ser interpretada como indicação de desfibrilar toda parada\ncardiorrespiratória. [3]\nNo algoritmo de suporte básico, após o choque indicado, a RCP é retomada imediatamente por\ncerca de dois minutos até a próxima análise. Quando o ritmo não é chocável, as compressões\ntambém são retomadas. O desfibrilador externo automático orienta a análise e informa se há\nindicação de choque. [1]\nExemplo fictício de coordenação\nDurante um plantão, a equipe identifica um adulto inconsciente e inicia a avaliação prevista no\nalgoritmo. Um profissional aciona o código azul, outro providencia o desfibrilador e os demais\nassumem as funções definidas. O responsável pelo registro anota os horários das avaliações e\nintervenções anunciadas.\nO relato do evento distingue a chegada do equipamento da aplicação efetiva de um choque.\nTambém informa o ritmo identificado e a sequência das ações. Essa separação evita que a\nexpressão desfibrilador disponível seja confundida com desfibrilação realizada, o que prejudicaria a\nreconstrução do atendimento.\nRegistro e revisão do evento\nUm formulário de reanimação pode reunir horário de reconhecimento, início das compressões,\nritmos analisados, choques administrados e desfecho observado. Cada informação deve\ncorresponder a um evento realmente identificado. Na revisão do caso, a equipe pode avaliar a\ncomunicação e a sequência temporal sem transformar o exemplo em um resultado clínico\npresumido.\nParada cardiorrespiratória, PCR, reanimação, ressuscitação, compressões torácicas, RCP, RCR,\ndesfibrilação e código azul compõem o vocabulário relacionado a este documento.'), 'Doc 6': ('Procedimentos de UTI Geral\nFrase de contexto original\nPara diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a\nmonitorização cardíaca contínua por telemetria.\nMonitorização cardíaca na terapia intensiva\nEste documento aborda o acompanhamento do ritmo cardíaco de pacientes hospitalizados, com\nfoco em arritmias complexas e documentação dos eventos. Embora o título seja amplo, o conteúdo\ndesenvolve o tema específico da frase de contexto: a relação entre registro eletrocardiográfico e\nobservação contínua.\nA monitorização contínua pode auxiliar na detecção de arritmias, de deterioração clínica e de\nalterações que exigem avaliação. Também pode relacionar sintomas como palpitações ou síncope\nao ritmo registrado. A indicação e a duração devem considerar as características do paciente, com\nobjetivos definidos para o acompanhamento. [1]\nECG de 12 derivações e telemetria\nCÓD-ECG-12D é um identificador fornecido pelo corpus, não uma doença ou classificação\ndiagnóstica universal. A expressão diagnóstico do protocolo, presente na frase original, é imprecisa:\no diagnóstico diz respeito à condição clínica, enquanto o código identifica um exame dentro do\nmaterial.\nO ECG de 12 derivações apresenta um registro da atividade elétrica em determinado momento. A\ntelemetria permite acompanhar sinais ao longo do tempo por transmissão a um sistema de\nmonitorização. São recursos distintos e complementares; a menção ao mesmo código não\ntransforma telemetria e ECG convencional no mesmo procedimento. [1, 2]\nObjetivo do acompanhamento\nNa admissão ou transferência, o registro pode informar o motivo da monitorização, os sintomas de\ninteresse e quais episódios devem ser comunicados. Isso ajuda a equipe seguinte a compreender\nse o acompanhamento foi iniciado por um evento já documentado, por sintomas intermitentes ou\npor outra indicação estabelecida.', 'Procedimentos de UTI Geral\nQualidade do sinal e alarmes\nArtefatos de movimento, ruído e falhas de eletrodos podem gerar registros enganosos e alarmes. A\ninterpretação deve levar em conta a qualidade do sinal e a avaliação do paciente. Um alarme é um\naviso que exige verificação, não uma confirmação automática de arritmia. [2]\nAs recomendações de monitorização hospitalar incluem gestão de alarmes, educação da equipe e\ndocumentação. O acompanhamento pode ter objetivos relacionados a arritmias, isquemia ou\nintervalo QT, conforme a indicação. O plano precisa explicitar o que está sendo observado para que\no sistema e a resposta da equipe atendam à necessidade clínica. [1]\nExemplo fictício de correlação clínica\nUm paciente em UTI relata palpitações durante a madrugada. O profissional registra o horário do\nsintoma e conserva o trecho correspondente da telemetria. Na avaliação, um ECG CÓD-ECG-12D\né solicitado e associado ao mesmo episódio, mantendo a identificação de cada modalidade de\nregistro.\nO relatório informa se o sintoma ocorreu durante o traçado armazenado e qual interpretação foi\ndocumentada. Um ECG obtido depois não é descrito como se tivesse sido capturado no instante da\nqueixa. A sequência permite comparar o acompanhamento contínuo com o exame pontual e\nexplicita os limites temporais de cada informação.\nPassagem de plantão\nO resumo do caso pode reunir indicação da monitorização, episódios relevantes, sintomas\nassociados, qualidade dos registros e decisões comunicadas. Se houver revisão de um alarme\ninicialmente interpretado como arritmia, o motivo da revisão deve constar na evolução, preservando\na rastreabilidade do evento.\nUTI, unidade de terapia intensiva, telemetria, monitorização cardíaca contínua, arritmia complexa,\npalpitação, traçado e CÓD-ECG-12D são termos associados a este cenário. O significado de cada\nregistro depende de seu objetivo e do momento em que foi obtido.')}

STOPWORDS = set('a o as os um uma de da do das dos em na no nas nos e ou que se ao aos por para com pelo pela ser ate durante'.split())


def tokenizar(texto):
    limpo = ''.join(c for c in unicodedata.normalize('NFD', texto.lower()) if unicodedata.category(c) != 'Mn')
    # Hífens internos preservam códigos completos como CÓD-ECG-12D.
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', limpo)
    return [t for t in tokens if t not in STOPWORDS]


def scores_bm25(textos, consulta, k1=1.2, b=.75):
    if not (0 <= k1 <= 3 and 0 <= b <= 1):
        raise ValueError('k1 deve estar em [0,3] e b em [0,1].')
    corpus = [tokenizar(t) for t in textos]
    if not corpus or not any(corpus):
        return np.zeros(len(corpus))
    motor = BM25Okapi(corpus, k1=k1, b=b)
    query = tokenizar(consulta)
    if k1 == 0:
        # Limite matemático: termo ausente contribui zero, evitando 0/0 da biblioteca.
        return np.array([sum(motor.idf.get(t, 0.0) for t in query if t in doc) for doc in corpus])
    return motor.get_scores(query)


def ranks(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Scores devem ser finitos.')
    ordem = np.argsort(-scores, kind='stable')
    posicoes = np.empty(len(scores), dtype=int)
    posicoes[ordem] = np.arange(1, len(scores)+1)
    return posicoes


def rrf(lexico, semantico, alpha=.5):
    if not 0 <= alpha <= 1 or len(lexico) != len(semantico):
        raise ValueError('Peso ou dimensões inválidos.')
    return alpha / (60 + ranks(lexico)) + (1-alpha) / (60 + ranks(semantico))


@st.cache_resource(show_spinner=False)
def encoder():
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO, device='cpu', trust_remote_code=False)


@st.cache_resource(show_spinner=False)
def cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER, device='cpu', trust_remote_code=False)


@st.cache_data(show_spinner=False)
def embeddings(textos):
    return encoder().encode(list(textos), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


@st.cache_data(show_spinner=False)
def dividir_corpus(textos):
    """Janelas sobrepostas de tokens, respeitando o limite do encoder."""
    m = encoder()
    limite = m.max_seq_length - m.tokenizer.num_special_tokens_to_add(pair=False)
    trechos, origens = [], []
    for i,texto in enumerate(textos):
        ids = m.tokenizer.encode(texto, add_special_tokens=False)
        for inicio in range(0,len(ids),max(1,limite-24)):
            trechos.append(m.tokenizer.decode(ids[inicio:inicio+limite],skip_special_tokens=True))
            origens.append(i)
            if inicio+limite >= len(ids): break
    return tuple(trechos), tuple(origens)


def pesquisar(consulta, k1=1.2, b=.75, alpha=.5):
    if not consulta.strip():
        raise ValueError('Digite uma consulta.')
    textos = tuple('\n\n'.join(PAGINAS_EXPANDIDAS[d[0]]) for d in CORPUS)
    inicio = time.perf_counter()
    lex = scores_bm25(textos, consulta, k1, b)
    meio = time.perf_counter()
    trechos, origens = dividir_corpus(textos)
    scores_trechos = np.clip(embeddings(trechos) @ embeddings((consulta,))[0], -1, 1)
    melhores = [max((j for j,o in enumerate(origens) if o == i), key=lambda j: scores_trechos[j]) for i in range(len(textos))]
    sem = np.array([scores_trechos[j] for j in melhores])
    fim = time.perf_counter()
    fusao = rrf(lex, sem, alpha)
    df = pd.DataFrame(CORPUS, columns=['ID','Título','Texto'])
    df['Trecho recuperado'] = [trechos[j] for j in melhores]
    df['BM25'], df['Cosseno'], df['RRF'] = lex, sem, fusao
    df['Rank BM25'], df['Rank Semântico'], df['Rank RRF'] = ranks(lex), ranks(sem), ranks(fusao)
    df['Parcela BM25'] = alpha / (60+df['Rank BM25'])
    df['Parcela semântica'] = (1-alpha) / (60+df['Rank Semântico'])
    return df, {'BM25 (ms)': (meio-inicio)*1000, 'Semântico (ms)': (fim-meio)*1000}


def reranquear(df, consulta):
    candidatos = df.sort_values('Rank RRF').head(3).copy()
    candidatos['Score Cross-Encoder'] = cross_encoder().predict([(consulta, t) for t in candidatos['Trecho recuperado']], show_progress_bar=False)
    candidatos['Rank após re-ranking'] = ranks(candidatos['Score Cross-Encoder'].to_numpy())
    return candidatos.sort_values('Rank após re-ranking')


def main():
    st.set_page_config(page_title='HealthSearch', page_icon='🔎', layout='wide')
    st.title('HealthSearch')
    st.caption('BM25 + embeddings multilíngues + Reciprocal Rank Fusion')
    st.info('Protótipo acadêmico de recuperação de informação. Corpus didático ampliado com IA, a partir dos seis trechos do enunciado. Não constitui orientação clínica.')
    k1 = st.sidebar.slider('k₁ · saturação', 0.0, 3.0, 1.2, .1)
    b = st.sidebar.slider('b · comprimento', 0.0, 1.0, .75, .05)
    alpha = st.sidebar.slider('α · peso do BM25', 0.0, 1.0, .5, .05)
    bonus = st.sidebar.checkbox('Cross-Encoder nos Top-3', False)
    st.sidebar.caption('RRF usa posições de 1 a 6 e k=60. Empates usam ID crescente. Primeiro uso baixa os modelos.')
    consulta = st.text_input('Consulta', 'ataque cardíaco')
    if not consulta.strip():
        st.warning('Digite uma consulta para comparar os motores.')
        return
    try:
        with st.spinner('Calculando rankings…'):
            df, tempos = pesquisar(consulta, k1, b, alpha)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f'Não foi possível carregar o modelo ou realizar a busca: {exc}')
        st.info('Confirme a instalação das dependências e o acesso ao Hugging Face no primeiro uso. Não há substituição silenciosa por vetores simulados.')
        return
    if not np.any(df['BM25']):
        st.warning('O BM25 não encontrou correspondência. Seu ranking está empatado; a ordem por ID ainda participa da fórmula RRF exigida.')
    for col,(nome,valor) in zip(st.columns(2),tempos.items()):
        col.metric(nome, f'{valor:.1f}')
    tabs = st.tabs(['Léxico', 'Semântico', 'Híbrido RRF', 'Matriz Comparativa'])
    for tab, score, rank in [(tabs[0],'BM25','Rank BM25'),(tabs[1],'Cosseno','Rank Semântico'),(tabs[2],'RRF','Rank RRF')]:
        with tab:
            ordem = df.sort_values(rank)
            st.subheader(f"Primeiro resultado: {ordem.iloc[0]['ID']}")
            st.dataframe(ordem[['ID','Título',score,rank,'Trecho recuperado']], hide_index=True)
    with tabs[2]:
        st.latex(r'RRF(D)=\frac{\alpha}{60+rank_{BM25}(D)}+\frac{1-\alpha}{60+rank_{sem}(D)}')
        st.dataframe(df[['ID','Parcela BM25','Parcela semântica','RRF']], hide_index=True)
        if bonus:
            try:
                st.dataframe(reranquear(df, consulta), hide_index=True)
                st.caption('O Cross-Encoder reordena só três candidatos. Seus logits não são probabilidades; o RRF original permanece na tabela.')
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f'Falha no modelo opcional de re-ranking: {exc}')
    with tabs[3]:
        st.dataframe(df.drop(columns='Texto'), hide_index=True)
        st.line_chart(df.set_index('ID')[['Rank BM25','Rank Semântico','Rank RRF']])
        st.caption('Rank 1 é o melhor. RRF não garante superar cada motor isolado; consultas sem evidência também recebem posições.')
        st.download_button('Baixar comparação CSV', df.to_csv(index=False).encode('utf-8-sig'), 'rankings.csv', 'text/csv')
    with st.expander('Corpus e pré-processamento'):
        st.dataframe(pd.DataFrame(CORPUS, columns=['ID','Título','Texto']), hide_index=True)
        st.write('Tokens da consulta:', tokenizar(consulta))
        st.json(PAGINAS_EXPANDIDAS)


if __name__ == '__main__':
    main()
