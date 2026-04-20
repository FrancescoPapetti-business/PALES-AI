"""
LLM-as-Judge Prompts per validazione zero-knowledge
"""

def get_claim_extraction_prompt(response: str) -> str:
    """Prompt per estrarre claim atomici dalla risposta"""
    return f"""Estrai tutti i claim atomici (affermazioni verificabili) da questa risposta.
Un claim atomico è una singola affermazione che può essere vera o falsa.

Regole:
- Separa affermazioni composte in claim singoli
- Ignora frasi di cortesia o meta-commenti
- Mantieni solo informazioni fattuali

Esempio:
Risposta: "Per associare un veterinario, devi caricare il documento di identità e inviarlo a info@classyfarm.it"
Claim:
1. È necessario caricare il documento di identità
2. Il documento deve essere inviato a info@classyfarm.it

Ora estrai i claim da questa risposta:

RISPOSTA:
{response}

Rispondi SOLO con una lista numerata dei claim, senza commenti aggiuntivi."""


def get_claim_verification_prompt(claim: str, context: str) -> str:
    """Prompt per verificare se un claim è supportato dai documenti"""
    return f"""Verifica se questo claim è supportato dai documenti forniti.

CLAIM DA VERIFICARE:
{claim}

DOCUMENTI:
{context}

Rispondi SOLO con una di queste tre opzioni:
- SUPPORTATO: il claim è esplicitamente presente o chiaramente derivabile dai documenti
- PARZIALMENTE_SUPPORTATO: il claim è parzialmente presente o richiede inferenza ragionevole
- NON_SUPPORTATO: il claim non è presente nei documenti o contraddice i documenti

Formato risposta: [SUPPORTATO/PARZIALMENTE_SUPPORTATO/NON_SUPPORTATO]"""


def get_self_contained_prompt(question: str, response: str) -> str:
    """Prompt per valutare autosufficienza della risposta"""
    return f"""Valuta questa risposta su una scala 1-5 per AUTOSUFFICIENZA.

DOMANDA:
{question}

RISPOSTA:
{response}

SCALA DI VALUTAZIONE:
5 = Risposta completa e actionable. L'utente può agire immediatamente senza consultare altre fonti.
    Contiene tutti i passaggi/informazioni necessari. Esempi concreti se applicabile.

4 = Risposta quasi completa. Manca qualche dettaglio minore non critico.
    L'utente può agire ma potrebbe dover fare assunzioni ragionevoli.

3 = Risposta parziale. Contiene informazioni utili ma richiede consultare altre fonti
    per completare il task. Link esterni presenti ma con contesto.

2 = Risposta vaga o principalmente composta da riferimenti esterni.
    Poche informazioni concrete. Dice "consulta X" senza spiegare.

1 = Risposta inutile. Solo rinvii a link esterni, "non so", o fuori tema.

Rispondi nel formato:
SCORE: [1-5]
MOTIVAZIONE: [Breve spiegazione della valutazione in massimo 2 frasi]"""


def get_chunk_relevance_prompt(question: str, chunk: str) -> str:
    """Prompt per valutare pertinenza di un chunk rispetto alla domanda"""
    return f"""Questo documento è utile per rispondere alla domanda?

DOMANDA:
{question}

DOCUMENTO:
{chunk}

Rispondi SOLO con una di queste tre opzioni:
- UTILE: il documento contiene informazioni direttamente rilevanti per rispondere
- PARZIALMENTE_UTILE: il documento contiene informazioni tangenzialmente rilevanti o di contesto
- INUTILE: il documento non è pertinente alla domanda

Formato risposta: [UTILE/PARZIALMENTE_UTILE/INUTILE]"""


def get_answer_relevance_prompt(question: str, response: str) -> str:
    """Prompt per valutare pertinenza della risposta (alternativa a embedding)"""
    return f"""La risposta affronta direttamente la domanda posta?

DOMANDA:
{question}

RISPOSTA:
{response}

Rispondi SOLO con una di queste tre opzioni:
- RILEVANTE: la risposta affronta direttamente la domanda
- PARZIALMENTE_RILEVANTE: la risposta è tangenziale ma contiene informazioni utili
- NON_RILEVANTE: la risposta è fuori tema o risponde a una domanda diversa

Formato risposta: [RILEVANTE/PARZIALMENTE_RILEVANTE/NON_RILEVANTE]"""


def get_citation_extraction_prompt(response: str) -> str:
    """Prompt per estrarre citazioni di fonti dalla risposta"""
    return f"""Estrai tutte le citazioni di fonti/riferimenti da questa risposta.

Cerca pattern come:
- "pagina X"
- "sezione Y"
- "documento Z"
- "guida W"
- "(fonte: ...)"
- "vedi ..."

RISPOSTA:
{response}

Rispondi con lista delle citazioni trovate (una per riga).
Se non ci sono citazioni, rispondi: NESSUNA_CITAZIONE"""