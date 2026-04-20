# FAQ Input

Metti qui file TXT o MD con FAQ aggiuntive. Verranno processate con priorita' massima nel retrieval.

## Formati supportati

### Formato 1: Q: / A:

```
Q: Come accedo a ClassyFarm?
A: Utilizza SPID o CIE per accedere al sistema.

Q: Quale email usare per le deleghe?
A: sanita.animale@sanita.it in copia a info@classyfarm.it.
```

### Formato 2: DOMANDA: / RISPOSTA:

```
DOMANDA: Come accedo a ClassyFarm?
RISPOSTA: Utilizza SPID o CIE per accedere al sistema.

DOMANDA: Quale email usare per le deleghe?
RISPOSTA: sanita.animale@sanita.it in copia a info@classyfarm.it.
```

### Formato 3: Markdown (### Domanda + paragrafo)

```
### Come accedo a ClassyFarm?
Utilizza SPID o CIE per accedere al sistema.

### Quale email usare per le deleghe?
sanita.animale@sanita.it in copia a info@classyfarm.it.
```

## Come usare

1. Crea un file `.txt` o `.md` in questa cartella
2. Scrivi le Q/A in uno dei formati sopra
3. Esegui: `python -m scripts.rebuild_knowledge_base`
4. Le FAQ verranno indicizzate con priorita' massima

Organizza i file per argomento: `faq_operatori.md`, `faq_veterinari.md`, `faq_checklist.md`, ecc.
Il nome del file viene usato come sezione FAQ.

Ogni FAQ ha un boost del +100% nel ranking: se la tua domanda matcha una FAQ,
quella risposta vince sempre sui chunk dei PDF.
