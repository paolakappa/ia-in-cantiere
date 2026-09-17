import io
import json
import re
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ------------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="IA in cantiere",
    page_icon="🦺",
    layout="wide"
)

# ------------------------------------------------------------
# STILE
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.65rem 1.7rem;
    border: 1px solid rgba(120,120,120,.18);
    border-radius: 22px;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(255,193,7,.12), rgba(25,118,210,.07));
    box-shadow: 0 8px 28px rgba(0,0,0,.05);
}
.hero h1 { margin-bottom: .25rem; font-size: 2.35rem; }
.hero p { margin-bottom: .65rem; font-size: 1.08rem; }
.badge {
    display: inline-block;
    padding: .28rem .58rem;
    margin-right: .35rem;
    margin-top: .2rem;
    border-radius: 999px;
    border: 1px solid rgba(120,120,120,.22);
    font-size: .82rem;
    font-weight: 600;
    background: rgba(255,255,255,.45);
}
.topic-box, .feature-box, .flow-box {
    border: 1px solid rgba(120,120,120,.20);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: .55rem;
    background: rgba(255,255,255,.025);
}
.topic-box { min-height: 128px; }
.feature-box { min-height: 128px; }
.flow-box { text-align: center; min-height: 92px; }
.small-muted { opacity: .75; font-size: .92rem; }
.source-box {
    border-left: 4px solid #999;
    padding-left: 0.8rem;
    margin-bottom: 0.8rem;
}
.coverage-box {
    border: 1px solid rgba(120,120,120,.20);
    border-radius: 14px;
    padding: .75rem .9rem;
    margin: .7rem 0;
}
.footer {
    margin-top: 2.2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(120,120,120,.18);
    opacity: .72;
    text-align: center;
    font-size: .88rem;
}

/* ---------------------------------------------------------
   RESPONSIVE / MOBILE
   Streamlit è già responsive; queste regole migliorano
   leggibilità, card, tab, pulsanti e colonne su smartphone.
   --------------------------------------------------------- */
html, body, [class*="css"] {
    overflow-wrap: anywhere;
}

[data-testid="stChatMessage"] {
    max-width: 100%;
}

[data-testid="stChatMessageContent"],
[data-testid="stMarkdownContainer"] {
    overflow-wrap: anywhere;
    word-break: normal;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
    overflow-x: auto;
    scrollbar-width: thin;
}

.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
}

/* Smartphone e piccoli tablet */
@media (max-width: 768px) {
    .block-container {
        max-width: 100%;
        padding-top: .65rem;
        padding-left: .85rem;
        padding-right: .85rem;
        padding-bottom: 5rem;
    }

    .hero {
        padding: 1.05rem 1rem;
        border-radius: 16px;
        margin-bottom: .7rem;
    }

    .hero h1 {
        font-size: 1.78rem;
        line-height: 1.15;
    }

    .hero p {
        font-size: .98rem;
        line-height: 1.4;
    }

    .badge {
        font-size: .74rem;
        padding: .22rem .45rem;
        margin-right: .18rem;
        margin-top: .28rem;
    }

    .topic-box, .feature-box, .flow-box {
        min-height: auto;
        padding: .85rem;
        border-radius: 14px;
    }

    .topic-box h3 {
        font-size: 1.12rem;
        margin-bottom: .35rem;
    }

    .small-muted {
        font-size: .86rem;
    }

    /* Impila le colonne Streamlit su schermi stretti. */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: .55rem !important;
    }

    [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }

    /* I pulsanti diventano comodi da toccare con il pollice. */
    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        min-height: 2.75rem;
        white-space: normal;
        line-height: 1.25;
    }

    /* Tab compatti e scorrevoli orizzontalmente. */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-start;
        overflow-x: auto;
        padding-bottom: .15rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: .9rem;
        padding-left: .75rem;
        padding-right: .75rem;
    }

    /* Evita che testi lunghi nelle fonti escano dallo schermo. */
    .source-box {
        padding-left: .65rem;
        overflow-wrap: anywhere;
    }

    .coverage-box {
        padding: .65rem .75rem;
    }

    .footer {
        margin-top: 1.4rem;
        font-size: .78rem;
        line-height: 1.35;
    }
}

/* Telefoni molto stretti */
@media (max-width: 420px) {
    .block-container {
        padding-left: .65rem;
        padding-right: .65rem;
    }

    .hero h1 {
        font-size: 1.55rem;
    }

    .badge {
        display: inline-flex;
        margin-bottom: .1rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🦺 IA in cantiere</h1>
<p><b>Assistente educativo multilingue per la sicurezza nei cantieri</b></p>
<span class="badge">🎓 Prototipo di tesi</span>
<span class="badge">📚 Fonti verificabili</span>
<span class="badge">🌍 8 lingue</span>
<span class="badge">🔎 Risposte tracciabili</span>
</div>
""", unsafe_allow_html=True)

st.info(
    "Prototipo a scopo formativo. Le risposte sono costruite solo sui documenti caricati. "
    "Non sostituisce formazione, addestramento, procedure aziendali o indicazioni dei soggetti responsabili della sicurezza."
)

st.caption("📱 Su smartphone le impostazioni (lingua, stile e PDF aggiuntivi) sono disponibili dal menu laterale ☰.")

# ------------------------------------------------------------
# DATI DI BASE
# ------------------------------------------------------------
LANGUAGES = {
    "Italiano": "Italian",
    "English": "English",
    "Română": "Romanian",
    "Shqip": "Albanian",
    "العربية": "Arabic",
    "Français": "French",
    "اردو": "Urdu",
    "বাংলা": "Bengali",
}

STYLES = {
    "Tecnico": """
Usa un registro tecnico-professionale, ma chiaro.
Mantieni la terminologia normativa e tecnica presente nelle fonti.
Non semplificare termini tecnici se questo ne altera il significato.
""",
    "Semplice": """
Usa frasi brevi e parole comuni.
Spiega i termini tecnici quando compaiono.
Mantieni invariati numeri, misure e riferimenti normativi.
Evita formulazioni burocratiche quando puoi esprimere lo stesso concetto in modo più chiaro.
""",
    "Molto semplice": """
Usa frasi molto brevi, una sola idea per frase.
Preferisci parole comuni.
Se devi usare un termine tecnico, spiegalo subito.
Non eliminare numeri, misure, obblighi o riferimenti normativi presenti nelle fonti.
Non banalizzare il contenuto tecnico.
""",
}

TOPICS = {
    "Cadute dall'alto": {
        "emoji": "⚠️",
        "description": "Lavori in quota, rischio di caduta e protezioni collettive.",
        "query": "cadute dall'alto lavori in quota rischio protezioni collettive parapetti prevenzione",
        "prompts": [
            "Quando un lavoro è considerato lavoro in quota?",
            "Quali protezioni collettive sono previste contro le cadute dall'alto?",
            "Quali sono i principali rischi nelle cadute dall'alto?"
        ]
    },
    "Scale portatili": {
        "emoji": "🪜",
        "description": "Scelta, uso, stabilità, appoggio e condizioni di impiego.",
        "query": "scale portatili uso lavoro in quota stabilità appoggio presa sicurezza",
        "prompts": [
            "Quali condizioni devono essere verificate prima di usare una scala portatile?",
            "Come deve essere posizionata una scala portatile?",
            "Quando una scala portatile non è adatta al lavoro da svolgere?"
        ]
    },
    "Parapetti": {
        "emoji": "🚧",
        "description": "Parapetti provvisori, protezione dei bordi, scelta e utilizzo.",
        "query": "parapetti provvisori protezione collettiva bordi caduta dall'alto montaggio uso",
        "prompts": [
            "A cosa serve un parapetto provvisorio?",
            "Quali elementi deve avere un parapetto?",
            "Quando è necessario proteggere un bordo contro la caduta?"
        ]
    },
    "Ponteggi e trabattelli": {
        "emoji": "🏗️",
        "description": "Ponteggi, PiMUS, trabattelli, stabilità, montaggio e uso.",
        "query": "ponteggi fissi trabattelli PiMUS montaggio uso smontaggio stabilità protezione",
        "prompts": [
            "Che cos'è il PiMUS?",
            "Quali controlli sono importanti prima di utilizzare un trabattello?",
            "Quali aspetti incidono sulla stabilità di un trabattello?"
        ]
    },
}

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("Impostazioni")

    output_language_label = st.selectbox(
        "Lingua della risposta",
        list(LANGUAGES.keys()),
        index=0
    )
    output_language = LANGUAGES[output_language_label]

    response_style_label = st.selectbox(
        "Stile della risposta",
        list(STYLES.keys()),
        index=1,
        help="Scegli quanto vuoi rendere il contenuto tecnico più accessibile."
    )
    response_style = STYLES[response_style_label]

    st.divider()
    st.markdown("**🎓 Progetto accademico**")
    st.caption(
        "Versione v0.8 · interfaccia responsive · prototipo sperimentale sviluppato per una tesi sulla formazione "
        "alla salute e sicurezza nei cantieri."
    )

    st.divider()
    st.subheader("Fonti")

    st.caption(
        "I documenti ufficiali inclusi nel progetto vengono caricati automaticamente. "
        "Qui puoi aggiungere altri PDF senza modificare il sito."
    )

    uploaded_files = st.file_uploader(
        "Aggiungi PDF facoltativi",
        type=["pdf"],
        accept_multiple_files=True,
        help="Esempi: D.Lgs. 81/08 aggiornato, ulteriori documenti INAIL, linee guida o manuali ufficiali."
    )

    st.divider()

    if st.button("🏠 Torna agli argomenti", use_container_width=True):
        st.session_state.selected_topic = None
        st.session_state.messages = []
        st.session_state.micro_lesson = None
        st.session_state.micro_sources = None
        st.session_state.quiz = None
        st.session_state.quiz_sources = None
        st.rerun()

# ------------------------------------------------------------
# FUNZIONI TESTO / PDF
# ------------------------------------------------------------
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_text_with_offsets(text, chunk_size=1300, overlap=220):
    """
    Divide il testo in blocchi mantenendo la posizione del blocco
    all'interno della pagina. Serve per associare il blocco all'articolo
    normativo più vicino.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        raw_chunk = text[start:end]
        chunk = raw_chunk.strip()

        if chunk:
            # Calcola dove inizia davvero il testo dopo eventuali spazi iniziali.
            left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            real_start = start + left_trim
            chunks.append((chunk, real_start, end))

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks

def pretty_source_name(filename):
    name = filename.lower()

    if "parapett" in name:
        return "INAIL – Parapetti provvisori"
    if "trabatt" in name or "volume" in name:
        return "INAIL – Trabattelli"
    if "ponteggi" in name or "facciata" in name:
        return "INAIL – I ponteggi di facciata"
    if "81" in name and ("dlgs" in name or "d.lgs" in name or "testo" in name):
        return "D.Lgs. 9 aprile 2008, n. 81"
    if "manuale" in name and "cantier" in name:
        return "Manuale sicurezza nei cantieri"

    return filename

def extract_article_refs(text):
    """
    Cerca riferimenti del tipo:
    Art. 111
    art. 113
    Articolo 107
    articolo 37-bis
    """
    if not text:
        return []

    patterns = [
        r"\bart\.?\s*(\d+(?:[-–]\w+)?)\b",
        r"\barticolo\s+(\d+(?:[-–]\w+)?)\b",
    ]

    found = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = str(match).replace("–", "-")
            if normalized not in found:
                found.append(normalized)

    return found

def find_article_headings(text):
    """
    Restituisce gli articoli presenti nella pagina con la loro posizione.
    Cerca forme come:
    Art. 111
    ARTICOLO 113
    Articolo 107
    """
    if not text:
        return []

    pattern = re.compile(
        r"\b(?:art\.?|articolo)\s*(\d+(?:[-–][A-Za-z0-9]+)?)\b",
        flags=re.IGNORECASE
    )

    headings = []
    for match in pattern.finditer(text):
        article = match.group(1).replace("–", "-")
        headings.append({
            "article": article,
            "position": match.start()
        })

    return headings


def closest_article_for_chunk(page_text, chunk_text, chunk_start, headings):
    """
    Associa il chunk all'articolo più plausibile:
    1. se nel chunk compare esplicitamente un articolo, usa quello;
    2. altrimenti usa l'ultimo articolo che compare PRIMA dell'inizio del chunk.
    In questo modo evita di mostrare tutti gli articoli presenti nella stessa pagina.
    """
    explicit = extract_article_refs(chunk_text)
    if explicit:
        return explicit[:2]

    previous = [
        h for h in headings
        if h["position"] <= chunk_start
    ]

    if previous:
        return [previous[-1]["article"]]

    return []

@st.cache_data(show_spinner=False)
def extract_chunks(file_payloads):
    records = []

    for filename, file_bytes in file_payloads:
        reader = PdfReader(io.BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text())

            if not text:
                continue

            article_headings = find_article_headings(text)

            for chunk_number, (chunk, chunk_start, chunk_end) in enumerate(
                split_text_with_offsets(text),
                start=1
            ):
                article_refs = closest_article_for_chunk(
                    text,
                    chunk,
                    chunk_start,
                    article_headings
                )

                records.append({
                    "source": filename,
                    "source_pretty": pretty_source_name(filename),
                    "page": page_number,
                    "chunk": chunk_number,
                    "articles": article_refs,
                    "text": chunk,
                })

    return records

# ------------------------------------------------------------
# GEMINI
# ------------------------------------------------------------
def get_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error(
            "Manca la chiave Gemini. In Streamlit vai su App settings → Secrets "
            'e inserisci: GEMINI_API_KEY = "LA_TUA_CHIAVE"'
        )
        st.stop()

    return genai.Client(api_key=api_key)

def generate_text(client, prompt):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()

def translate_query_to_italian(client, question):
    prompt = f"""
Traduci la domanda seguente in italiano per permettere una ricerca documentale.

REGOLE:
- Mantieni invariati numeri, misure, sigle, articoli di legge e termini tecnici.
- Non aggiungere spiegazioni.
- Restituisci SOLTANTO la traduzione italiana.

DOMANDA:
{question}
"""
    return generate_text(client, prompt)

# ------------------------------------------------------------
# RICERCA DOCUMENTALE
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_search_index(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=60000
    )
    matrix = vectorizer.fit_transform(list(texts))
    return vectorizer, matrix

def retrieve(query, records, top_k=8):
    texts = tuple(r["text"] for r in records)
    vectorizer, matrix = build_search_index(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked:
        item = dict(records[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results

def build_context(results):
    parts = []

    for i, item in enumerate(results, start=1):
        article_text = ""
        if item.get("articles"):
            if len(item["articles"]) == 1:
                article_text = ", articolo " + item["articles"][0]
            else:
                article_text = ", articoli " + ", ".join(item["articles"][:2])

        parts.append(
            f"[FONTE {i}: {item['source_pretty']}{article_text}, file {item['source']}, pagina {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(parts)

def unique_source_results(results, max_items=5):
    unique = []
    seen = set()

    for item in results:
        key = (item["source"], item["page"])

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

        if len(unique) >= max_items:
            break

    return unique

def show_sources(results, title="📚 Fonti utilizzate"):
    source_results = unique_source_results(results, max_items=5)

    if not source_results:
        return

    with st.expander(title):
        for item in source_results:
            article_html = ""
            if item.get("articles"):
                if len(item["articles"]) == 1:
                    article_html = f"<b>Riferimento:</b> art. {item['articles'][0]}<br>"
                else:
                    labels = ", ".join(f"art. {a}" for a in item["articles"][:2])
                    article_html = f"<b>Riferimenti:</b> {labels}<br>"

            st.markdown(
                f"<div class='source-box'><b>{item['source_pretty']}</b><br>"
                f"{article_html}"
                f"Pagina {item['page']}<br>"
                f"<span class='small-muted'>File: {item['source']}</span></div>",
                unsafe_allow_html=True
            )

            excerpt = item["text"]
            if len(excerpt) > 1200:
                excerpt = excerpt[:1200] + "..."
            st.markdown("**Passaggio originale recuperato**")
            st.caption(excerpt)
            st.divider()

def coverage_label(best_score):
    """
    Indicatore euristico di copertura documentale.
    NON è una misura di correttezza o affidabilità della risposta.
    """
    if best_score >= 0.16:
        return "Alta", "Molti termini della domanda trovano corrispondenza nei passaggi recuperati."
    if best_score >= 0.07:
        return "Media", "La documentazione contiene passaggi pertinenti, ma la corrispondenza è parziale."
    if best_score >= 0.015:
        return "Limitata", "La risposta è possibile, ma i passaggi recuperati sono meno vicini alla formulazione della domanda."
    return "Insufficiente", "Non è stato trovato un passaggio abbastanza pertinente per rispondere in modo affidabile."

def show_transparency(results, best_score):
    label, explanation = coverage_label(best_score)
    docs = len({r["source"] for r in results if r.get("score", 0) > 0})
    with st.expander("🔎 Perché questa risposta? · Trasparenza del recupero"):
        st.markdown(
            f"**Copertura documentale: {label}**  \n"
            f"{explanation}  \n"
            f"Passaggi esaminati per la risposta: **{len(results)}** · Documenti coinvolti: **{docs}**."
        )
        st.caption(
            "L'indicatore descrive solo quanto la domanda assomiglia testualmente ai passaggi recuperati. "
            "Non è una percentuale di affidabilità e non certifica la correttezza della risposta."
        )

def build_download_text(question, answer, results):
    lines = [
        "IA IN CANTIERE — RISPOSTA ESPORTATA",
        "",
        f"Domanda: {question}",
        "",
        "Risposta:",
        answer,
        "",
        "Fonti recuperate:"
    ]
    for item in unique_source_results(results, max_items=5):
        article = ""
        if item.get("articles"):
            article = " · art. " + ", ".join(item["articles"][:2])
        lines.append(f"- {item['source_pretty']}{article} · pagina {item['page']}")
    lines += [
        "",
        "Nota: contenuto a scopo formativo; verificare sempre la documentazione ufficiale e le procedure applicabili."
    ]
    return "\n".join(lines)

# ------------------------------------------------------------
# GENERAZIONE RISPOSTA CHAT
# ------------------------------------------------------------
def answer_from_context(
    client,
    question,
    selected_topic,
    results,
    output_language,
    response_style
):
    context = build_context(results)

    prompt = f"""
Sei "IA in cantiere", un assistente EDUCATIVO sulla salute e sicurezza nei cantieri.

ARGOMENTO SELEZIONATO:
{selected_topic}

REGOLE OBBLIGATORIE:
1. Rispondi esclusivamente usando le informazioni contenute nel CONTESTO DOCUMENTALE.
2. Non usare conoscenze esterne.
3. Non inventare obblighi, divieti, misure, distanze, sanzioni, procedure,
   articoli di legge o requisiti tecnici.
4. Se il contesto non è sufficiente, dillo chiaramente.
5. Mantieni invariati numeri, unità di misura e riferimenti normativi presenti nel contesto.
6. Non presentare la risposta come sostitutiva della formazione o delle procedure aziendali.
7. Rispondi in {output_language}.
8. Non inventare citazioni.
9. Non aggiungere informazioni solo perché sembrano ragionevoli.
10. Se due fonti sembrano in contrasto, segnala che è necessaria una verifica.
11. Se nel contesto è presente un riferimento normativo esplicito (es. art. 111, art. 113),
    cita SOLO quegli articoli effettivamente presenti nel contesto.
12. Quando pertinente, chiudi la risposta con una riga breve:
    "Riferimento normativo: D.Lgs. 81/2008, art. X."
    Non inventare mai un articolo non presente nel contesto.

STILE RICHIESTO:
{response_style}

DOMANDA DELL'UTENTE:
{question}

CONTESTO DOCUMENTALE:
{context}
"""

    return generate_text(client, prompt)

# ------------------------------------------------------------
# MICRO-LEZIONE
# ------------------------------------------------------------
def create_micro_lesson(
    client,
    selected_topic,
    results,
    output_language,
    response_style
):
    context = build_context(results)

    prompt = f"""
Sei "IA in cantiere", un assistente EDUCATIVO sulla salute e sicurezza nei cantieri.

Crea una MICRO-LEZIONE sull'argomento:
{selected_topic}

REGOLE:
- Usa esclusivamente il CONTESTO DOCUMENTALE.
- Non aggiungere conoscenze esterne.
- Non inventare requisiti tecnici.
- Durata di lettura: circa 45-60 secondi.
- Struttura:
  1. Titolo breve
  2. "Cosa devi sapere"
  3. 3-5 punti essenziali
  4. "Ricorda"
- Mantieni invariati numeri, misure e riferimenti normativi.
- Rispondi in {output_language}.
- Non scrivere fonti inventate: l'app le mostrerà separatamente.
- Se nel contesto è presente un articolo del D.Lgs. 81/2008 pertinente alla lezione,
  puoi indicarlo in fondo come "Riferimento normativo", ma SOLO se compare davvero nel contesto.
- Ricorda che lo strumento è formativo e non sostitutivo della formazione obbligatoria.

STILE:
{response_style}

CONTESTO DOCUMENTALE:
{context}
"""

    return generate_text(client, prompt)

# ------------------------------------------------------------
# QUIZ
# ------------------------------------------------------------
def strip_json_fences(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()

def create_quiz(
    client,
    selected_topic,
    results,
    output_language,
    response_style
):
    context = build_context(results)

    prompt = f"""
Sei "IA in cantiere", un assistente EDUCATIVO sulla salute e sicurezza nei cantieri.

Crea UN SOLO quiz basato sull'argomento:
{selected_topic}

Deve essere uno scenario realistico ma semplice, ricavato esclusivamente dal CONTESTO DOCUMENTALE.

REGOLE:
- Non usare conoscenze esterne.
- Non inventare obblighi, numeri, misure o requisiti.
- Crea esattamente 3 alternative.
- Una sola alternativa deve essere corretta.
- L'indice della risposta corretta deve essere 0, 1 oppure 2.
- L'explanation deve spiegare perché la risposta corretta è corretta usando solo il contesto.
- Se nel contesto è presente un articolo normativo pertinente, l'explanation può citarlo,
  ma SOLO se compare davvero nel contesto.
- Rispondi in {output_language}.
- Linguaggio coerente con questo stile:
{response_style}

RESTITUISCI SOLO JSON VALIDO, SENZA MARKDOWN, con questa struttura esatta:
{{
  "question": "testo della domanda",
  "options": ["risposta A", "risposta B", "risposta C"],
  "correct_index": 0,
  "explanation": "spiegazione"
}}

CONTESTO DOCUMENTALE:
{context}
"""

    raw = generate_text(client, prompt)
    clean = strip_json_fences(raw)
    data = json.loads(clean)

    if not isinstance(data.get("options"), list) or len(data["options"]) != 3:
        raise ValueError("Il quiz generato non contiene esattamente 3 opzioni.")

    if data.get("correct_index") not in [0, 1, 2]:
        raise ValueError("Indice risposta corretta non valido.")

    return data

# ------------------------------------------------------------
# PREPARAZIONE FONTI
# ------------------------------------------------------------
APP_DIR = Path(__file__).parent

# Tutti i PDF presenti nella cartella del progetto vengono considerati fonti permanenti.
bundled_pdf_paths = sorted(APP_DIR.glob("*.pdf"))

bundled_payloads = []
for pdf_path in bundled_pdf_paths:
    try:
        bundled_payloads.append((pdf_path.name, pdf_path.read_bytes()))
    except Exception:
        pass

uploaded_payloads = []
if uploaded_files:
    uploaded_payloads = [(f.name, f.getvalue()) for f in uploaded_files]

# Evita duplicati se un PDF permanente viene anche caricato manualmente.
combined = {}
for filename, payload in bundled_payloads + uploaded_payloads:
    combined[filename] = payload

file_payloads = tuple(combined.items())

if not file_payloads:
    st.warning(
        "Non risultano fonti disponibili. Inserisci almeno un PDF nel repository "
        "oppure caricalo dalla barra laterale."
    )
    st.stop()

with st.spinner("Leggo e preparo le fonti..."):
    records = extract_chunks(file_payloads)

if not records:
    st.error(
        "Non sono riuscito a estrarre testo dai PDF. "
        "Uno o più documenti potrebbero essere scansioni composte solo da immagini."
    )
    st.stop()

st.success(
    f"Fonti pronte: {len(file_payloads)} PDF "
    f"({len(bundled_payloads)} permanenti, {len(uploaded_payloads)} aggiunti), "
    f"{len(records)} sezioni indicizzate."
)

with st.expander("📚 Documenti disponibili"):
    if bundled_pdf_paths:
        st.markdown("**Fonti permanenti del prototipo**")
        for pdf_path in bundled_pdf_paths:
            st.write(f"• {pretty_source_name(pdf_path.name)}")
    if uploaded_files:
        st.markdown("**Fonti aggiunte in questa sessione**")
        for f in uploaded_files:
            st.write(f"• {pretty_source_name(f.name)}")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "micro_lesson" not in st.session_state:
    st.session_state.micro_lesson = None

if "micro_sources" not in st.session_state:
    st.session_state.micro_sources = None

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_sources" not in st.session_state:
    st.session_state.quiz_sources = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ------------------------------------------------------------
# HOME / SELEZIONE ARGOMENTO
# ------------------------------------------------------------
if st.session_state.selected_topic is None:
    st.subheader("Scegli un argomento")

    st.write(
        "Seleziona l'area sulla quale vuoi fare una domanda, seguire una micro-lezione "
        "oppure metterti alla prova con uno scenario."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documenti", len(file_payloads))
    m2.metric("Sezioni indicizzate", len(records))
    m3.metric("Lingue", len(LANGUAGES))
    m4.metric("Modalità", 3)

    st.markdown("### Cosa rende diverso il prototipo")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("<div class='feature-box'><b>📚 Risposte ancorate alle fonti</b><br><span class='small-muted'>Il modello riceve prima i passaggi recuperati dai PDF selezionati.</span></div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div class='feature-box'><b>🌍 Adattamento linguistico</b><br><span class='small-muted'>Una stessa base documentale può essere spiegata con registri e lingue differenti.</span></div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div class='feature-box'><b>🔎 Tracciabilità</b><br><span class='small-muted'>Documento, pagina e passaggio originale restano sempre consultabili.</span></div>", unsafe_allow_html=True)

    with st.expander("⚙️ Come funziona in 5 passaggi", expanded=False):
        cols = st.columns(5)
        steps = [
            ("1", "Domanda", "L'utente scrive in linguaggio naturale"),
            ("2", "Ricerca", "TF-IDF individua i passaggi pertinenti"),
            ("3", "Contesto", "I passaggi vengono forniti al modello"),
            ("4", "Risposta", "Gemini riformula secondo lingua e livello"),
            ("5", "Verifica", "L'utente può aprire la fonte originale"),
        ]
        for col, (num, title, desc) in zip(cols, steps):
            with col:
                st.markdown(f"<div class='flow-box'><b>{num}. {title}</b><br><span class='small-muted'>{desc}</span></div>", unsafe_allow_html=True)

    names = list(TOPICS.keys())
    row1 = st.columns(2)
    row2 = st.columns(2)

    for col, topic_name in zip(row1 + row2, names):
        topic = TOPICS[topic_name]

        with col:
            st.markdown(
                f"""
                <div class="topic-box">
                    <h3>{topic['emoji']} {topic_name}</h3>
                    <p>{topic['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                f"Apri {topic_name}",
                key=f"topic_{topic_name}",
                use_container_width=True
            ):
                st.session_state.selected_topic = topic_name
                st.session_state.messages = []
                st.session_state.micro_lesson = None
                st.session_state.micro_sources = None
                st.session_state.quiz = None
                st.session_state.quiz_sources = None
                st.rerun()

    st.stop()

# ------------------------------------------------------------
# AREA ARGOMENTO
# ------------------------------------------------------------
selected_topic = st.session_state.selected_topic
topic_info = TOPICS[selected_topic]

st.markdown(
    f"## {topic_info['emoji']} {selected_topic}"
)

st.caption(topic_info["description"])

with st.expander("✨ Demo guidata · prova una domanda", expanded=False):
    st.caption("Domande pronte per mostrare rapidamente il funzionamento del prototipo.")
    quick_cols = st.columns(3)
    for i, (col, prompt_text) in enumerate(zip(quick_cols, topic_info.get("prompts", []))):
        with col:
            if st.button(prompt_text, key=f"quick_{selected_topic}_{i}", use_container_width=True):
                st.session_state.pending_question = prompt_text
                st.rerun()

tab_chat, tab_lesson, tab_quiz = st.tabs(
    ["💬 Chat", "🎓 Micro-lezione", "🎯 Quiz"]
)

client = get_client()

# ------------------------------------------------------------
# TAB CHAT
# ------------------------------------------------------------
with tab_chat:
    st.write(
        "Fai una domanda. Il sistema cercherà prima nei documenti caricati e "
        "costruirà la risposta solo sui passaggi recuperati."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                show_transparency(msg["sources"], msg.get("best_score", 0.0))
                show_sources(msg["sources"])

    typed_question = st.chat_input(
        f"Scrivi una domanda su: {selected_topic}..."
    )
    question = typed_question
    if st.session_state.pending_question and not question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Cerco nelle fonti e preparo la risposta..."):
                question_it = translate_query_to_italian(client, question)

                search_query = (
                    f"{topic_info['query']} {question_it}"
                )

                results = retrieve(
                    search_query,
                    records,
                    top_k=8
                )

                best_score = results[0]["score"] if results else 0.0

                if best_score < 0.015:
                    answer = (
                        "Non ho trovato nei documenti caricati un passaggio sufficientemente "
                        "pertinente per rispondere in modo affidabile."
                    )
                else:
                    answer = answer_from_context(
                        client,
                        question,
                        selected_topic,
                        results,
                        output_language,
                        response_style
                    )

            st.markdown(answer)
            show_transparency(results, best_score)
            show_sources(results)

            export_text = build_download_text(question, answer, results)
            st.download_button(
                "⬇️ Esporta risposta e fonti",
                data=export_text,
                file_name="ia_in_cantiere_risposta.txt",
                mime="text/plain",
                use_container_width=False
            )

            st.caption(
                "⚠️ Strumento a scopo formativo. Verificare sempre procedure aziendali, "
                "documentazione ufficiale e indicazioni dei soggetti della prevenzione."
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": results,
            "best_score": best_score
        })

# ------------------------------------------------------------
# TAB MICRO-LEZIONE
# ------------------------------------------------------------
with tab_lesson:
    st.write(
        "Genera una spiegazione breve dell'argomento, pensata per essere letta in circa un minuto."
    )

    if st.button(
        "🎓 Genera micro-lezione",
        use_container_width=True
    ):
        with st.spinner("Preparo la micro-lezione dalle fonti..."):
            results = retrieve(
                topic_info["query"],
                records,
                top_k=8
            )

            try:
                lesson = create_micro_lesson(
                    client,
                    selected_topic,
                    results,
                    output_language,
                    response_style
                )

                st.session_state.micro_lesson = lesson
                st.session_state.micro_sources = results

            except Exception as e:
                st.error(
                    "Non sono riuscito a generare la micro-lezione. "
                    "Riprova tra qualche secondo."
                )

    if st.session_state.micro_lesson:
        st.markdown(st.session_state.micro_lesson)

        if st.session_state.micro_sources:
            micro_best = st.session_state.micro_sources[0]["score"] if st.session_state.micro_sources else 0.0
            show_transparency(st.session_state.micro_sources, micro_best)
            show_sources(
                st.session_state.micro_sources,
                title="📚 Fonti della micro-lezione"
            )

        st.caption(
            "⚠️ Contenuto educativo di rinforzo: non sostituisce formazione e addestramento previsti."
        )

# ------------------------------------------------------------
# TAB QUIZ
# ------------------------------------------------------------
with tab_quiz:
    st.write(
        "Mettiti alla prova con uno scenario costruito a partire dai documenti caricati."
    )

    if st.button(
        "🎯 Genera nuovo quiz",
        use_container_width=True
    ):
        with st.spinner("Creo uno scenario dalle fonti..."):
            results = retrieve(
                topic_info["query"],
                records,
                top_k=8
            )

            try:
                quiz = create_quiz(
                    client,
                    selected_topic,
                    results,
                    output_language,
                    response_style
                )

                st.session_state.quiz = quiz
                st.session_state.quiz_sources = results

            except Exception:
                st.session_state.quiz = None
                st.session_state.quiz_sources = None
                st.error(
                    "Il quiz non è stato generato correttamente. Premi di nuovo "
                    "'Genera nuovo quiz'."
                )

    if st.session_state.quiz:
        quiz = st.session_state.quiz

        st.markdown(f"### Scenario")
        st.write(quiz["question"])

        option_labels = [
            f"A. {quiz['options'][0]}",
            f"B. {quiz['options'][1]}",
            f"C. {quiz['options'][2]}",
        ]

        selected = st.radio(
            "Qual è la scelta più appropriata?",
            option_labels,
            index=None,
            key="quiz_answer"
        )

        if st.button(
            "Verifica risposta",
            use_container_width=True,
            disabled=(selected is None)
        ):
            selected_index = option_labels.index(selected)

            if selected_index == quiz["correct_index"]:
                st.success("✅ Risposta corretta")
            else:
                correct_letter = ["A", "B", "C"][quiz["correct_index"]]
                st.error(f"❌ Risposta non corretta. La risposta prevista è {correct_letter}.")

            st.markdown("**Perché?**")
            st.write(quiz["explanation"])

        if st.session_state.quiz_sources:
            quiz_best = st.session_state.quiz_sources[0]["score"] if st.session_state.quiz_sources else 0.0
            show_transparency(st.session_state.quiz_sources, quiz_best)
            show_sources(
                st.session_state.quiz_sources,
                title="📚 Fonti del quiz"
            )

        st.caption(
            "⚠️ Il quiz ha finalità formative e non sostituisce la valutazione dei rischi "
            "o le procedure previste per l'attività reale."
        )


st.markdown(
    "<div class='footer'>IA in cantiere · v0.8 responsive · prototipo sperimentale sviluppato nell'ambito di una tesi di laurea · 2026</div>",
    unsafe_allow_html=True
)
