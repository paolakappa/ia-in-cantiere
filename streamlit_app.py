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
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(120,120,120,.20);
    border-radius: 18px;
    margin-bottom: 1rem;
}
.hero h1 {
    margin-bottom: .2rem;
}
.hero p {
    margin-bottom: 0;
    font-size: 1.05rem;
}
.topic-box {
    border: 1px solid rgba(120,120,120,.22);
    border-radius: 16px;
    padding: 1rem;
    min-height: 120px;
    margin-bottom: .5rem;
}
.small-muted {
    opacity: .75;
    font-size: .92rem;
}
.source-box {
    border-left: 4px solid #999;
    padding-left: 0.8rem;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🦺 IA in cantiere</h1>
<p>Assistente educativo multilingue per la sicurezza nei cantieri</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Prototipo a scopo formativo. Le risposte sono costruite solo sui documenti caricati. "
    "Non sostituisce formazione, addestramento, procedure aziendali o indicazioni dei soggetti responsabili della sicurezza."
)

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
        "query": "cadute dall'alto lavori in quota rischio protezioni collettive parapetti prevenzione"
    },
    "Scale portatili": {
        "emoji": "🪜",
        "description": "Scelta, uso, stabilità, appoggio e condizioni di impiego.",
        "query": "scale portatili uso lavoro in quota stabilità appoggio presa sicurezza"
    },
    "Parapetti": {
        "emoji": "🚧",
        "description": "Parapetti provvisori, protezione dei bordi, scelta e utilizzo.",
        "query": "parapetti provvisori protezione collettiva bordi caduta dall'alto montaggio uso"
    },
    "Ponteggi e trabattelli": {
        "emoji": "🏗️",
        "description": "Ponteggi, PiMUS, trabattelli, stabilità, montaggio e uso.",
        "query": "ponteggi fissi trabattelli PiMUS montaggio uso smontaggio stabilità protezione"
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

def split_text(text, chunk_size=1300, overlap=220):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

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

@st.cache_data(show_spinner=False)
def extract_chunks(file_payloads):
    records = []

    for filename, file_bytes in file_payloads:
        reader = PdfReader(io.BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text())

            if not text:
                continue

            page_articles = extract_article_refs(text)

            for chunk_number, chunk in enumerate(split_text(text), start=1):
                chunk_articles = extract_article_refs(chunk)

                # Se il chunk non contiene il titolo dell'articolo ma la pagina sì,
                # conserva comunque il riferimento della pagina.
                article_refs = chunk_articles if chunk_articles else page_articles

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
def retrieve(query, records, top_k=8):
    texts = [r["text"] for r in records]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=60000
    )

    matrix = vectorizer.fit_transform(texts)
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
            article_text = ", articoli " + ", ".join(item["articles"])

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
                labels = ", ".join(f"art. {a}" for a in item["articles"][:6])
                article_html = f"<b>Riferimenti:</b> {labels}<br>"

            st.markdown(
                f"<div class='source-box'><b>{item['source_pretty']}</b><br>"
                f"{article_html}"
                f"Pagina {item['page']}<br>"
                f"<span class='small-muted'>File: {item['source']}</span></div>",
                unsafe_allow_html=True
            )

            with st.expander(
                f"Mostra il passaggio recuperato – pagina {item['page']}"
            ):
                excerpt = item["text"]

                if len(excerpt) > 1500:
                    excerpt = excerpt[:1500] + "..."

                st.write(excerpt)

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

# ------------------------------------------------------------
# HOME / SELEZIONE ARGOMENTO
# ------------------------------------------------------------
if st.session_state.selected_topic is None:
    st.subheader("Scegli un argomento")

    st.write(
        "Seleziona l'area sulla quale vuoi fare una domanda, seguire una micro-lezione "
        "oppure metterti alla prova con uno scenario."
    )

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

    question = st.chat_input(
        f"Scrivi una domanda su: {selected_topic}..."
    )

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
            show_sources(results)

            st.caption(
                "⚠️ Strumento a scopo formativo. Verificare sempre procedure aziendali, "
                "documentazione ufficiale e indicazioni dei soggetti della prevenzione."
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
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
            show_sources(
                st.session_state.quiz_sources,
                title="📚 Fonti del quiz"
            )

        st.caption(
            "⚠️ Il quiz ha finalità formative e non sostituisce la valutazione dei rischi "
            "o le procedure previste per l'attività reale."
        )
