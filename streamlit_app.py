import io
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

st.set_page_config(
    page_title="IA in cantiere",
    page_icon="🦺",
    layout="centered"
)

st.title("🦺 IA in cantiere")
st.caption("Assistente educativo multilingue per la sicurezza nei cantieri")

st.info(
    "Prototipo a scopo formativo. Le risposte sono costruite solo sui documenti caricati. "
    "Non sostituisce formazione, addestramento, procedure aziendali o indicazioni dei soggetti responsabili della sicurezza."
)

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
Se devi usare un termine tecnico, spiegalo subito tra parentesi.
Non eliminare numeri, misure, obblighi o riferimenti normativi presenti nelle fonti.
Non banalizzare il contenuto tecnico.
""",
}

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
    uploaded_files = st.file_uploader(
        "Carica uno o più PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Esempi: D.Lgs. 81/08, documenti INAIL, linee guida, manuali ufficiali."
    )

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
    return filename

@st.cache_data(show_spinner=False)
def extract_chunks(file_payloads):
    records = []
    for filename, file_bytes in file_payloads:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text())
            if not text:
                continue
            for chunk_number, chunk in enumerate(split_text(text), start=1):
                records.append({
                    "source": filename,
                    "source_pretty": pretty_source_name(filename),
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                })
    return records

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
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()

def retrieve(question_it, records, top_k=8):
    texts = [r["text"] for r in records]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=60000
    )

    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question_it])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked:
        item = dict(records[idx])
        item["score"] = float(scores[idx])
        results.append(item)

    return results

def answer_from_context(client, question, results, output_language, response_style):
    context_parts = []

    for i, item in enumerate(results, start=1):
        context_parts.append(
            f"[FONTE {i}: {item['source_pretty']}, file {item['source']}, pagina {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Sei "IA in cantiere", un assistente EDUCATIVO sulla salute e sicurezza nei cantieri.

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
10. Se due fonti sembrano in contrasto, segnala che è necessaria una verifica e non scegliere arbitrariamente.

STILE RICHIESTO:
{response_style}

DOMANDA DELL'UTENTE:
{question}

CONTESTO DOCUMENTALE:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()

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

if not uploaded_files:
    st.warning("Per iniziare, carica almeno un PDF nella barra laterale.")
    st.stop()

file_payloads = tuple((f.name, f.getvalue()) for f in uploaded_files)

with st.spinner("Leggo e preparo le fonti..."):
    records = extract_chunks(file_payloads)

if not records:
    st.error(
        "Non sono riuscito a estrarre testo dai PDF. "
        "Il documento potrebbe essere una scansione composta solo da immagini."
    )
    st.stop()

st.success(
    f"Fonti pronte: {len(uploaded_files)} PDF, {len(records)} sezioni indicizzate."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Scrivi una domanda sulla sicurezza in cantiere...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    client = get_client()

    with st.chat_message("assistant"):
        with st.spinner("Cerco nelle fonti e preparo la risposta..."):
            question_it = translate_query_to_italian(client, question)

            results = retrieve(
                question_it,
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
                    results,
                    output_language,
                    response_style
                )

        st.markdown(answer)

        source_results = unique_source_results(results, max_items=5)

        if source_results:
            with st.expander("📚 Fonti utilizzate"):
                for item in source_results:
                    st.markdown(
                        f"**{item['source_pretty']}**  \n"
                        f"Pagina {item['page']}  \n"
                        f"*File: {item['source']}*"
                    )

                    with st.expander(
                        f"Mostra il passaggio recuperato – pagina {item['page']}"
                    ):
                        excerpt = item["text"]
                        if len(excerpt) > 1200:
                            excerpt = excerpt[:1200] + "..."
                        st.write(excerpt)

                    st.divider()

        st.caption(
            "⚠️ Strumento a scopo formativo. Verificare sempre procedure aziendali, "
            "documentazione ufficiale e indicazioni dei soggetti della prevenzione."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
