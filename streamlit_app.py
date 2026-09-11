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

with st.sidebar:
    st.header("Impostazioni")
    output_language_label = st.selectbox(
        "Lingua della risposta",
        list(LANGUAGES.keys()),
        index=0
    )
    output_language = LANGUAGES[output_language_label]

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
Mantieni invariati numeri, misure, sigle, articoli di legge e termini tecnici.
Restituisci SOLTANTO la traduzione italiana, senza commenti.

DOMANDA:
{question}
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()

def retrieve(question_it, records, top_k=6):
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

def answer_from_context(client, question, results, output_language):
    context_parts = []
    for i, item in enumerate(results, start=1):
        context_parts.append(
            f"[FONTE {i}: {item['source']}, pagina {item['page']}]\n{item['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
Sei "IA in cantiere", un assistente EDUCATIVO sulla salute e sicurezza nei cantieri.

REGOLE OBBLIGATORIE:
1. Rispondi esclusivamente usando le informazioni contenute nel CONTESTO.
2. Non usare conoscenze esterne e non inventare obblighi, divieti, misure, distanze,
   sanzioni, procedure, articoli di legge o requisiti tecnici.
3. Se il contesto non è sufficiente, dillo chiaramente e invita l'utente a consultare
   una fonte ufficiale o un professionista della prevenzione.
4. Mantieni invariati numeri, unità di misura e riferimenti normativi presenti nel contesto.
5. Usa frasi brevi, concrete e comprensibili.
6. Non presentare la risposta come sostitutiva della formazione o delle procedure aziendali.
7. Rispondi in {output_language}.
8. Non inventare citazioni. I riferimenti alle fonti saranno mostrati separatamente dall'app.

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

st.success(f"Fonti pronte: {len(uploaded_files)} PDF, {len(records)} sezioni indicizzate.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Scrivi una domanda sulla sicurezza in cantiere...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = get_client()

    with st.chat_message("assistant"):
        with st.spinner("Cerco nelle fonti e preparo la risposta..."):
            question_it = translate_query_to_italian(client, question)
            results = retrieve(question_it, records, top_k=6)

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
                    output_language
                )

        st.markdown(answer)

        if results:
            with st.expander("📚 Fonti recuperate"):
                seen = set()
                for item in results:
                    key = (item["source"], item["page"])
                    if key in seen:
                        continue
                    seen.add(key)
                    st.write(
                        f"• **{item['source']}**, pagina {item['page']} "
                        f"(pertinenza: {item['score']:.2f})"
                    )

        st.caption(
            "⚠️ Strumento a scopo formativo. Verificare sempre procedure aziendali, "
            "documentazione ufficiale e indicazioni dei soggetti della prevenzione."
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
