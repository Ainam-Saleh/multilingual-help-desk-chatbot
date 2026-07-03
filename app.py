import streamlit as st
import json
import torch
import os
import zipfile
import gdown
from simpletransformers.question_answering import QuestionAnsweringModel
from datetime import datetime
from deep_translator import GoogleTranslator
import re
from difflib import SequenceMatcher


# ============================================================
# MODEL DOWNLOAD (from Google Drive)
# The trained model is too large for GitHub, so it's hosted on
# Google Drive as a zip and downloaded here the first time the
# app starts on the server.
# ============================================================
MODEL_DIR = "outputs/nile_qa_model"
MODEL_ZIP_FILE_ID = "1QkgrQGyS2rx1xUre1ZCc1TwS5MPudqZD"


def ensure_model_downloaded():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        os.makedirs("outputs", exist_ok=True)
        zip_path = "nile_qa_model.zip"
        gdown.download(id=MODEL_ZIP_FILE_ID, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("outputs")
        os.remove(zip_path)


# ============================================================
# PAGE CONFIGURATION
# Sidebar holds language selection + chat management.
# ============================================================
def setup_page_config():
    st.set_page_config(
        page_title="NileAssist - Multilingual Q&A Chat",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded"
    )


# ============================================================
# CUSTOM CSS STYLING
# Matches the indigo / ochre design used on the People and
# Projects pages: Newsreader for headings, Inter for body text,
# IBM Plex Mono for small accents. Built for a sidebar-free,
# top-bar layout.
# ============================================================
def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --ink: #1B2430;
            --paper: #F6F4EF;
            --indigo: #2F4B7C;
            --ochre: #C98A2C;
            --sage: #5F6E5C;
        }

        .main {
            background: var(--paper);
        }

        .block-container {
           padding-top: 3.5rem !important;
           padding-bottom: 1rem !important;
           max-width: 760px !important;
        }

        /* Font/color applied only to our own content areas — never
           to html/body or any broad "[class*=...]" match, since
           Streamlit's native toolbar and icon buttons live in the
           same document and get caught by anything that broad,
           breaking their icon glyphs (rendered via a ligature font
           where the "text" IS the icon, e.g. "light_mode"). */
        .main .block-container,
        [data-testid="stSidebar"] {
            font-family: 'Inter', sans-serif;
            color: var(--ink);
        }

        /* ---------- Header ---------- */
        .na-badge {
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--ochre);
            background: rgba(201,138,44,0.1);
            padding: 5px 12px;
            border-radius: 100px;
            margin-bottom: 10px;
        }

        .na-title {
            font-family: 'Newsreader', serif;
            font-weight: 500;
            font-size: 36px;
            color: var(--ink);
            margin: 0 0 4px;
        }

        .na-subtitle {
            font-size: 15px;
            color: var(--sage);
            margin: 0 0 18px;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: var(--ink);
        }

        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown li {
            color: #E9E7E1 !important;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #E9E7E1 !important;
            font-family: 'Newsreader', serif !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(233,231,225,0.15);
        }

        /* Alert boxes inside the sidebar keep their own light
           background + dark text for readability. */
        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
           background: rgba(255,255,255,0.08) !important;
           border: 1px solid rgba(255,255,255,0.2) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
            color: #E9E7E1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: rgba(255,255,255,0.06);
            color: #E9E7E1;
        }

        .na-about-link {
            display: block;
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--ochre) !important;
            text-decoration: none;
            padding: 8px 0;
        }

        .na-about-link:hover {
            color: #E9E7E1 !important;
        }

        /* ---------- Control bar ---------- */
        [data-testid="stSelectbox"] > div > div {
           background: white !important;
           border: 1px solid rgba(27,36,48,0.15) !important;
           border-radius: 8px !important;
           color: var(--ink) !important;
           min-height: 38px !important;
        }

        [data-testid="stSelectbox"] > div > div > div {
            color: var(--ink) !important;
            font-size: 14px !important;
        }

        [data-testid="stSelectbox"] label p {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--sage);
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: transparent;
            color: var(--indigo);
            border: 1px solid rgba(47,75,124,0.3);
            border-radius: 100px;
            padding: 8px 18px;
            font-weight: 500;
            font-size: 13.5px;
            font-family: 'Inter', sans-serif;
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            background: var(--indigo);
            color: white;
            border-color: var(--indigo);
        }

        .stButton > button p {
            color: inherit !important;
        }

        /* Primary action buttons (Clear all, Delete) get the ochre fill */
        .na-primary-btn .stButton > button {
            background: var(--ochre);
            color: white;
            border: none;
        }

        .na-primary-btn .stButton > button:hover {
            background: var(--indigo);
        }

        /* ---------- Chat ---------- */
        [data-testid="stChatMessage"] {
            background-color: #FFFFFF;
            border: 1px solid rgba(27,36,48,0.12);
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0;
            box-shadow: none;
        }

        /* Explicit text color so the bubble stays readable even when
           the browser/Streamlit theme is set to Dark — otherwise the
           default dark-mode text color (light gray) ends up almost
           invisible against our always-white bubble background. */
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessageContent"] {
            color: var(--ink) !important;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid rgba(27,36,48,0.12);
        }

        [data-testid="stChatInput"] textarea {
            font-family: 'Inter', sans-serif;
        }

        /* ---------- Expander (Manage chat) ---------- */
        [data-testid="stExpander"] summary {
            background-color: rgba(47,75,124,0.06);
            border-radius: 8px;
            font-weight: 500;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12.5px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        [data-testid="stExpander"] {
            border: none !important;
            margin-bottom: 8px;
        }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            background-color: #FFFFFF;
            border-radius: 8px;
            border-left: 3px solid var(--ochre);
        }

        hr {
            border-color: rgba(27,36,48,0.1);
        }

        .na-caption {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: var(--sage);
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# TRANSLATION FUNCTION
# ============================================================
def translate_to_nigerian_lang(text, target_lang='en'):
    try:
        if not text or not text.strip():
            return text
        if target_lang == 'en':
            return text
        result = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return result if result is not None else text
    except:
        return text


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.model_loaded = False
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    if "source_lang" not in st.session_state:
        st.session_state.source_lang = "en"
    if "target_lang" not in st.session_state:
        st.session_state.target_lang = "en"


# ============================================================
# DATASET LOADER
# ============================================================
@st.cache_data
def load_dataset():
    try:
        with open("nile_dataset_comprehensive.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Dataset file not found! Make sure nile_dataset_comprehensive.json is in the project folder.")
        return []


# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_qa_model():
    try:
        ensure_model_downloaded()
        use_cuda = torch.cuda.is_available()
        model = QuestionAnsweringModel(
            "distilbert",
            "outputs/nile_qa_model",
            use_cuda=use_cuda
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def ensure_model_loaded():
    """Loads the model + dataset on first use, if not already loaded."""
    if not st.session_state.model_loaded:
        with st.spinner("Setting things up for the first time — this can take a minute..."):
            st.session_state.model = load_qa_model()
            st.session_state.dataset = load_dataset()
            st.session_state.model_loaded = st.session_state.model is not None
    return st.session_state.model_loaded


# ============================================================
# TEXT CLEANER
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# SMART CONTEXT MATCHING
# ============================================================
def find_best_context(question, dataset, threshold=0.15):
    cleaned_question = clean_text(question)

    stop_words = {
        'is', 'are', 'there', 'the', 'a', 'an', 'do', 'does', 'what',
        'how', 'where', 'when', 'at', 'in', 'of', 'for', 'to', 'please',
        'my', 'me', 'we', 'our', 'i', 'can', 'will', 'would', 'should',
        'could', 'give', 'has', 'have', 'nile', 'university', 'campus',
        'student', 'students', 'tell', 'about', 'get', 'need', 'want',
        'know', 'information', 'info', 'help', 'college', 'school',
        'like', 'any', 'some', 'which', 'who', 'that', 'this',
        'with', 'from', 'up', 'out', 'on', 'or', 'and', 'if', 'by'
    }

    query_words = set(cleaned_question.split()) - stop_words
    if not query_words:
        query_words = set(cleaned_question.split())

    all_matches = []

    for item in dataset:
        cleaned_q = clean_text(item["question"])
        cleaned_a = clean_text(item["answer"])

        faq_q_words = set(cleaned_q.split()) - stop_words
        faq_a_words = set(cleaned_a.split()) - stop_words

        if not faq_q_words:
            faq_q_words = set(cleaned_q.split())

        union_q = faq_q_words.union(query_words)
        jaccard_score = len(faq_q_words.intersection(query_words)) / len(union_q) if union_q else 0

        seq_score = SequenceMatcher(None, cleaned_question, cleaned_q).ratio()

        coverage_score = len(query_words.intersection(faq_q_words)) / len(query_words) if query_words else 0

        union_a = faq_a_words.union(query_words)
        answer_score = len(faq_a_words.intersection(query_words)) / len(union_a) if union_a else 0

        combined_score = (
            jaccard_score  * 0.40 +
            coverage_score * 0.30 +
            seq_score      * 0.20 +
            answer_score   * 0.10
        )

        if combined_score > threshold:
            all_matches.append((combined_score, item))

    if not all_matches:
        return None, 0

    all_matches.sort(key=lambda x: x[0], reverse=True)
    best_score, best_match = all_matches[0]
    return best_match, best_score


# ============================================================
# RESPONSE GENERATOR
# ============================================================
def generate_response(prompt):
    source_lang = st.session_state.get("source_lang", "en")
    target_lang = st.session_state.get("target_lang", "en")

    english_prompt = prompt
    if source_lang != "en":
        try:
            english_prompt = GoogleTranslator(source=source_lang, target='en').translate(prompt)
            st.caption(f"🔄 Translated question: '{english_prompt}'")
        except:
            english_prompt = prompt

    matched_item, match_score = find_best_context(english_prompt, st.session_state.dataset)

    if matched_item and match_score >= 0.2:
        full_response = matched_item["answer"]
        confidence = match_score
    else:
        full_response = "I'm sorry, I don't have specific information on that. Please contact the Nile University help desk directly or visit https://www.nileuniversity.edu.ng for more information."
        confidence = 0.0

    if target_lang != "en":
        try:
            translated = GoogleTranslator(source='en', target=target_lang).translate(full_response)
            if translated:
                full_response = translated
        except:
            pass

    return full_response, confidence


# ============================================================
# HEADER
# ============================================================
def render_header():
    st.markdown("""
        <div class="na-badge">AIISRG Student Project</div>
        <div class="na-title">💬 NileAssist</div>
        <div class="na-subtitle">Ask about tuition, accommodation, admissions, and programs — in English, Hausa, Yoruba, or Igbo.</div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# Language selection + chat management.
# ============================================================
def render_sidebar():
    lang_map = {"English": "en", "Hausa": "ha", "Yoruba": "yo", "Igbo": "ig"}

    with st.sidebar:
        st.markdown("### 🎓 NileAssist")
        st.markdown("---")

        st.markdown("#### Language")
        from_lang = st.selectbox("Question language", options=list(lang_map.keys()), index=0, key="from_lang")
        to_lang = st.selectbox("Response language", options=list(lang_map.keys()), index=0, key="to_lang")

        st.session_state.source_lang = lang_map[from_lang]
        st.session_state.target_lang = lang_map[to_lang]

        st.markdown("---")
        st.markdown("#### Manage chat")

        if not st.session_state.messages:
            st.caption("No messages yet.")
        else:
            questions = [
                (i, msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"])
                for i, msg in enumerate(st.session_state.messages)
                if msg["role"] == "user"
            ]

            selected_label = st.selectbox(
                "Select a question to remove",
                options=[q[1] for q in questions],
                key="delete_select"
            )
            selected_index = next((q[0] for q in questions if q[1] == selected_label), None)

            if st.button("Delete this exchange", use_container_width=True):
                if selected_index is not None:
                    del st.session_state.messages[selected_index:selected_index + 2]
                    st.rerun()

            if st.button("Clear all", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<a href="https://aiisrg.com.ng/university-help-desk/projects.php" '
            'target="_blank" class="na-about-link">About this app ↗</a>',
            unsafe_allow_html=True
        )


# ============================================================
# WELCOME MESSAGE
# ============================================================
def render_welcome_message():
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown("""
        **Welcome to NileAssist!** I can help with:
        - Tuition fees
        - Accommodation
        - Admissions
        - Programs

        Type a question below to get started.
        """)


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================
def render_chat_history():
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🎓"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "confidence" in message and isinstance(message.get("confidence"), (int, float)) and message["confidence"] > 0:
                st.markdown(f'<span class="na-caption">CONFIDENCE {message["confidence"]:.0%}</span>', unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================
def main():
    setup_page_config()
    apply_custom_css()
    initialize_session_state()

    render_sidebar()
    render_header()

    if len(st.session_state.messages) == 0:
        render_welcome_message()

    render_chat_history()

    if prompt := st.chat_input("Ask a question about Nile University..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🎓"):
            if not ensure_model_loaded():
                st.error("Couldn't load the model. Please refresh and try again.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        full_response, confidence = generate_response(prompt)
                        st.markdown(full_response)
                        if confidence > 0:
                            st.markdown(f'<span class="na-caption">CONFIDENCE {confidence:.0%}</span>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "confidence": confidence
                        })
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")


if __name__ == '__main__':
    main()
