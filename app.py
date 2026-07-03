import streamlit as st
import json
import torch
import os
import zipfile
import gdown
from simpletransformers.question_answering import QuestionAnsweringModel
import time
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
# TRANSLATION FUNCTION
# Translates text to a target Nigerian language (Hausa, Yoruba, Igbo)
# If translation fails for any reason, it safely returns the original text
# ============================================================
def translate_to_nigerian_lang(text, target_lang='en'):
    try:
        if not text or not text.strip():
            return text
        if target_lang == 'en':
            return text
        result = GoogleTranslator(source='auto', target=target_lang).translate(text)
        if result is None:
            return text
        return result
    except:
        return text


# ============================================================
# PAGE CONFIGURATION
# Sets the browser tab title, icon, and layout of the Streamlit app
# This must be the first Streamlit command called
# ============================================================
def setup_page_config():
    st.set_page_config(
        page_title="NileAssist - Multilingual Q&A Chat",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# ============================================================
# CUSTOM CSS STYLING
# Matches the indigo / ochre design used on the People and
# Projects pages: Newsreader for headings, Inter for body text,
# IBM Plex Mono for small accents.
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
           padding-top: 1.5rem !important;
           padding-bottom: 1rem !important;
           max-width: 100% !important;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: var(--ink);
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] .stMarkdown {
            color: #F6F4EF !important;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F6F4EF !important;
            font-family: 'Newsreader', serif !important;
            font-weight: 500 !important;
        }

        /* Don't touch icon fonts (e.g. the sidebar collapse arrow) —
           overriding their font-family breaks the icon glyph. */
        [data-testid="stSidebar"] [data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [class*="material-symbols"] {
            font-family: 'Material Symbols Outlined' !important;
            color: #F6F4EF !important;
        }

        /* Keep success/warning/info/error alert text dark and readable
           against their own light backgrounds, even inside the dark sidebar. */
        [data-testid="stSidebar"] [data-testid="stAlertContentSuccess"] *,
        [data-testid="stSidebar"] [data-testid="stAlertContentWarning"] *,
        [data-testid="stSidebar"] [data-testid="stAlertContentInfo"] *,
        [data-testid="stSidebar"] [data-testid="stAlertContentError"] * {
            color: var(--ink) !important;
        }

        h1, h2, h3 {
            font-family: 'Newsreader', serif;
            color: var(--ink);
            font-weight: 500;
        }

        .stChatMessage {
            background-color: #FFFFFF;
            border: 1px solid rgba(27,36,48,0.12);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: none;
        }

        [data-testid="stChatMessageContent"] {
            background-color: transparent;
        }

        [data-testid="stMetricValue"] {
            font-family: 'Newsreader', serif;
            font-size: 26px;
            color: var(--indigo);
        }

        [data-testid="stMetricLabel"] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--sage);
        }

        .stButton>button {
            background: var(--ochre);
            color: white;
            border: none;
            border-radius: 100px;
            padding: 10px 26px;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            transition: transform 0.15s ease, background 0.15s ease;
        }

        .stButton>button:hover {
            background: var(--indigo);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] .stButton>button {
            background: var(--ochre);
            color: white;
        }

        .stChatInputContainer {
            border-top: 1px solid rgba(27,36,48,0.12);
            padding-top: 20px;
        }

        .streamlit-expanderHeader {
            background-color: rgba(47,75,124,0.06);
            border-radius: 8px;
            font-weight: 500;
            font-family: 'IBM Plex Mono', monospace;
        }

        .stAlert {
            background-color: #FFFFFF;
            border-radius: 8px;
            border-left: 3px solid var(--ochre);
        }

        [data-testid="stSelectbox"] > div > div {
           background: white !important;
           border: 1px solid rgba(27,36,48,0.15) !important;
           border-radius: 8px !important;
           color: var(--ink) !important;
           min-height: 38px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
           background: rgba(255,255,255,0.08) !important;
           border: 1px solid rgba(255,255,255,0.2) !important;
        }

        [data-testid="stSelectbox"] > div > div > div {
            color: var(--ink) !important;
            font-size: 14px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# Streamlit reruns the whole script on every interaction,
# so session_state is used to remember things between reruns
# ============================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.model_loaded = False
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    if "auto_retrieve" not in st.session_state:
        st.session_state.auto_retrieve = True


# ============================================================
# DATASET LOADER
# Reads the FAQ JSON file from disk into memory.
# @st.cache_data means it only loads once and reuses the result
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
# Loads the fine-tuned DistilBERT model.
# @st.cache_resource means the model is only loaded once.
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


# ============================================================
# TEXT CLEANER
# Removes punctuation and extra spaces from text before matching.
# Example: "How do I pay fees?" → "how do i pay fees"
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# SMART CONTEXT MATCHING
# Uses 4 methods combined to find the best FAQ match:
# Method 1 - Jaccard Similarity: keyword overlap score
# Method 2 - Sequence Matching: sentence structure similarity
# Method 3 - Keyword Coverage: % of user keywords matched
# Method 4 - Answer Keyword Check: topic confirmation
# ============================================================
def find_best_context(question, dataset, threshold=0.15):
    cleaned_question = clean_text(question)

    # Words too common to help distinguish between topics
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

        # Method 1: Jaccard Similarity
        union_q = faq_q_words.union(query_words)
        jaccard_score = len(faq_q_words.intersection(query_words)) / len(union_q) if union_q else 0

        # Method 2: Sequence Matching
        seq_score = SequenceMatcher(None, cleaned_question, cleaned_q).ratio()

        # Method 3: Keyword Coverage
        coverage_score = len(query_words.intersection(faq_q_words)) / len(query_words) if query_words else 0

        # Method 4: Answer Keyword Check
        union_a = faq_a_words.union(query_words)
        answer_score = len(faq_a_words.intersection(query_words)) / len(union_a) if union_a else 0

        # Combined weighted score
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
# Called every time the user sends a message.
# 1. Translates question to English if needed
# 2. Searches dataset for best matching FAQ
# 3. Returns answer directly if good match found
# 4. Returns polite fallback if no match found
# 5. Translates response back to user's chosen language
# ============================================================
def generate_response(prompt):
    source_lang = st.session_state.get("source_lang", "en")
    target_lang = st.session_state.get("target_lang", "en")

    # Step 1: Translate question to English for searching
    english_prompt = prompt
    if source_lang != "en":
        try:
            english_prompt = GoogleTranslator(source=source_lang, target='en').translate(prompt)
            st.caption(f"🔄 Translated question: '{english_prompt}'")
        except:
            english_prompt = prompt

    message_placeholder = st.empty()

    # Step 2: Search dataset using smart matching
    matched_item, match_score = find_best_context(english_prompt, st.session_state.dataset)

    if matched_item and match_score >= 0.2:
        # Good match — return FAQ answer directly
        full_response = matched_item["answer"]
        confidence = match_score
        st.info(f"📍 Found relevant context (match: {match_score:.1%})")
    else:
        # No match — polite fallback
        full_response = "I'm sorry, I don't have specific information on that. Please contact the Nile University help desk directly or visit https://www.nileuniversity.edu.ng for more information."
        confidence = 0.0
        st.warning("🔍 No relevant match found.")

    # Step 3: Translate response to user's chosen language
    if target_lang != "en":
        try:
            translated = GoogleTranslator(source='en', target=target_lang).translate(full_response)
            if translated:
                full_response = translated
        except:
            pass

    message_placeholder.markdown(full_response)
    return full_response, confidence


# ============================================================
# WELCOME MESSAGE
# Shown only when the chat history is empty
# ============================================================
def render_welcome_message():
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown("""
        👋 **Welcome to NileAssist!**

        I'm here to help you with questions about:
        - Tuition fees 💰
        - Accommodation 🏠
        - Admissions 📝
        - Programs 📚
        - And more!

        Just type your question below to get started!
        """)


# ============================================================
# SIDEBAR
# Left panel containing all controls:
# - Load Model button
# - Model status indicator
# - Settings toggles
# - Language selectors
# - Selective chat deletion
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.title("🎓 Nile University")
        st.markdown("### NileAssist")
        st.markdown("---")

        # Load Model button
        if st.button("🔄 Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                st.session_state.model = load_qa_model()
                st.session_state.dataset = load_dataset()
                if st.session_state.model:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded!")
                else:
                    st.error("❌ Failed to load model")

        # Model status indicator
        if st.session_state.model_loaded:
            st.success("🟢 Model Ready")
        else:
            st.warning("🟡 Model Not Loaded")

        st.markdown("---")

        # Settings toggles
        st.markdown("### ⚙️ Settings")
        st.session_state.auto_retrieve = st.toggle(
            "Auto-retrieve context",
            value=True,
            help="Automatically find relevant context from the dataset"
        )
        show_confidence = st.toggle("Show confidence scores", value=True)

        st.markdown("---")

        # Language selection
        st.markdown("### 🌍 Language")
        from_lang = st.selectbox(
            "Question language",
            options=["English", "Hausa", "Yoruba", "Igbo"],
            index=0,
            key="from_lang"
        )
        to_lang = st.selectbox(
            "Response language",
            options=["English", "Hausa", "Yoruba", "Igbo"],
            index=0,
            key="to_lang"
        )

        lang_map = {
            "English": "en",
            "Hausa": "ha",
            "Yoruba": "yo",
            "Igbo": "ig"
        }
        st.session_state.source_lang = lang_map[from_lang]
        st.session_state.target_lang = lang_map[to_lang]

        st.markdown("---")

        # --------------------------------------------------------
        # CHAT DELETION SECTION
        # Allows deleting a specific message pair (question +
        # answer together) or clearing the entire chat history.
        # --------------------------------------------------------
        st.markdown("### 🗑️ Manage Chat")

        if len(st.session_state.messages) == 0:
            st.caption("No messages to delete.")
        else:
            # Build list of user questions for the dropdown
            questions = []
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    short_q = msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"]
                    questions.append((i, short_q))

            # Dropdown to pick which message pair to delete
            selected_label = st.selectbox(
                "Select message to delete:",
                options=[q[1] for q in questions],
                key="delete_select"
            )

            # Find index of selected question
            selected_index = next(
                (q[0] for q in questions if q[1] == selected_label), None
            )

            col1, col2 = st.columns(2)

            with col1:
                # Delete selected question + its answer pair
                if st.button("🗑️ Delete", use_container_width=True):
                    if selected_index is not None:
                        del st.session_state.messages[selected_index:selected_index + 2]
                        st.rerun()

            with col2:
                # Delete all messages at once
                if st.button("🧹 Clear All", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        st.markdown("---")

        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant answers questions about Nile University.

        **How to use:**
        1. Load the model
        2. Select languages
        3. Ask a question
        4. Get instant answers!
        """)

    return show_confidence


# ============================================================
# CHAT HISTORY DISPLAY
# Loops through all previous messages and displays them
# as chat bubbles with optional confidence scores
# ============================================================
def render_chat_history(show_confidence):
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🎓"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "confidence" in message and show_confidence and isinstance(message.get("confidence"), (int, float)):
                st.caption(f"🎯 Confidence: {message['confidence']:.1%}")


# ============================================================
# FOOTER
# Displays three metrics at the bottom:
# - Total messages in this session
# - Model status
# - Current time
# ============================================================
def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 Messages", len(st.session_state.messages))
    with col2:
        st.metric("🤖 Model", "Ready" if st.session_state.model_loaded else "Not Loaded")
    with col3:
        current_time = datetime.now().strftime("%H:%M")
        st.metric("🕒 Time", current_time)


# ============================================================
# MAIN APPLICATION ENTRY POINT
# Runs everything in the correct order when app starts
# ============================================================
def main():
    setup_page_config()
    apply_custom_css()
    initialize_session_state()

    show_confidence = render_sidebar()

    st.title("💬 NileAssist - Multilingual Q&A Chat")
    st.markdown("Ask me anything about Nile University!")

    if len(st.session_state.messages) == 0:
        render_welcome_message()

    render_chat_history(show_confidence)

    if prompt := st.chat_input("Ask a question about Nile University..."):
        if not st.session_state.model_loaded:
            st.error("⚠️ Please load the model first using the sidebar button!")
        else:
            # Save and display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Thinking..."):
                    try:
                        full_response, confidence = generate_response(prompt)

                        if show_confidence and isinstance(confidence, (int, float)):
                            st.caption(f"🎯 Confidence: {confidence:.1%}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "confidence": confidence
                        })

                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        st.info("💡 Try rephrasing your question or check if the model is properly loaded.")

    render_footer()


if __name__ == '__main__':
    main()
