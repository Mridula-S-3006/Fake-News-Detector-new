import os
import streamlit as st
from transformers import pipeline

os.environ["XDG_CONFIG_HOME"] = os.path.join(os.getcwd(), ".config")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), ".cache")

os.makedirs(".config/streamlit", exist_ok=True)
os.makedirs(".cache", exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = os.path.join(os.path.expanduser("~"), ".streamlit")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.path.expanduser("~"), ".cache")
os.environ["STREAMLIT_CONFIG_FILE"] = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

os.makedirs('.streamlit', exist_ok=True)

with open('.streamlit/config.toml', 'w') as f:
    f.write("""
[server]
headless = true
port = 7860
enableCORS = false

[browser]
gatherUsageStats = false
""")
    
with open(".config/streamlit/config.toml", "w") as f:
    f.write("[browser]\ngatherUsageStats = false\n")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Classify news as **real or fake** using a transformer model (demo).")

@st.cache_resource
def load_model():
    try:
        return pipeline("text-classification", model="bert-fake-news-model")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# Example input
example_text = "The Prime Minister announced a new healthcare plan today."

# Session state to store input
if "user_input" not in st.session_state:
    st.session_state.user_input = example_text

# Input box
user_input = st.text_area("✍️ Enter news headline or article here:", 
                          value=st.session_state.user_input, height=200)

# File uploader
uploaded_file = st.file_uploader("📂 Or upload a .txt file with one article per line", type=["txt"])

# Buttons
col1, col2 = st.columns(2)
with col1:
    analyze_clicked = st.button("🚀 Analyze")
with col2:
    clear_clicked = st.button("🔄 Clear")

# Handle Clear
if clear_clicked:
    st.session_state.user_input = ""
    st.experimental_rerun()

# Handle Analyze
if analyze_clicked:
    # Handle file upload
    if uploaded_file is not None:
        articles = uploaded_file.read().decode("utf-8").strip().splitlines()
        st.info(f"📄 {len(articles)} articles found in file.")
        for idx, article in enumerate(articles, 1):
            with st.spinner(f"Analyzing Article {idx}..."):
                prediction = model(article)[0]
                label = prediction["label"]
                score = prediction["score"]
                if label == "NEGATIVE":
                    st.error(f"Article {idx}: 🛑 **FAKE** ({score:.2%})")
                else:
                    st.success(f"Article {idx}: ✅ **REAL** ({score:.2%})")
    elif user_input.strip():
        with st.spinner("Analyzing..."):
            prediction = model(user_input)[0]
            label = prediction["label"]
            score = prediction["score"]
            if label == "NEGATIVE":
                st.error(f"🛑 Possibly **FAKE** ({score:.2%} confidence)")
            else:
                st.success(f"✅ Possibly **REAL** ({score:.2%} confidence)")
    else:
        st.warning("Please enter text or upload a file.")
