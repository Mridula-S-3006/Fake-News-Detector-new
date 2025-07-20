import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Classify news as **real or fake** using a transformer model (demo).")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="bert-fake-news-model")

model = load_model()

user_input = st.text_area("✍️ Enter news headline or article here:", height=200)

if st.button("🚀 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some news content.")
    else:
        with st.spinner("Analyzing..."):
            prediction = model(user_input)[0]
            label = prediction["label"]
            score = prediction["score"]

            # Dummy logic: treat positive as real, negative as fake
            if label == "NEGATIVE":
                st.error(f"🛑 Possibly **FAKE** ({score:.2%} confidence)")
            else:
                st.success(f"✅ Possibly **REAL** ({score:.2%} confidence)")
