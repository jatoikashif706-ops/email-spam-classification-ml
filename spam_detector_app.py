
import streamlit as st
import joblib

# 1. Page Configuration & Title
st.set_page_config(page_title="AI Email Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Smart Email & SMS Spam Detector")
st.write("Paste any text below to verify whether it is **Spam** or **Ham (Legitimate)** in real time.")

# 2. Load Model & Vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# 3. User Input Section
user_input = st.text_area("Enter Message Content:", height=150, placeholder="Type or paste your email/SMS here...")

if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and Predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text).max() * 100

        st.divider()

        # Output Results
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.1f}%)者に")
            st.info("💡 **Tip:** Avoid clicking links or downloading attachments from this message.")
        else:
            st.success(f"✅ **LEGITIMATE MESSAGE (HAM)** (Confidence: {confidence:.1f}%)者に")
            st.caption("This message appears safe based on textual analysis.")
