
"""
AI Emotion Detection Dashboard
Powered by DistilBERT and Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from app.predictor import EmotionPredictor


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e1b4b,
        #111827
    );
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #38bdf8;
    text-align: center;
}

.stButton button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #9333ea
    );
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    padding: 12px 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL
# ==========================================

@st.cache_resource
def load_predictor():

    return EmotionPredictor()


try:

    predictor = load_predictor()

except Exception as error:

    st.error(
        f"Model loading failed: {error}"
    )

    st.info(
        "Train the model or download its checkpoint "
        "before launching the dashboard."
    )

    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🧠 AI Emotion Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Advanced NLP | Deep Learning | DistilBERT'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 5. SIDEBAR
# ==========================================

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Emotion Prediction",
        "About Project",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.success(
    "AI Emotion Detection System"
)

st.sidebar.caption(
    "Powered by Hugging Face Transformers"
)


# ==========================================
# 6. EMOTION PREDICTION
# ==========================================

if page == "Emotion Prediction":

    st.subheader("✨ Analyze Your Text")

    st.write(
        "Enter a sentence to identify its emotional language patterns."
    )

    text = st.text_area(
        "Your text",
        placeholder=(
            "Example: I am extremely happy "
            "because I achieved my dream!"
        ),
        height=150,
        max_chars=5000
    )

    analyze = st.button(
        "🚀 Detect Emotion",
        use_container_width=True
    )

    if analyze:

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing emotional patterns..."
                ):

                    result = predictor.predict(text)

                emotion = result["predicted_emotion"]

                confidence = result["confidence"]

                st.divider()

                # Prediction result

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    st.caption("PREDICTED EMOTION")

                    st.markdown(
                        f"<h1>{emotion.upper()}</h1>",
                        unsafe_allow_html=True
                    )

                    st.markdown("</div>",
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    st.caption("MODEL CONFIDENCE")

                    st.markdown(
                        f"<h1>{confidence * 100:.2f}%</h1>",
                        unsafe_allow_html=True
                    )

                    st.markdown("</div>",
                        unsafe_allow_html=True
                    )

                # Probability scores

                st.subheader(
                    "📊 Emotion Probability Distribution"
                )

                df = pd.DataFrame(
                    [
                        {
                            "Emotion": emotion_name,
                            "Probability": score * 100
                        }
                        for emotion_name, score
                        in result["all_emotions"].items()
                    ]
                )

                df = df.sort_values(
                    "Probability",
                    ascending=False
                )

                fig = px.bar(
                    df,
                    x="Emotion",
                    y="Probability",
                    color="Emotion",
                    title="Emotion Analysis",
                    text_auto=".2f",
                    template="plotly_dark"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_range=[0, 100]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader(
                    "Detailed Emotion Scores"
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    "Emotion analysis completed!"
                )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )


# ==========================================
# 7. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.subheader("About This Project")

    st.write(
        """
        Emotion Detection from Text is an advanced
        Natural Language Processing project.

        It uses a fine-tuned DistilBERT Transformer
        model to classify six emotion categories.
        """
    )

    st.markdown("""
    ### Emotion Categories

    - 😊 Joy
    - 😢 Sadness
    - ❤️ Love
    - 😡 Anger
    - 😨 Fear
    - 😲 Surprise

    ### Technologies

    - Python
    - PyTorch
    - Hugging Face
    - DistilBERT
    - Streamlit
    - Scikit-learn
    """)


# ==========================================
# 8. MODEL INFORMATION
# ==========================================

elif page == "Model Information":

    st.subheader("Model Architecture")

    st.write(
        "Base model: DistilBERT"
    )

    st.write(
        "Task: Six-class text classification"
    )

    st.write(
        "Dataset: dair-ai/emotion"
    )

    st.write(
        "Framework: PyTorch and Hugging Face"
    )

    st.info(
        "Model confidence represents its predicted "
        "class probability, not a verified measure "
        "of a person's actual emotional state."
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
    <p style='color:#38bdf8'>
    🧠 AI Emotion Detection |
    Built with Python, NLP and Deep Learning
    </p>
    </center>
    """,
    unsafe_allow_html=True
  )
              
