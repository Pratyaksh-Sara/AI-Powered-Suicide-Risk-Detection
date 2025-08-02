import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from googletrans import Translator
from langdetect import detect

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
translator = Translator()

def translate_to_english(text):
    lang = detect(text)  # Detect language
    if lang != 'en':
        translated = translator.translate(text, dest='en')
        return translated.text, lang  # Return translated text and detected language
    return text, None  # If already English, return original text with None for language

# Define the model architecture
class SuicideRiskModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(SuicideRiskModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(768, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output[0][:, 0, :]
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load pre-trained BERT model
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize the Suicide Risk Model
model = SuicideRiskModel(bert_model)

# Load trained model weights
model.load_state_dict(torch.load("suicide_model.pth", map_location=torch.device('cpu')))
model.eval()

# Function to predict risk score
def predict_text(text):
    translated_text, original_language = translate_to_english(text)  # Translate if needed
    tokenized = tokenizer(translated_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
    
    if score < 0.20:
        risk_level = "🟢 Minimal Risk"
    elif score < 0.40:
        risk_level = "🟡 Low Risk"
    elif score < 0.60:
        risk_level = "🟠 Moderate Risk"
    elif score < 0.80:
        risk_level = "🔴 High Risk"
    else:
        risk_level = "⚠️ Severe Risk"
    
    return {"risk_score": round(score, 4), "risk_category": risk_level, "translated_text": translated_text, "original_language": original_language}

# Streamlit UI
st.set_page_config(page_title="HOPE Index", page_icon="💡", layout="centered")

st.markdown(
    """
    <style>
        .main { background-color: #f5f7fa; }
        h1 { color: #333366; text-align: center; }
        .stTextArea textarea { font-size: 16px; }
        .stButton button { background-color: #4CAF50; color: white; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 HARM Index - Harm Outcome Predictive Estimator")
st.markdown("**An AI-powered tool for assessing suicide risk in text-based posts.**")
st.write("Enter a Reddit-style post below to analyze its risk level.")

# Sidebar for Information
with st.sidebar:
    st.header("ℹ️ About the Risk Index")
    st.markdown(
        """
        **HARM Index Ranges:**
        - 🟢 **Minimal Risk (0.00 - 0.19):** No significant indications of suicide risk.
        - 🟡 **Low Risk (0.20 - 0.39):** Some distress, but no strong indications of suicidality.
        - 🟠 **Moderate Risk (0.40 - 0.59):** Shows distress or mild suicidal ideation.
        - 🔴 **High Risk (0.60 - 0.79):** Strong signs of suicidal ideation, requires attention.
        - ⚠️ **Severe Risk (0.80 - 1.00):** Critical risk, may require immediate intervention.
        """
    )

# User Input
user_input = st.text_area("Enter text here:", "", height=150)

# Analyze Button
if st.button("🔍 Analyze Risk"):
    if user_input.strip():
        result = predict_text(user_input)
        st.subheader("Risk Assessment Results")
        st.write(f"**Predicted Risk Score:** `{result['risk_score']}`")
        st.markdown(f"**Risk Level:** `{result['risk_category']}`")
        if result['original_language']:
            st.markdown(f"**Translated from {result['original_language'].upper()}:** `{result['translated_text']}`")
    else:
        st.warning("⚠️ Please enter a valid text prompt to analyze.")

# Footer
st.markdown("""
    ---
    **Disclaimer:** This tool is an AI-based risk assessment and does not replace professional mental health support. If you or someone you know is in crisis, seek help from a mental health professional.
""")
