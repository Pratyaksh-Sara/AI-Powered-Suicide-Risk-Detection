# 📊 AI-Powered Suicide Risk Detection

This project leverages advanced deep learning models to **detect suicide risk in social media posts**, using publicly available Reddit and Twitter datasets. The models classify user-generated text into five risk categories — ranging from minimal to severe — and are integrated into a multilingual web interface.

---

## 🚀 Key Features

- 🧠 **Transformer-based Models**: Fine-tuned **BERT** and **XLNet**, with a baseline **LSTM**.
- 🖥️ **Interactive Streamlit App**: Real-time text input and risk analysis with automatic language detection & translation.
- 📈 **5-Level Risk Categorization**: Continuous probability scores mapped to risk bands (e.g., minimal, low, moderate).
- 🌐 **Multilingual Input Support**: Detects and translates non-English inputs for global applicability.
- 📄 **Academic Validation**: Developed under NTU URECA research programme with a published report.

---

## 📂 Repository Structure

| File                         | Description                                                   |
|-----------------------------|---------------------------------------------------------------|
| `LSTM_model.py`             | Baseline LSTM model training script                           |
| `XLNet_model.py`            | XLNet-based classification model                              |
| `neural_network_suicide.py` | BERT-based model with custom classifier head (final model)    |
| `suiciderisk_UI.py`         | Streamlit app UI and logic for real-time risk scoring         |
| `U2331312B_URECA_Final Paper.pdf` | Research paper detailing methodology and results       |

---

## 🔍 Suicide Risk Categories

| Risk Score Range | Risk Level     | Interpretation                                      |
|------------------|----------------|-----------------------------------------------------|
| 0.00 - 0.19      | 🟢 Minimal      | No significant indications of suicide risk          |
| 0.20 - 0.39      | 🟡 Low          | Some distress, but no strong suicidality            |
| 0.40 - 0.59      | 🟠 Moderate     | Signs of distress or mild suicidal ideation         |
| 0.60 - 0.79      | 🔴 High         | Strong signs of suicidal ideation, needs attention  |
| 0.80 - 1.00      | ⚠️ Severe      | Critical risk, may require immediate intervention   |

---

## 📊 Model Performance

| Model   | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| LSTM    | 0.72     | 0.84      | 0.52   | 0.64     | 0.77    |
| XLNet   | 0.85     | 0.87      | 0.80   | 0.84     | 0.92    |
| **BERT** | **0.90** | **0.90**  | **0.90** | **0.90** | **0.96** |

---


git clone https://github.com/Pratyaksh-Sara/AI-Powered-Suicide-Risk-Detection.git
