# 🏗️ Construction Site Accident Risk Analyzer

> End-to-end NLP + Machine Learning pipeline to classify construction accident risk levels and generate AI-powered safety recommendations — deployed with Streamlit and Docker.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat)

---

## 📌 Overview

This project rebuilds and significantly extends a research project originally conducted as part of undergraduate research at MRECW. The system analyzes **18,617 real construction accident reports** from OSHA's Severe Injury Database (2015–2025) to:

- **Classify** accident reports as HIGH RISK or LOW RISK using NLP + ML
- **Extract** key entities — equipment, hazards, locations — using spaCy NER
- **Explain** predictions using TF-IDF feature importance
- **Generate** structured safety recommendations using rule-based AI (Gemini LLM integration ready)
- **Visualize** insights through an interactive Streamlit dashboard

---

## 🏆 Model Performance

| Model | CV F1 Score | CV Std |
|-------|------------|--------|
| **SVM** *(Best)* | **0.9977** | ±0.0004 |
| Random Forest | 0.9957 | ±0.0007 |
| Gradient Boosting | 0.9868 | ±0.0010 |
| Logistic Regression | 0.9782 | ±0.0020 |

**Test Set Results (Best Model — SVM):**
- AUC-ROC: **0.9867**
- F1 Score: **0.9493**
- Precision: **0.9575**
- Recall: **0.9578**
- Accuracy: **96%**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| NLP Preprocessing | NLTK, spaCy, TF-IDF (bigrams) |
| Named Entity Recognition | spaCy `en_core_web_sm` |
| ML Classification | Scikit-learn (SVM, RF, GB, LR) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| LLM Recommendations | Google Gemini (gemini-1.5-flash) |
| Dashboard | Streamlit |
| Containerization | Docker + Docker Compose |
| Testing | pytest (14 tests) |
| Data | OSHA Severe Injury Reports (2015–2025) |

---

## 📁 Project Structure
