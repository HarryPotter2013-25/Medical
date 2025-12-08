# =============================================================
# Educational Symptom-Matching System
# For educational purposes only. Not a medical diagnosis.
# =============================================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Sample Illness Dataset
# -------------------------------
illnesses = [
    "Common Cold","Flu","Strep Throat","Food Poisoning","Stomach Bug",
    "Migraine","COVID-19","Allergies","Sinus Infection","Bronchitis",
    "UTI","Ear Infection","Chickenpox","Measles","Gastroenteritis",
    "Pneumonia","Tonsillitis","Influenza B","Mononucleosis","Malaria",
]

symptom_keywords = [
    "fever cough sore throat runny nose fatigue headache",
    "fever cough headache body aches chills fatigue",
    "sore throat pain swallowing fever headache",
    "nausea vomiting diarrhea abdominal pain fatigue",
    "nausea vomiting diarrhea stomach cramps",
    "headache nausea sensitivity light sound fatigue",
    "fever cough fatigue loss smell taste headache",
    "sneezing runny nose itchy eyes fatigue",
    "nasal congestion facial pain headache postnasal drip",
    "cough phlegm fever fatigue shortness breath",
    "burning urination frequent urination lower abdomen pain",
    "ear pain hearing loss fluid discharge fever",
    "fever rash itchy spots fatigue malaise",
    "fever rash Koplik spots fatigue cough",
    "nausea vomiting diarrhea cramps dehydration",
    "cough fever shortness breath chest pain fatigue",
    "sore throat difficulty swallowing fever swollen tonsils",
    "fever chills cough fatigue body aches",
    "fatigue fever sore throat swollen lymph nodes",
    "fever chills headache fatigue sweating nausea",
]

prescriptions = [
    "Rest, hydration, paracetamol (educational)",
    "Rest, hydration, oseltamivir (educational)",
    "Warm saltwater gargle, pain relief, amoxicillin (educational)",
    "Hydration, antiemetics, dietary adjustments (educational)",
    "Hydration, electrolyte drinks, dietary adjustments (educational)",
    "Rest, hydration, pain relief (educational)",
    "Isolation, rest, fever management (educational)",
    "Antihistamines, nasal spray, rest (educational)",
    "Decongestants, nasal irrigation (educational)",
    "Cough suppressant, hydration, rest (educational)",
    "Hydration, pain relief, cranberry supplements (educational)",
    "Pain relief, warm compress, rest (educational)",
    "Calamine lotion, fever management (educational)",
    "Rest, hydration, fever management (educational)",
    "Hydration, dietary adjustments, rest (educational)",
    "Amoxicillin or azithromycin (educational), rest, hydration",
    "Pain relief, warm saltwater gargle, penicillin (educational)",
    "Rest, hydration, oseltamivir (educational)",
    "Rest, hydration, pain relief (educational)",
    "Chloroquine or artemisinin (educational), rest, hydration",
]

# Build DataFrame
illnesses = illnesses[:len(symptom_keywords)]
df_illness = pd.DataFrame({
    "illness": illnesses,
    "symptom_text": symptom_keywords,
    "prescription_edu": prescriptions[:len(symptom_keywords)]
})

# -------------------------------
# 2. TF-IDF Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_illness['symptom_text'])

# -------------------------------
# 3. Symptom Matching Function
# -------------------------------
def match_illness(user_input, top_n=5):
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    top_indices = sim_scores.argsort()[::-1][:top_n]

    user_words = set(user_input.lower().split())
    results = []

    for idx in top_indices:
        illness_text = df_illness.iloc[idx]['symptom_text'].lower()
        illness_words = set(illness_text.split())
        matched_keywords = user_words & illness_words

        results.append({
            "illness": df_illness.iloc[idx]['illness'],
            "similarity": round(sim_scores[idx]*100,2),
            "keywords_matched": ", ".join(matched_keywords) if matched_keywords else "None",
            "prescription_edu": df_illness.iloc[idx]['prescription_edu']
        })
    return results

# -------------------------------
# 4. Streamlit App UI
# -------------------------------
st.title("🩺 Educational Symptom-Matching System")
st.markdown("**⚠️ For educational purposes only. Not a medical diagnosis.**")

user_input = st.text_area("Enter your symptoms:", height=120)

if st.button("Analyze Symptoms"):
    if not user_input.strip():
        st.warning("Please enter your symptoms.")
    else:
        matches = match_illness(user_input)
        st.subheader("Top Possible Illnesses (Educational Use Only):")
        for i, m in enumerate(matches, start=1):
            st.markdown(f"**{i}. {m['illness']}** - Similarity: {m['similarity']}%")
            st.markdown(f"Keywords matched: {m['keywords_matched']}")
            st.markdown(f"Example prescription (educational): {m['prescription_edu']}")
            st.markdown("---")

# Optional: Add a new illness
st.subheader("Add a New Illness (Educational Only)")
new_name = st.text_input("Illness Name:")
new_keywords = st.text_area("Keywords (space-separated)")
new_prescription = st.text_area("Educational Prescription")

if st.button("Add New Illness"):
    if not new_name.strip() or not new_keywords.strip() or not new_prescription.strip():
        st.warning("Please enter all fields to add a new illness.")
    else:
        df_illness.loc[len(df_illness)] = {
            "illness": new_name,
            "symptom_text": new_keywords,
            "prescription_edu": new_prescription
        }
        tfidf_matrix = vectorizer.fit_transform(df_illness['symptom_text'])
        st.success(f"Added new illness '{new_name}' with educational prescription!")
