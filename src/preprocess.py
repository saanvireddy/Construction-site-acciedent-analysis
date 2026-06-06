import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ── Construction NAICS codes ──────────────────────────────────────────────────
# NAICS codes starting with 23 = Construction industry
CONSTRUCTION_NAICS_PREFIX = '23'

# ── High-risk injury keywords for labeling ────────────────────────────────────
HIGH_RISK_KEYWORDS = [
    'amputation', 'amputat', 'fracture', 'fatal', 'death', 'died',
    'crush', 'crushed', 'electrocution', 'electrocuted', 'fall', 'fell',
    'hospitalized', 'laceration', 'burn', 'fractur', 'broken',
    'loss of eye', 'concussion', 'dislocation'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load OSHA CSV and return raw dataframe."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f" Loaded {len(df):,} total records")
    print(f"   Columns: {list(df.columns)}")
    return df


def filter_construction(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only construction industry records (NAICS starting with 23)."""
    df['Primary NAICS'] = df['Primary NAICS'].astype(str)
    construction_df = df[
        df['Primary NAICS'].str.startswith(CONSTRUCTION_NAICS_PREFIX)
    ].copy()
    print(f" Construction records: {len(construction_df):,} "
          f"({len(construction_df)/len(df)*100:.1f}% of total)")
    return construction_df


def clean_text(text: str) -> str:
    """
    Clean a single accident report narrative:
    - Lowercase
    - Remove special characters and numbers
    - Remove stopwords
    - Lemmatize
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''

    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return ' '.join(tokens)


def create_risk_label(row) -> int:
    """
    Create binary risk label:
    1 = HIGH RISK (amputation, hospitalization, loss of eye, or high-risk injury)
    0 = LOW RISK
    """
    # Direct severity flags from dataset
    if str(row.get('Amputation', '0')) == '1':
        return 1
    if str(row.get('Loss of Eye', '0')) == '1':
        return 1
    if str(row.get('Hospitalized', '0')) == '1':
        return 1

    # Check NatureTitle for high-risk keywords
    nature = str(row.get('NatureTitle', '')).lower()
    narrative = str(row.get('Final Narrative', '')).lower()

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in nature or keyword in narrative:
            return 1

    return 0


def preprocess_pipeline(filepath: str,
                         save_path: str = 'data/processed.csv') -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load data
    2. Filter construction industry
    3. Drop rows with no narrative
    4. Clean text
    5. Create risk labels
    6. Save processed data
    """
    # Load
    df = load_data(filepath)

    # Filter to construction only
    df = filter_construction(df)

    # Drop rows with missing narrative
    df = df.dropna(subset=['Final Narrative'])
    df = df[df['Final Narrative'].str.strip() != '']
    print(f" After dropping empty narratives: {len(df):,} records")

    # Parse date
    df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
    df['Year'] = df['EventDate'].dt.year

    # Clean narrative text
    print("⏳ Cleaning text (this may take a minute)...")
    df['clean_text'] = df['Final Narrative'].apply(clean_text)

    # Drop rows where clean_text is empty after processing
    df = df[df['clean_text'].str.strip() != '']

    # Create risk label
    df['high_risk'] = df.apply(create_risk_label, axis=1)

    # Print label distribution
    label_counts = df['high_risk'].value_counts()
    print(f" Label distribution:")
    print(f"   HIGH RISK (1): {label_counts.get(1, 0):,} "
          f"({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   LOW RISK  (0): {label_counts.get(0, 0):,} "
          f"({label_counts.get(0, 0)/len(df)*100:.1f}%)")

    # Save processed data
    os.makedirs('data', exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f" Saved processed data to {save_path}")

    return df


def build_tfidf(texts,
                max_features: int = 5000,
                save_path: str = 'models/tfidf_vectorizer.pkl'):
    """
    Build and save TF-IDF vectorizer.
    Returns: X (sparse matrix), vectorizer
    """
    os.makedirs('models', exist_ok=True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,                  # ignore terms appearing in <2 docs
        max_df=0.95,               # ignore terms in >95% of docs
        sublinear_tf=True          # apply log normalization
    )

    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, save_path)

    print(f" TF-IDF matrix: {X.shape[0]:,} docs × {X.shape[1]:,} features")
    print(f" Vectorizer saved to {save_path}")

    return X, vectorizer


def load_vectorizer(path: str = 'models/tfidf_vectorizer.pkl'):
    """Load saved TF-IDF vectorizer."""
    return joblib.load(path)


if __name__ == '__main__':
    # Quick test — run this file directly to verify it works
    import sys
    filepath = 'data/osha_accidents.csv'

    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        print("   Place your OSHA CSV in the data/ folder as osha_accidents.csv")
        sys.exit(1)

    df = preprocess_pipeline(filepath)
    X, vectorizer = build_tfidf(df['clean_text'])
    print("\n preprocess.py working correctly!")
    print(f"   Sample cleaned text:\n   {df['clean_text'].iloc[0][:200]}")