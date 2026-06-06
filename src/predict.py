import joblib
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
from preprocess import clean_text
from ner import extract_entities
from llm_recommender import generate_safety_recommendation


def load_model_and_vectorizer(
        model_path: str = 'models/best_model.pkl',
        vectorizer_path: str = 'models/tfidf_vectorizer.pkl'):
    """Load saved model and vectorizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def get_top_features(vectorizer, text: str, n: int = 10) -> list:
    """
    Get top TF-IDF features (keywords) that are present
    in the given text — used to explain the prediction.
    """
    feature_names = vectorizer.get_feature_names_out()
    X = vectorizer.transform([text])
    scores = X.toarray()[0]

    # Get indices of top scoring features
    top_indices = np.argsort(scores)[::-1][:n]
    top_features = [
        feature_names[i]
        for i in top_indices
        if scores[i] > 0
    ]
    return top_features


def predict_risk(text: str,
                  model=None,
                  vectorizer=None,
                  include_recommendation: bool = True) -> dict:
    """
    Full prediction pipeline for a single accident report.

    Args:
        text: Raw accident report narrative
        model: Loaded model (loads from disk if None)
        vectorizer: Loaded vectorizer (loads from disk if None)
        include_recommendation: Whether to generate LLM recommendation

    Returns:
        Dictionary with prediction results
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()

    # Clean text
    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            'risk_level':       'UNKNOWN',
            'probability':      0.0,
            'prediction':       -1,
            'top_features':     [],
            'entities':         {},
            'recommendation':   'Could not process empty text.'
        }

    # Vectorize
    X = vectorizer.transform([cleaned])

    # Predict
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # probability of HIGH RISK

    # Risk level label
    risk_level = 'HIGH RISK' if prediction == 1 else 'LOW RISK'

    # Top features driving prediction
    top_features = get_top_features(vectorizer, cleaned)

    # Named entity extraction
    entities = extract_entities(text)

    # Safety recommendation
    recommendation = ''
    if include_recommendation:
        recommendation = generate_safety_recommendation(
            accident_text=text,
            risk_level=risk_level,
            top_features=top_features
        )

    return {
        'risk_level':     risk_level,
        'probability':    round(float(probability), 4),
        'prediction':     int(prediction),
        'top_features':   top_features,
        'entities':       entities,
        'recommendation': recommendation
    }


if __name__ == '__main__':
    print(" Testing predict.py...")
    print("=" * 55)

    test_cases = [
        {
            'label': 'HIGH RISK — Fall from scaffold',
            'text': """Worker fell from a scaffold while using a nail gun 
            to install roof panels. The scaffold collapsed causing 
            the worker to fall 15 feet and hit the concrete floor. 
            Worker was hospitalized with broken collarbone and head injury."""
        },
        {
            'label': 'HIGH RISK — Electrocution',
            'text': """Electrician was electrocuted while working on 
            an unguarded power panel. Worker contacted a live wire 
            and was thrown backwards. Rushed to hospital with severe burns."""
        },
        {
            'label': 'LOW RISK — Minor incident',
            'text': """Worker slipped on wet floor and bruised their knee. 
            First aid was administered on site. 
            Worker returned to work the same day."""
        }
    ]

    model, vectorizer = load_model_and_vectorizer()

    for case in test_cases:
        print(f"\n Test: {case['label']}")
        print("-" * 45)
        result = predict_risk(
            case['text'],
            model=model,
            vectorizer=vectorizer,
            include_recommendation=True
        )
        print(f"   Risk Level:   {result['risk_level']}")
        print(f"   Probability:  {result['probability']:.1%}")
        print(f"   Top Features: {result['top_features'][:5]}")
        print(f"   Equipment:    {result['entities']['equipment']}")
        print(f"   Hazards:      {result['entities']['hazards']}")
        print(f"\n   Recommendation Preview:")
        # Show first 3 lines of recommendation
        rec_lines = result['recommendation'].strip().split('\n')[:4]
        for line in rec_lines:
            print(f"   {line}")

    print("\n\n predict.py working correctly!")