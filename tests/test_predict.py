import sys
import os
sys.path.insert(0, 'src')

from predict import predict_risk, load_model_and_vectorizer, get_top_features


def test_model_loads():
    """Test that model and vectorizer load correctly."""
    model, vectorizer = load_model_and_vectorizer()
    assert model is not None
    assert vectorizer is not None
    print("✅ test_model_loads passed")


def test_predict_returns_dict():
    """Test that predict returns correct dict structure."""
    model, vectorizer = load_model_and_vectorizer()
    result = predict_risk(
        "Worker fell from scaffold and was hospitalized.",
        model=model,
        vectorizer=vectorizer,
        include_recommendation=False
    )
    assert 'risk_level' in result
    assert 'probability' in result
    assert 'prediction' in result
    assert 'top_features' in result
    assert 'entities' in result
    print("✅ test_predict_returns_dict passed")


def test_predict_high_risk():
    """Test that severe accident is classified as HIGH RISK."""
    model, vectorizer = load_model_and_vectorizer()
    result = predict_risk(
        """Worker fell from a 20-foot scaffold and was hospitalized 
        with severe head injury and broken bones.""",
        model=model,
        vectorizer=vectorizer,
        include_recommendation=False
    )
    assert result['prediction'] == 1
    assert result['risk_level'] == 'HIGH RISK'
    assert result['probability'] > 0.5
    print("✅ test_predict_high_risk passed")


def test_predict_probability_range():
    """Test that probability is between 0 and 1."""
    model, vectorizer = load_model_and_vectorizer()
    result = predict_risk(
        "Worker slipped and bruised knee.",
        model=model,
        vectorizer=vectorizer,
        include_recommendation=False
    )
    assert 0.0 <= result['probability'] <= 1.0
    print("✅ test_predict_probability_range passed")


def test_predict_empty_text():
    """Test handling of empty text."""
    model, vectorizer = load_model_and_vectorizer()
    result = predict_risk(
        "",
        model=model,
        vectorizer=vectorizer,
        include_recommendation=False
    )
    assert result['prediction'] == -1
    assert result['risk_level'] == 'UNKNOWN'
    print("✅ test_predict_empty_text passed")


def test_top_features_not_empty():
    """Test that top features are returned for valid text."""
    model, vectorizer = load_model_and_vectorizer()
    features = get_top_features(
        vectorizer,
        "worker fell from scaffold hit concrete floor",
        n=5
    )
    assert isinstance(features, list)
    assert len(features) > 0
    print("✅ test_top_features_not_empty passed")


if __name__ == '__main__':
    test_model_loads()
    test_predict_returns_dict()
    test_predict_high_risk()
    test_predict_probability_range()
    test_predict_empty_text()
    test_top_features_not_empty()
    print("\n🎉 All predict tests passed!")