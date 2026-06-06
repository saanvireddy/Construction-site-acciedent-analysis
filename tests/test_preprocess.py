import sys
import os
sys.path.insert(0, 'src')

from preprocess import clean_text, create_risk_label


def test_clean_text_basic():
    """Test basic text cleaning."""
    text = "Worker FELL from a Scaffold!!! He was injured badly."
    result = clean_text(text)
    assert 'fell' in result
    assert 'scaffold' in result
    assert '!!!' not in result
    assert result == result.lower()
    print("✅ test_clean_text_basic passed")


def test_clean_text_empty():
    """Test that empty text returns empty string."""
    assert clean_text('') == ''
    assert clean_text(None) == ''
    print("✅ test_clean_text_empty passed")


def test_clean_text_removes_stopwords():
    """Test that stopwords are removed."""
    text = "the worker was on the scaffold"
    result = clean_text(text)
    assert 'the' not in result.split()
    assert 'was' not in result.split()
    assert 'scaffold' in result
    print("✅ test_clean_text_removes_stopwords passed")


def test_clean_text_removes_numbers():
    """Test that numbers are removed."""
    text = "worker fell 15 feet from scaffold"
    result = clean_text(text)
    assert '15' not in result
    print("✅ test_clean_text_removes_numbers passed")


def test_create_risk_label_amputation():
    """Amputation = HIGH RISK."""
    row = {'Amputation': '1', 'Hospitalized': '0',
           'Loss of Eye': '0', 'NatureTitle': '', 'Final Narrative': ''}
    assert create_risk_label(row) == 1
    print("✅ test_create_risk_label_amputation passed")


def test_create_risk_label_hospitalized():
    """Hospitalization = HIGH RISK."""
    row = {'Amputation': '0', 'Hospitalized': '1',
           'Loss of Eye': '0', 'NatureTitle': '', 'Final Narrative': ''}
    assert create_risk_label(row) == 1
    print("✅ test_create_risk_label_hospitalized passed")


def test_create_risk_label_low_risk():
    """No severity flags = LOW RISK."""
    row = {'Amputation': '0', 'Hospitalized': '0',
           'Loss of Eye': '0', 'NatureTitle': 'bruise',
           'Final Narrative': 'minor bruise on arm'}
    assert create_risk_label(row) == 0
    print("✅ test_create_risk_label_low_risk passed")


def test_create_risk_label_narrative_keyword():
    """High risk keyword in narrative = HIGH RISK."""
    row = {'Amputation': '0', 'Hospitalized': '0',
           'Loss of Eye': '0', 'NatureTitle': '',
           'Final Narrative': 'worker fell from ladder'}
    assert create_risk_label(row) == 1
    print("✅ test_create_risk_label_narrative_keyword passed")


if __name__ == '__main__':
    test_clean_text_basic()
    test_clean_text_empty()
    test_clean_text_removes_stopwords()
    test_clean_text_removes_numbers()
    test_create_risk_label_amputation()
    test_create_risk_label_hospitalized()
    test_create_risk_label_low_risk()
    test_create_risk_label_narrative_keyword()
    print("\n🎉 All preprocess tests passed!")