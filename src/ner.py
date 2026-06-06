import spacy
import pandas as pd
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Construction-specific equipment and hazard keywords
CONSTRUCTION_EQUIPMENT = [
    'scaffold', 'ladder', 'forklift', 'crane', 'excavator', 'saw',
    'drill', 'nail gun', 'hammer', 'grinder', 'compressor', 'conveyor',
    'bulldozer', 'backhoe', 'lift', 'scissor lift', 'boom lift',
    'trencher', 'concrete mixer', 'jackhammer', 'welding', 'torch'
]

HAZARD_TYPES = [
    'fall', 'struck', 'caught', 'electrocution', 'explosion',
    'collapse', 'crush', 'cut', 'burn', 'slip', 'trip'
]


def extract_entities(text: str) -> dict:
    """
    Extract named entities + construction-specific keywords
    from accident report text.
    """
    if not text or str(text).strip() == '':
        return {
            'equipment': [],
            'locations': [],
            'organizations': [],
            'hazards': [],
            'body_parts': []
        }

    doc = nlp(str(text)[:1000])  # limit to 1000 chars for speed

    entities = {
        'equipment': [],
        'locations': [],
        'organizations': [],
        'hazards': [],
        'body_parts': []
    }

    # spaCy NER
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'COMPANY'):
            entities['organizations'].append(ent.text.lower())
        elif ent.label_ in ('GPE', 'LOC', 'FAC'):
            entities['locations'].append(ent.text.lower())

    # Construction keyword matching
    text_lower = text.lower()
    for equip in CONSTRUCTION_EQUIPMENT:
        if equip in text_lower:
            entities['equipment'].append(equip)

    for hazard in HAZARD_TYPES:
        if hazard in text_lower:
            entities['hazards'].append(hazard)

    return entities


def get_top_equipment(df: pd.DataFrame,
                       text_col: str = 'Final Narrative',
                       n: int = 20) -> list:
    """Get most frequently mentioned equipment across all reports."""
    all_equipment = []
    for text in df[text_col].dropna():
        ents = extract_entities(text)
        all_equipment.extend(ents['equipment'])
    return Counter(all_equipment).most_common(n)


def get_top_hazards(df: pd.DataFrame,
                     text_col: str = 'Final Narrative',
                     n: int = 10) -> list:
    """Get most frequently mentioned hazard types."""
    all_hazards = []
    for text in df[text_col].dropna():
        ents = extract_entities(text)
        all_hazards.extend(ents['hazards'])
    return Counter(all_hazards).most_common(n)


if __name__ == '__main__':
    # Quick test
    sample = """Worker fell from a scaffold while using a nail gun 
    to install roof panels. The scaffold collapsed causing 
    the worker to fall 15 feet and hit the concrete floor."""

    result = extract_entities(sample)
    print(" NER test:")
    print(f"   Equipment: {result['equipment']}")
    print(f"   Hazards:   {result['hazards']}")
    print(f"   Locations: {result['locations']}")
    print("\n ner.py working correctly!")