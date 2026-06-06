import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def configure_gemini():
    """Configure Gemini API with key from .env file."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in .env file. "
            "Add it as: GOOGLE_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def generate_safety_recommendation(accident_text: str,
                                    risk_level: str,
                                    top_features: list = None) -> str:
    try:
        model = configure_gemini()

        features_text = (
            f"Key risk factors identified: {', '.join(top_features)}"
            if top_features else ""
        )

        prompt = f"""You are a certified workplace safety expert specializing 
in construction site accident prevention.

ACCIDENT REPORT:
{accident_text}

RISK CLASSIFICATION: {risk_level}
{features_text}

Provide a structured safety recommendation using exactly this format:

## Immediate Actions
[2-3 specific actions that should be taken right now]

## Root Cause Analysis  
[Most likely root cause of this type of accident]

## Prevention Measures
1. [Specific prevention step 1]
2. [Specific prevention step 2]  
3. [Specific prevention step 3]

## OSHA Standard Reference
[Most relevant OSHA standard number and title]

Keep each section concise and actionable. 
Focus on construction-specific safety protocols."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return _fallback_recommendation(accident_text, risk_level)

def _fallback_recommendation(accident_text: str,
                               risk_level: str) -> str:
    """
    Rule-based fallback recommendation when Gemini API
    is not available.
    """
    text_lower = accident_text.lower()

    # Detect accident type from keywords
    if any(w in text_lower for w in ['fall', 'fell', 'scaffold', 'ladder', 'roof']):
        return """## Immediate Actions
- Secure the fall area and ensure the injured worker receives immediate medical attention
- Inspect all scaffolding and fall protection equipment before resuming work
- Document the incident and notify OSHA within 24 hours if hospitalization occurred

## Root Cause Analysis
Fall from elevation — likely caused by inadequate fall protection, 
improper scaffold setup, or failure to use personal fall arrest system (PFAS)

## Prevention Measures
1. Ensure all workers at heights >6 feet use approved fall protection (harness, guardrails, or safety nets)
2. Conduct daily scaffold inspections before work begins
3. Provide mandatory fall protection training for all workers

## OSHA Standard Reference
OSHA 1926.502 — Fall Protection Systems Criteria and Practices"""

    elif any(w in text_lower for w in ['electric', 'electro', 'wire', 'power line']):
        return """## Immediate Actions
- Immediately cut power to the affected area
- Do not touch the worker until power is confirmed off
- Call emergency services and notify OSHA within 24 hours

## Root Cause Analysis
Electrical contact — likely caused by inadequate lockout/tagout procedures 
or proximity to unguarded power lines

## Prevention Measures
1. Implement lockout/tagout (LOTO) procedures for all electrical equipment
2. Maintain minimum 10-foot clearance from overhead power lines
3. Use insulated tools and PPE when working near electrical hazards

## OSHA Standard Reference
OSHA 1926.416 — General Electrical Requirements"""

    elif any(w in text_lower for w in ['crush', 'caught', 'machine', 'equipment']):
        return """## Immediate Actions
- Stop all machinery immediately and secure the area
- Provide immediate medical attention to the injured worker
- Do not move the worker if crush injury is suspected

## Root Cause Analysis
Caught-in/between machinery — likely caused by missing machine guards 
or failure to follow lockout/tagout procedures

## Prevention Measures
1. Install and maintain machine guards on all moving parts
2. Enforce strict lockout/tagout procedures before maintenance
3. Establish exclusion zones around heavy equipment

## OSHA Standard Reference
OSHA 1926.300 — General Requirements for Power Tools and Equipment"""

    else:
        return """## Immediate Actions
- Secure the accident site and provide immediate medical attention
- Document the incident with photos and witness statements
- Report to OSHA within required timeframes

## Root Cause Analysis
Workplace accident — requires thorough investigation to identify 
specific root cause and contributing factors

## Prevention Measures
1. Conduct a Job Hazard Analysis (JHA) for this type of work
2. Ensure all workers have proper PPE for the task
3. Review and update safety training for the work crew

## OSHA Standard Reference
OSHA 1926.20 — General Safety and Health Provisions for Construction"""


if __name__ == '__main__':
    # Test with sample accident report
    sample = """Worker fell from a scaffold while using a nail gun 
    to install roof panels. The scaffold collapsed causing 
    the worker to fall 15 feet and hit the concrete floor. 
    Worker was hospitalized with broken collarbone and head injury."""

    print(" Testing LLM Recommender...")
    print("=" * 55)

    # Test fallback (works without API key)
    result = _fallback_recommendation(sample, "HIGH RISK")
    print(" Fallback Recommendation:")
    print(result)
    print("\n llm_recommender.py working correctly!")
    print("   (Add GOOGLE_API_KEY to .env for Gemini recommendations)")