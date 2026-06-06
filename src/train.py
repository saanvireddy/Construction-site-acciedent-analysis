import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE


def train_all_models(X, y):
    """
    Train 4 classifiers with SMOTE + 5-fold cross validation.
    Returns results dict and best model name.
    """
    print(" Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f" After SMOTE: {X_res.shape[0]:,} samples "
          f"({sum(y_res==1):,} high risk, {sum(y_res==0):,} low risk)")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=42, learning_rate=0.1
        ),
        'SVM': SVC(
            kernel='rbf', probability=True,
            random_state=42, C=1.0
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n Training and evaluating models...")
    print("-" * 55)

    for name, model in models.items():
        print(f" Training {name}...")

        # Cross validation
        cv_scores = cross_val_score(
            model, X_res, y_res,
            cv=cv, scoring='f1_weighted', n_jobs=-1
        )

        # Fit on full resampled data
        model.fit(X_res, y_res)

        results[name] = {
            'model':       model,
            'cv_f1_mean':  cv_scores.mean(),
            'cv_f1_std':   cv_scores.std(),
            'cv_scores':   cv_scores.tolist()
        }

        print(f"    {name}: F1 = {cv_scores.mean():.4f} "
              f"± {cv_scores.std():.4f}")

    print("-" * 55)

    # Find best model
    best_name = max(results, key=lambda k: results[k]['cv_f1_mean'])
    best_score = results[best_name]['cv_f1_mean']
    print(f"\n Best model: {best_name} (F1 = {best_score:.4f})")

    return results, best_name


def evaluate_on_test(model, X_test, y_test, model_name: str):
    """Evaluate best model on held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred,
                                    target_names=['Low Risk', 'High Risk'])
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')

    print(f"\n Test Set Evaluation — {model_name}")
    print("=" * 55)
    print(report)
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   F1:        {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

    return {
        'model_name': model_name,
        'auc_roc':    round(auc, 4),
        'f1':         round(f1, 4),
        'precision':  round(precision, 4),
        'recall':     round(recall, 4),
    }


def save_results(results: dict, metrics: dict,
                  save_path: str = 'models/training_results.json'):
    """Save training results to JSON for later use in Streamlit."""
    os.makedirs('models', exist_ok=True)

    summary = {
        name: {
            'cv_f1_mean': round(r['cv_f1_mean'], 4),
            'cv_f1_std':  round(r['cv_f1_std'], 4),
        }
        for name, r in results.items()
    }
    summary['best_model_metrics'] = metrics

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n Results saved to {save_path}")


if __name__ == '__main__':
    import sys
    from preprocess import preprocess_pipeline, build_tfidf

    # ── Load or create processed data ────────────────────────────────────────
    processed_path = 'data/processed.csv'

    if os.path.exists(processed_path):
        print(" Loading existing processed data...")
        df = pd.read_csv(processed_path)
        df = df.dropna(subset=['clean_text'])
        df = df[df['clean_text'].str.strip() != '']
    else:
        print(" Processing raw data first...")
        df = preprocess_pipeline('data/osha_accidents.csv')

    print(f" Dataset: {len(df):,} records")

    # ── Build TF-IDF features ─────────────────────────────────────────────────
    print("\n Building TF-IDF features...")
    X, vectorizer = build_tfidf(df['clean_text'])
    y = df['high_risk'].values

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Train all models ──────────────────────────────────────────────────────
    results, best_name = train_all_models(X_train, y_train)

    # ── Evaluate best model on test set ──────────────────────────────────────
    best_model = results[best_name]['model']
    metrics = evaluate_on_test(best_model, X_test, y_test, best_name)

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f" Best model saved to models/best_model.pkl")

    # ── Save all results ──────────────────────────────────────────────────────
    save_results(results, metrics)

    print("\n Training complete!")