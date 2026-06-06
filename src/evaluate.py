import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import seaborn as sns
import joblib
import os
import json
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    confusion_matrix, roc_curve, auc
)
from wordcloud import WordCloud


def plot_model_comparison(results_path: str = 'models/training_results.json',
                           save_path: str = 'reports/model_comparison.png'):
    """Bar chart comparing all 4 models by CV F1 score."""
    os.makedirs('reports', exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Filter only model entries
    models = {k: v for k, v in results.items()
              if k != 'best_model_metrics' and 'cv_f1_mean' in v}

    names  = list(models.keys())
    scores = [models[n]['cv_f1_mean'] for n in names]
    errors = [models[n]['cv_f1_std']  for n in names]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, scores, yerr=errors, capsize=6,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel('Weighted F1 Score (5-Fold CV)', fontsize=13)
    ax.set_title('Model Comparison — Construction Accident Risk Classification',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0.97, color='red', linestyle='--',
               alpha=0.5, label='0.97 baseline')
    ax.legend(fontsize=11)

    for bar, score, err in zip(bars, scores, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.001,
                f'{score:.4f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Model comparison chart saved to {save_path}")


def plot_confusion_matrix(model, X_test, y_test,
                           save_path: str = 'reports/confusion_matrix.png'):
    """Confusion matrix for best model."""
    os.makedirs('reports', exist_ok=True)

    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Low Risk', 'High Risk']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Best Model (SVM)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix saved to {save_path}")


def plot_roc_curve(model, X_test, y_test,
                   save_path: str = 'reports/roc_curve.png'):
    """ROC curve for best model."""
    os.makedirs('reports', exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray',
            linestyle='--', lw=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Construction Accident Risk Classifier',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" ROC curve saved to {save_path}")


def plot_wordcloud(df: pd.DataFrame,
                   text_col: str = 'Final Narrative',
                   label_col: str = 'high_risk',
                   save_path: str = 'reports/wordcloud_high_risk.png'):
    """Word cloud of most common words in HIGH RISK reports."""
    os.makedirs('reports', exist_ok=True)

    high_risk_text = ' '.join(
        df[df[label_col] == 1][text_col].dropna().astype(str)
    )

    wordcloud = WordCloud(
        width=1200, height=600,
        background_color='black',
        colormap='Reds',
        max_words=100,
        collocations=False
    ).generate(high_risk_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Words in HIGH RISK Accident Reports',
                 fontsize=16, fontweight='bold', color='black', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Word cloud saved to {save_path}")


def plot_risk_by_state(df: pd.DataFrame,
                        save_path: str = 'reports/risk_by_state.png'):
    """Top 15 states by number of high-risk construction accidents."""
    os.makedirs('reports', exist_ok=True)

    state_risk = (df[df['high_risk'] == 1]
                  .groupby('State')
                  .size()
                  .sort_values(ascending=False)
                  .head(15))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(state_risk.index, state_risk.values,
                  color='#E53935', alpha=0.85,
                  edgecolor='black', linewidth=0.6)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Number of High-Risk Accidents', fontsize=12)
    ax.set_title('Top 15 States — High-Risk Construction Accidents (2015–2025)',
                 fontsize=14, fontweight='bold')

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                str(int(bar.get_height())),
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Risk by state chart saved to {save_path}")


def plot_accidents_over_time(df: pd.DataFrame,
                              save_path: str = 'reports/accidents_over_time.png'):
    """Construction accidents per year trend."""
    os.makedirs('reports', exist_ok=True)

    df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
    yearly = (df.groupby(df['EventDate'].dt.year)
               .size()
               .reset_index(name='count'))
    yearly = yearly[yearly['EventDate'].between(2015, 2025)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly['EventDate'], yearly['count'],
            marker='o', color='#1565C0', linewidth=2.5,
            markersize=8, markerfacecolor='white',
            markeredgewidth=2)
    ax.fill_between(yearly['EventDate'], yearly['count'],
                    alpha=0.15, color='#1565C0')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title('Construction Site Accidents Over Time (2015–2025)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(yearly['EventDate'])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Accidents over time chart saved to {save_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from preprocess import build_tfidf
    from sklearn.model_selection import train_test_split

    # Load data and model
    print(" Loading data and model...")
    df = pd.read_csv('data/processed.csv')
    df = df.dropna(subset=['clean_text'])
    df = df[df['clean_text'].str.strip() != '']

    model     = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    X = vectorizer.transform(df['clean_text'])
    y = df['high_risk'].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Generate all plots
    print("\n Generating reports...")
    plot_model_comparison()
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_wordcloud(df)
    plot_risk_by_state(df)
    plot_accidents_over_time(df)

    print("\n All reports generated in reports/ folder!")