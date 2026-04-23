import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# --- Path Anchoring ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_tickets.joblib"
MODEL_DIR = SCRIPT_DIR.parent / "models"
GRAPHS_DIR = BASE_DIR / "graphs"
METRICS_PATH = BASE_DIR / "data" / "model_comparison.csv"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def train_baseline():
    print(f"📦 Loading Gold Standard data from {DATA_PATH}...")
    df = joblib.load(DATA_PATH)
    
    # --- Instructor Style: Data Prep ---
    X = df[['clean_text', 'text_len', 'brand_sector']]
    y = df['priority']
    
    # 60/20/20 Stratified Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # --- The Preprocessing Pipeline ---
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'clean_text'),
        ('num', RobustScaler(), ['text_len']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand_sector'])
    ])

     # --- Models to Test ---
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    }

    results = []
    best_macro_f1 = 0
    best_pipe = None
    best_model_name = ""

    print("\n🏁 Starting Model Race...")

    for name, model in models.items():
        print(f"🛠️ Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_val_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # --- Visualization: Confusion Matrix (Instructor Way) ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_val_pred, 
            display_labels=['Low', 'Medium', 'High'],
            cmap='Blues', 
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f"cm_{name.lower()}.png")
        plt.close() # Close to save memory
        
        print(f"📊 {name} -> Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        # Store for CSV export
        results.append({
            "model_name": name,
            "accuracy": acc,
            "macro_f1": macro_f1
        })
        
        # Selection logic based on Macro F1 (Honesty Guardrail)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_pipe = pipeline
            best_model_name = name

    # --- Save Comparison CSV ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_PATH, index=False)
    print(f"\n📈 Metrics comparison saved to {METRICS_PATH}")

    # --- Final Deployment ---
    print(f"🏆 Winner based on Macro F1: {best_model_name}")
    
    # The Final Exam (Sealed Test Set)
    y_test_pred = best_pipe.predict(X_test)
    print("\nFinal Test Set Report (Sealed Data):")
    print(classification_report(y_test, y_test_pred, target_names=['Low', 'Medium', 'High']))

    save_path = MODEL_DIR / "priority_model.joblib"
    joblib.dump(best_pipe, save_path)
    print(f"✅ Best pipeline saved to {save_path}")

if __name__ == "__main__":
    train_baseline()