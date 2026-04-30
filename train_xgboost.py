"""
train_xgboost.py
────────────────────────────────────────────────────────────────────────
BTP-2 Phase 3 · Model 1: XGBoost engagement tier classifier

Predicts: engagement_tier (LOW / MID / HIGH)
Features: 34 structured features from Groups B, C, D, F + duration
Dataset : data/btp2_train_v2.parquet  (14,215 rows)
Val     : data/btp2_val_v2.parquet    (3,046 rows)
Test    : data/btp2_test_v2.parquet   (3,047 rows)

Run:
    python3 train_xgboost.py

Outputs:
    models/xgboost_engagement.json      ← trained model
    models/xgboost_results.json         ← metrics + feature importance
    data/plots/xgb_confusion_matrix.png
    data/plots/xgb_feature_importance.png
    data/plots/xgb_shap_summary.png     (if shap installed)
"""
import warnings, json, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path('data')
MODELS_DIR = Path('models')
PLOTS_DIR  = DATA_DIR / 'plots'
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
print('Loading data...')
train = pd.read_parquet(DATA_DIR / 'btp2_train_v2.parquet')
val   = pd.read_parquet(DATA_DIR / 'btp2_val_v2.parquet')
test  = pd.read_parquet(DATA_DIR / 'btp2_test_v2.parquet')

config = json.loads((DATA_DIR / 'feature_config.json').read_text())
FEATURES = [f for f in config['xgboost_features']
            if f in train.columns]

print(f'  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}')
print(f'  Features: {len(FEATURES)}')

# ── Prepare features ───────────────────────────────────────────────────
def prepare(df):
    X = df[FEATURES].copy()

    # Impute nulls — strategy per column type
    # flesch_readability null = empty description → score of 0 (unreadable)
    X['flesch_readability']  = X['flesch_readability'].fillna(0).clip(0, 121)
    # duration null = live stream → fill with median
    X['duration_seconds']    = X['duration_seconds'].fillna(
                                   X['duration_seconds'].median())
    # caps_ratio null = emoji-only title → 0
    X['title_caps_ratio']    = X['title_caps_ratio'].fillna(0)

    # Boolean columns → int (XGBoost handles both but int is explicit)
    bool_cols = ['title_has_number', 'title_has_question', 'cta_presence',
                 'hashtag_optimal', 'hashtag_zero', 'hashtag_spam',
                 'is_peak_local']
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].fillna(False).astype(int)

    return X

# Label encoding: LOW=0, MID=1, HIGH=2 (alphabetical by default)
le = LabelEncoder()
le.fit(['HIGH', 'LOW', 'MID'])   # fix order explicitly
label_map = {'LOW': 0, 'MID': 1, 'HIGH': 2}
inv_label  = {0: 'LOW', 1: 'MID', 2: 'HIGH'}

X_train = prepare(train);  y_train = train['engagement_tier'].map(label_map)
X_val   = prepare(val);    y_val   = val['engagement_tier'].map(label_map)
X_test  = prepare(test);   y_test  = test['engagement_tier'].map(label_map)

print(f'  Nulls after imputation: {X_train.isnull().sum().sum()}')

# ── Model definition ───────────────────────────────────────────────────
# Hyperparameters chosen for a 14k-row dataset on CPU:
# - n_estimators=500  with early stopping prevents overfitting
# - max_depth=6       balanced bias/variance for tabular data
# - learning_rate=0.05 slower but more stable
# - subsample=0.8     row sampling reduces variance
# - colsample_bytree=0.8  feature sampling
# - use_label_encoder deprecated in newer versions — disabled
MODEL_PARAMS = dict(
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 3,
    gamma              = 0.1,
    reg_alpha          = 0.1,    # L1 regularization
    reg_lambda         = 1.0,    # L2 regularization
    objective          = 'multi:softprob',
    num_class          = 3,
    eval_metric        = 'mlogloss',
    random_state       = 42,
    n_jobs             = -1,     # use all CPU cores
    verbosity          = 0,
)

print('\nTraining XGBoost...')
t0 = time.time()

model = xgb.XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=30)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
train_time = time.time() - t0
print(f'  Training time: {train_time:.1f}s')
print(f'  Best iteration: {model.best_iteration}')

# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate(X, y_true, split_name):
    y_pred  = model.predict(X)
    y_prob  = model.predict_proba(X)
    acc     = accuracy_score(y_true, y_pred)
    report  = classification_report(
                  y_true, y_pred,
                  target_names=['LOW', 'MID', 'HIGH'],
                  output_dict=True)
    print(f'\n  {split_name} accuracy: {acc*100:.2f}%')
    print(classification_report(y_true, y_pred,
                                 target_names=['LOW','MID','HIGH']))
    return acc, report, y_pred

print('\n' + '═'*55)
print('  EVALUATION RESULTS')
print('═'*55)

train_acc, train_report, train_pred = evaluate(X_train, y_train, 'Train')
val_acc,   val_report,   val_pred   = evaluate(X_val,   y_val,   'Val')
test_acc,  test_report,  test_pred  = evaluate(X_test,  y_test,  'Test')

# BTP-1 baseline to beat
BTP1_ACCURACY = 0.612
improvement   = (test_acc - BTP1_ACCURACY) * 100
print(f'  BTP-1 baseline (Random Forest) : {BTP1_ACCURACY*100:.1f}%')
print(f'  BTP-2 XGBoost                  : {test_acc*100:.2f}%')
print(f'  Improvement                    : +{improvement:.1f} percentage points')
if test_acc > BTP1_ACCURACY:
    print(f'  ✅ Beats BTP-1 baseline')
else:
    print(f'  ⚠️  Does not beat baseline yet — tune hyperparameters')

# ── Confusion matrix ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (y_true, y_pred, title) in zip(axes, [
    (y_val,  val_pred,  'Validation set'),
    (y_test, test_pred, 'Test set'),
]):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['LOW','MID','HIGH'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nAccuracy: {accuracy_score(y_true, y_pred)*100:.1f}%')
plt.suptitle('XGBoost Engagement Tier Classifier', fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'xgb_confusion_matrix.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'\n  Saved → data/plots/xgb_confusion_matrix.png')

# ── Feature importance ─────────────────────────────────────────────────
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#0D9488' if v > importance.median() else '#F59E0B'
          for v in importance.values]
importance.plot(kind='barh', ax=ax, color=colors[::-1])
ax.set_title('XGBoost Feature Importance (Gain)', fontsize=13)
ax.set_xlabel('Importance score')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'xgb_feature_importance.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'  Saved → data/plots/xgb_feature_importance.png')

print(f'\n  Top 10 most important features:')
for feat, score in importance.head(10).items():
    print(f'    {feat:<35} {score:.4f}')

print(f'\n  Bottom 5 least important features:')
for feat, score in importance.tail(5).items():
    print(f'    {feat:<35} {score:.4f}')

# ── SHAP values (optional — install with: pip3 install shap) ───────────
try:
    import shap
    print('\n  Computing SHAP values...')
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    fig, ax = plt.subplots(figsize=(10, 8))
    # shap_values is list of 3 arrays (one per class)
    # Show SHAP for HIGH class (most interesting for creators)
    shap.summary_plot(shap_values[2], X_val,
                      feature_names=FEATURES,
                      show=False, plot_type='bar')
    plt.title('SHAP Feature Importance — HIGH engagement class', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'xgb_shap_summary.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved → data/plots/xgb_shap_summary.png')
except ImportError:
    print('  SHAP not installed — skipping. Run: pip3 install shap')

# ── Per-sector accuracy ────────────────────────────────────────────────
print('\n  Per-sector accuracy (test set):')
test_with_pred = test.copy()
test_with_pred['predicted'] = [inv_label[p] for p in test_pred]
for sector in sorted(test['sector'].dropna().unique()):
    mask = test_with_pred['sector'] == sector
    sub  = test_with_pred[mask]
    if len(sub) < 10: continue
    acc  = (sub['engagement_tier'] == sub['predicted']).mean()
    print(f'    {sector:<20} {acc*100:.1f}%  (n={len(sub):,})')

# ── Save model and results ─────────────────────────────────────────────
model_path = MODELS_DIR / 'xgboost_engagement.json'
model.save_model(model_path)
print(f'\n  Model saved → {model_path}')

results = {
    'model'          : 'XGBoost',
    'task'           : 'engagement_tier_classification',
    'n_features'     : len(FEATURES),
    'features'       : FEATURES,
    'train_rows'     : len(train),
    'val_rows'       : len(val),
    'test_rows'      : len(test),
    'train_accuracy' : round(train_acc, 4),
    'val_accuracy'   : round(val_acc, 4),
    'test_accuracy'  : round(test_acc, 4),
    'btp1_baseline'  : BTP1_ACCURACY,
    'improvement_pp' : round(improvement, 2),
    'best_iteration' : int(model.best_iteration),
    'train_time_sec' : round(train_time, 1),
    'top_features'   : importance.head(10).round(4).to_dict(),
    'val_report'     : val_report,
    'test_report'    : test_report,
}
results_path = MODELS_DIR / 'xgboost_results.json'
results_path.write_text(json.dumps(results, indent=2))
print(f'  Results saved → {results_path}')

print(f'\n{"═"*55}')
print(f'  XGBoost training complete')
print(f'  Test accuracy : {test_acc*100:.2f}%')
print(f'  vs BTP-1      : {BTP1_ACCURACY*100:.1f}%  '
      f'({"+" if improvement > 0 else ""}{improvement:.1f} pp)')
print(f'{"═"*55}')
print(f'\n  Next: python3 train_catboost.py  (sector classifier)')
print(f'  Then: python3 train_roberta.py   (text model)')
