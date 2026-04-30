"""
train_roberta.py
────────────────────────────────────────────────────────────────────────
BTP-2 Phase 3 · Model 3: RoBERTa engagement tier classifier

Fine-tunes roberta-base on YouTube title + description text to predict
engagement_tier (LOW / MID / HIGH).

Hardware : Apple Silicon M-series · MPS GPU · 16 GB RAM
Device   : mps  (Apple GPU — ~5x faster than CPU)
Runtime  : ~2 hours for 3 epochs on 14,215 training samples

Run:
    cd ~/Downloads/btp2_scraper
    python3 train_roberta.py

Outputs:
    models/roberta_engagement/          ← saved model + tokenizer
    models/roberta_results.json         ← metrics
    data/plots/roberta_confusion.png
    data/plots/roberta_training_curve.png
"""
import warnings, json, time, os
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          get_linear_schedule_with_warmup)

# ── Device setup ───────────────────────────────────────────────────────
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device : {DEVICE}')

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path('data')
MODELS_DIR = Path('models')
PLOTS_DIR  = DATA_DIR / 'plots'
SAVE_DIR   = MODELS_DIR / 'roberta_engagement'
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

# ── Hyperparameters — tuned for 16 GB Apple Silicon ───────────────────
MAX_LEN    = 128     # tokens — covers 95% of title+desc pairs
BATCH_SIZE = 16      # fits in 16 GB with roberta-base
EPOCHS     = 3       # standard for fine-tuning transformers
LR         = 2e-5    # proven sweet spot for RoBERTa fine-tuning
WARMUP_PCT = 0.1     # 10% of steps for linear warmup
GRAD_ACCUM = 2       # effective batch = 16 × 2 = 32

LABEL_MAP  = {'LOW': 0, 'MID': 1, 'HIGH': 2}
INV_LABEL  = {0: 'LOW', 1: 'MID', 2: 'HIGH'}

# ── Load data ──────────────────────────────────────────────────────────
print('\nLoading data...')
train_df = pd.read_parquet(DATA_DIR / 'btp2_train_v2.parquet')
val_df   = pd.read_parquet(DATA_DIR / 'btp2_val_v2.parquet')
test_df  = pd.read_parquet(DATA_DIR / 'btp2_test_v2.parquet')

# Drop rows with no label
train_df = train_df[train_df['engagement_tier'].notna()].reset_index(drop=True)
val_df   = val_df[val_df['engagement_tier'].notna()].reset_index(drop=True)
test_df  = test_df[test_df['engagement_tier'].notna()].reset_index(drop=True)

print(f'  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}')

# ── Text preparation ───────────────────────────────────────────────────
def make_text(row):
    """Combine title and description into a single input string.

    Format: "[title] [SEP] [description truncated to 300 chars]"
    The tokenizer will further truncate to MAX_LEN tokens.
    Keeping description short avoids padding waste on long boilerplate.
    """
    title = str(row['title']) if pd.notna(row['title']) else ''
    desc  = str(row['description'])[:300] if pd.notna(row['description']) else ''
    return f"{title} </s> {desc}".strip()

train_df['text'] = train_df.apply(make_text, axis=1)
val_df['text']   = val_df.apply(make_text, axis=1)
test_df['text']  = test_df.apply(make_text, axis=1)

# ── Tokenizer ──────────────────────────────────────────────────────────
print('\nLoading RoBERTa tokenizer...')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# ── Dataset class ──────────────────────────────────────────────────────
class EngagementDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df['text'].tolist()
        self.labels = df['engagement_tier'].map(LABEL_MAP).tolist()
        self.tok    = tokenizer
        self.max    = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = EngagementDataset(train_df, tokenizer, MAX_LEN)
val_ds   = EngagementDataset(val_df,   tokenizer, MAX_LEN)
test_ds  = EngagementDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2,
                          shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────────────────────────
print('Loading roberta-base...')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=3,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Parameters: {n_params/1e6:.1f}M')

# ── Optimizer + scheduler ──────────────────────────────────────────────
total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_PCT)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f'\nTraining config:')
print(f'  Epochs          : {EPOCHS}')
print(f'  Batch size      : {BATCH_SIZE}  (effective: {BATCH_SIZE*GRAD_ACCUM})')
print(f'  Max token length: {MAX_LEN}')
print(f'  Learning rate   : {LR}')
print(f'  Total steps     : {total_steps:,}')
print(f'  Warmup steps    : {warmup_steps:,}')
print(f'  Device          : {DEVICE}')

# ── Evaluation function ────────────────────────────────────────────────
def evaluate(loader, split_name):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            lbls  = batch['label'].to(DEVICE)

            out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += out.loss.item()

            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    loss = total_loss / len(loader)
    return acc, loss, all_preds, all_labels

# ── Training loop ──────────────────────────────────────────────────────
print(f'\n{"═"*55}')
print('  TRAINING')
print(f'{"═"*55}')

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
best_epoch   = 0
t_start      = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss  = 0.0
    n_batches   = 0
    t_epoch     = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)

        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss = out.loss / GRAD_ACCUM
        loss.backward()

        train_loss += out.loss.item()
        n_batches  += 1

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Progress every 100 steps
        if (step + 1) % 100 == 0:
            elapsed = time.time() - t_epoch
            pct     = (step + 1) / len(train_loader) * 100
            print(f'  Epoch {epoch} [{step+1}/{len(train_loader)}] '
                  f'{pct:.0f}%  loss={train_loss/n_batches:.4f}  '
                  f'elapsed={elapsed:.0f}s')

    avg_train_loss = train_loss / n_batches
    val_acc, val_loss, _, _ = evaluate(val_loader, 'Val')
    epoch_time = time.time() - t_epoch

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'\n  ── Epoch {epoch}/{EPOCHS} complete ──')
    print(f'     Train loss : {avg_train_loss:.4f}')
    print(f'     Val loss   : {val_loss:.4f}')
    print(f'     Val acc    : {val_acc*100:.2f}%')
    print(f'     Time       : {epoch_time/60:.1f} min\n')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f'     ✅ New best — model saved to {SAVE_DIR}')

total_time = time.time() - t_start
print(f'\nTotal training time: {total_time/60:.1f} minutes')

# ── Load best model and evaluate on test ──────────────────────────────
print(f'\nLoading best model (epoch {best_epoch}, val={best_val_acc*100:.2f}%)...')
model = RobertaForSequenceClassification.from_pretrained(SAVE_DIR)
model = model.to(DEVICE)

print('\n' + '═'*55)
print('  FINAL EVALUATION')
print('═'*55)

val_acc,  val_loss,  val_preds,  val_labels  = evaluate(val_loader,  'Val')
test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, 'Test')

print(f'\n  Val accuracy  : {val_acc*100:.2f}%')
print(f'  Test accuracy : {test_acc*100:.2f}%')
print()
print('  Test classification report:')
print(classification_report(test_labels, test_preds,
                             target_names=['LOW','MID','HIGH']))

BTP1    = 0.612
XGB_ACC = 0.6912
print(f'  BTP-1 Random Forest : {BTP1*100:.1f}%')
print(f'  BTP-2 XGBoost       : {XGB_ACC*100:.1f}%')
print(f'  BTP-2 RoBERTa       : {test_acc*100:.2f}%')
print(f'  vs BTP-1            : {(test_acc-BTP1)*100:+.1f} pp')
print(f'  vs XGBoost          : {(test_acc-XGB_ACC)*100:+.1f} pp')

# ── Plots ──────────────────────────────────────────────────────────────
# Training curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_x = range(1, EPOCHS + 1)
axes[0].plot(epochs_x, history['train_loss'], 'o-', color='#0D9488', label='Train loss')
axes[0].plot(epochs_x, history['val_loss'],   'o-', color='#F59E0B', label='Val loss')
axes[0].set_title('Training curve — Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(epochs_x, [a*100 for a in history['val_acc']],
             'o-', color='#0F1B3C')
axes[1].set_title('Validation accuracy per epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].axhline(XGB_ACC*100, color='#F59E0B', linestyle='--', label='XGBoost 69.1%')
axes[1].axhline(BTP1*100,    color='#E11D48', linestyle='--', label='BTP-1 61.2%')
axes[1].legend()

plt.suptitle('RoBERTa Engagement Tier Classifier', fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'roberta_training_curve.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'\n  Saved → data/plots/roberta_training_curve.png')

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (labels, preds, title) in zip(axes, [
    (val_labels,  val_preds,  'Validation set'),
    (test_labels, test_preds, 'Test set'),
]):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['LOW','MID','HIGH'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nAccuracy: {accuracy_score(labels,preds)*100:.1f}%')
plt.suptitle('RoBERTa Engagement Tier Classifier', fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'roberta_confusion.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'  Saved → data/plots/roberta_confusion.png')

# ── Per-sector accuracy ────────────────────────────────────────────────
print('\n  Per-sector accuracy (test set):')
test_df['predicted'] = [INV_LABEL[p] for p in test_preds]
for sector in sorted(test_df['sector'].dropna().unique()):
    mask = test_df['sector'] == sector
    sub  = test_df[mask]
    if len(sub) < 10: continue
    acc  = (sub['engagement_tier'] == sub['predicted']).mean()
    print(f'    {sector:<20} {acc*100:.1f}%  (n={len(sub):,})')

# ── Save results ───────────────────────────────────────────────────────
results = {
    'model'           : 'RoBERTa-base',
    'task'            : 'engagement_tier_classification',
    'input'           : 'title + description (text)',
    'max_len'         : MAX_LEN,
    'batch_size'      : BATCH_SIZE,
    'epochs'          : EPOCHS,
    'best_epoch'      : best_epoch,
    'learning_rate'   : LR,
    'val_accuracy'    : round(val_acc, 4),
    'test_accuracy'   : round(test_acc, 4),
    'btp1_baseline'   : BTP1,
    'xgboost_accuracy': XGB_ACC,
    'vs_btp1_pp'      : round((test_acc - BTP1) * 100, 2),
    'vs_xgboost_pp'   : round((test_acc - XGB_ACC) * 100, 2),
    'train_time_min'  : round(total_time / 60, 1),
    'device'          : str(DEVICE),
}
results_path = MODELS_DIR / 'roberta_results.json'
results_path.write_text(json.dumps(results, indent=2))
print(f'\n  Results saved → {results_path}')

print(f'\n{"═"*55}')
print(f'  RoBERTa training complete')
print(f'  Test accuracy  : {test_acc*100:.2f}%')
print(f'  vs BTP-1       : {(test_acc-BTP1)*100:+.1f} pp')
print(f'  vs XGBoost     : {(test_acc-XGB_ACC)*100:+.1f} pp')
print(f'  Training time  : {total_time/60:.1f} minutes')
print(f'{"═"*55}')
print(f'\n  Next: python3 train_ensemble.py  (combine XGBoost + RoBERTa)')
