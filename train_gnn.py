"""
train_gnn.py
────────────────────────────────────────────────────────────────────────
BTP-2 Phase 3 · Model 5: GNN Hashtag Recommender

Builds a co-occurrence graph from the hashtag corpus, trains a
GraphSAGE model to learn tag embeddings, then recommends hashtags
for a video based on its text features + existing tags.

Architecture:
  1. Build co-occurrence graph  (tags = nodes, co-occurrence = edges)
  2. Train GraphSAGE           (learns neighbourhood-aware embeddings)
  3. Recommend via similarity  (cosine sim between video embedding + tag embeddings)
  4. Evaluate with Jaccard     (predicted ∩ actual / predicted ∪ actual)

Target: Jaccard > 0.460  (BTP-1 TF-IDF baseline)
Stretch: Jaccard > 0.550

Run:
    python3 train_gnn.py

Outputs:
    models/gnn_hashtag/          ← saved model + graph data
    models/gnn_results.json
    data/plots/gnn_training.png
    data/plots/gnn_cooccurrence.png
"""
import warnings, json, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import torch
import torch.nn.functional as F
from torch import nn

# Try torch_geometric — graceful fallback to custom GNN if not installed
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    HAS_PYGEOM = True
    print('torch-geometric available — using GraphSAGE')
except ImportError:
    HAS_PYGEOM = False
    print('torch-geometric not found — using custom lightweight GNN')

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_DIR   = Path('data')
MODELS_DIR = Path('models')
PLOTS_DIR  = DATA_DIR / 'plots'
SAVE_DIR   = MODELS_DIR / 'gnn_hashtag'
for d in [MODELS_DIR, PLOTS_DIR, SAVE_DIR]:
    d.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
MIN_TAG_FREQ    = 3     # ignore tags appearing fewer than 3 times
MIN_COOCCUR     = 2     # minimum co-occurrences to form an edge
TOP_K_TAGS      = 3000  # vocabulary size
EMBED_DIM       = 64    # tag embedding dimension
HIDDEN_DIM      = 128
EPOCHS          = 100
LR              = 0.01
TOP_N_RECOMMEND = 5     # how many tags to recommend per video

# ── Load data ──────────────────────────────────────────────────────────
print('\nLoading dataset...')
df = pd.read_parquet(DATA_DIR / 'btp2_v1_labeled.parquet')
df = df[df['hashtags_normalized'].notna()].reset_index(drop=True)

# Normalise hashtag column — handle numpy arrays
def to_list(x):
    if x is None: return []
    if hasattr(x, 'tolist'): return [t for t in x.tolist() if t]
    if isinstance(x, list):  return [t for t in x if t]
    return []

df['tags'] = df['hashtags_normalized'].apply(to_list)
videos_with_tags = df[df['tags'].apply(len) > 0]
print(f'  Total videos         : {len(df):,}')
print(f'  Videos with tags     : {len(videos_with_tags):,}')

# ── Build vocabulary ───────────────────────────────────────────────────
print('\nBuilding tag vocabulary...')
all_tags = [t for tags in df['tags'] for t in tags]
tag_freq = Counter(all_tags)
# Keep only frequent tags
vocab = [t for t, c in tag_freq.most_common(TOP_K_TAGS) if c >= MIN_TAG_FREQ]
tag2idx = {t: i for i, t in enumerate(vocab)}
idx2tag = {i: t for t, i in tag2idx.items()}
V = len(vocab)

print(f'  Raw unique tags      : {len(tag_freq):,}')
print(f'  Vocab (freq≥{MIN_TAG_FREQ})     : {V:,}')
print(f'  Top 10 tags: {vocab[:10]}')

# ── Build co-occurrence graph ──────────────────────────────────────────
print('\nBuilding co-occurrence graph...')
cooccur = defaultdict(int)
for tags in df['tags']:
    idxs = list({tag2idx[t] for t in tags if t in tag2idx})
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            a, b = min(idxs[i],idxs[j]), max(idxs[i],idxs[j])
            cooccur[(a,b)] += 1

# Filter edges by minimum co-occurrence
edges_raw = [(a, b, w) for (a,b), w in cooccur.items() if w >= MIN_COOCCUR]
print(f'  Total edges (freq≥{MIN_COOCCUR}) : {len(edges_raw):,}')

# Build edge index tensor
src = [a for a,b,w in edges_raw] + [b for a,b,w in edges_raw]
dst = [b for a,b,w in edges_raw] + [a for a,b,w in edges_raw]
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Edge weights (log-normalized)
weights = [np.log1p(w) for a,b,w in edges_raw]
weights = weights + weights  # bidirectional
edge_weight = torch.tensor(weights, dtype=torch.float32)

# Node features: one-hot + frequency embedding
node_freq = torch.tensor(
    [np.log1p(tag_freq.get(vocab[i], 0)) for i in range(V)],
    dtype=torch.float32
).unsqueeze(1)

print(f'  Nodes: {V:,}  Edges: {len(src):,} (bidirectional)')

# ── Model definition ───────────────────────────────────────────────────
if HAS_PYGEOM:
    class GraphSAGERecommender(nn.Module):
        """Two-layer GraphSAGE with mean aggregation."""
        def __init__(self, in_channels, hidden, out):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return F.normalize(x, p=2, dim=1)

    # Build PyG Data object
    graph_data = Data(
        x=node_freq,
        edge_index=edge_index,
    ).to(DEVICE)

    model = GraphSAGERecommender(
        in_channels=1, hidden=HIDDEN_DIM, out=EMBED_DIM
    ).to(DEVICE)

else:
    # Lightweight custom GNN using sparse message passing
    class LightGNN(nn.Module):
        """Custom 2-layer GNN with mean neighbourhood aggregation."""
        def __init__(self, n_nodes, hidden, out):
            super().__init__()
            self.embed   = nn.Embedding(n_nodes, hidden)
            self.lin1    = nn.Linear(hidden, hidden)
            self.lin2    = nn.Linear(hidden, out)
            self.dropout = nn.Dropout(0.3)

        def forward(self, node_ids, adj_dict):
            # Layer 1: aggregate neighbours
            h = self.embed(node_ids)
            agg = torch.zeros_like(h)
            for i, nid in enumerate(node_ids.tolist()):
                nbrs = adj_dict.get(nid, [])
                if nbrs:
                    nbr_emb = self.embed(torch.tensor(nbrs[:50],
                                         device=node_ids.device))
                    agg[i] = nbr_emb.mean(0)
            h = (h + agg) / 2
            h = self.lin1(h).relu()
            h = self.dropout(h)
            h = self.lin2(h)
            return F.normalize(h, p=2, dim=1)

    # Build adjacency dict
    adj_dict = defaultdict(list)
    for a,b,w in edges_raw:
        adj_dict[a].append(b)
        adj_dict[b].append(a)

    all_node_ids = torch.arange(V, device=DEVICE)
    model = LightGNN(V, HIDDEN_DIM, EMBED_DIM).to(DEVICE)

# ── Training: link prediction loss ────────────────────────────────────
# Train the GNN to predict which tags co-occur.
# Positive pairs = tags that co-occur
# Negative pairs = random tag pairs (unlikely to co-occur)
print(f'\nTraining GNN ({EPOCHS} epochs)...')

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# Prepare positive pairs
pos_src = torch.tensor([a for a,b,w in edges_raw], dtype=torch.long)
pos_dst = torch.tensor([b for a,b,w in edges_raw], dtype=torch.long)

history_loss = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # Forward pass — get all node embeddings
    if HAS_PYGEOM:
        embeddings = model(graph_data.x, graph_data.edge_index)
    else:
        embeddings = model(all_node_ids, adj_dict)

    # Positive scores — dot product of co-occurring tag pairs
    batch_size = min(2048, len(pos_src))
    idx = torch.randperm(len(pos_src))[:batch_size]
    p_src = pos_src[idx].to(DEVICE)
    p_dst = pos_dst[idx].to(DEVICE)

    pos_scores = (embeddings[p_src] * embeddings[p_dst]).sum(dim=1)

    # Negative sampling — random pairs
    neg_src = torch.randint(0, V, (batch_size,), device=DEVICE)
    neg_dst = torch.randint(0, V, (batch_size,), device=DEVICE)
    neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

    # BPR loss: log-sigmoid of (positive - negative)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    history_loss.append(loss.item())

    if epoch % 10 == 0:
        elapsed = time.time() - t0
        print(f'  Epoch {epoch:>3}/{EPOCHS}  loss={loss.item():.4f}  '
              f'elapsed={elapsed:.0f}s')

train_time = time.time() - t0
print(f'\nTraining complete in {train_time:.0f}s')

# ── Extract final embeddings ───────────────────────────────────────────
model.eval()
with torch.no_grad():
    if HAS_PYGEOM:
        tag_embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
    else:
        tag_embeddings = model(all_node_ids, adj_dict).cpu().numpy()

print(f'Tag embeddings shape: {tag_embeddings.shape}')

# ── Recommendation function ───────────────────────────────────────────
def recommend_tags(existing_tags, title_text='', n=TOP_N_RECOMMEND,
                   exclude_existing=True):
    """Recommend hashtags given a video's existing tags and title.

    Strategy:
      1. Get embeddings for existing tags
      2. Compute mean 'video embedding'
      3. Find top-N most similar tags in embedding space
    """
    known = [tag2idx[t] for t in existing_tags if t in tag2idx]
    if not known:
        # Cold start: return most popular tags
        return vocab[:n]

    video_emb = tag_embeddings[known].mean(axis=0, keepdims=True)
    sims      = cosine_similarity(video_emb, tag_embeddings)[0]

    if exclude_existing:
        for idx in known:
            sims[idx] = -1.0

    top_n = np.argsort(sims)[::-1][:n*3]
    recs  = [idx2tag[i] for i in top_n
             if idx2tag[i] not in existing_tags][:n]
    return recs

# ── Evaluation — Jaccard similarity ───────────────────────────────────
print('\nEvaluating Jaccard similarity...')

def jaccard(pred_set, true_set):
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set:  return 0.0
    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return inter / union

# Evaluate on videos with ≥2 tags: use first half as input, second as target
eval_df = videos_with_tags[videos_with_tags['tags'].apply(len) >= 4].copy()
eval_df = eval_df.sample(min(2000, len(eval_df)), random_state=42)

jaccard_gnn  = []
jaccard_tfidf_baseline = []  # BTP-1 approach: recommend by global frequency

for _, row in eval_df.iterrows():
    tags = row['tags']
    mid  = len(tags) // 2
    input_tags  = tags[:mid]
    target_tags = set(tags[mid:])

    # GNN recommendation
    gnn_recs = set(recommend_tags(input_tags, n=TOP_N_RECOMMEND))
    jaccard_gnn.append(jaccard(gnn_recs, target_tags))

    # Baseline: recommend globally most popular tags not already used
    baseline_recs = set(t for t in vocab if t not in input_tags)
    baseline_recs = set(list(baseline_recs)[:TOP_N_RECOMMEND])
    jaccard_tfidf_baseline.append(jaccard(baseline_recs, target_tags))

mean_jaccard_gnn      = np.mean(jaccard_gnn)
mean_jaccard_baseline = np.mean(jaccard_tfidf_baseline)
BTP1_JACCARD          = 0.460

print(f'\n{"═"*55}')
print(f'  HASHTAG RECOMMENDER RESULTS')
print(f'{"═"*55}')
print(f'  BTP-1 TF-IDF baseline   : {BTP1_JACCARD:.3f}')
print(f'  Popularity baseline      : {mean_jaccard_baseline:.3f}')
print(f'  BTP-2 GNN recommender    : {mean_jaccard_gnn:.3f}')
print(f'  vs BTP-1                 : {mean_jaccard_gnn - BTP1_JACCARD:+.3f}')

if mean_jaccard_gnn > BTP1_JACCARD:
    print(f'  ✅ Beats BTP-1 baseline')
else:
    print(f'  ⚠️  Below BTP-1 — see analysis below')

# ── Qualitative examples ───────────────────────────────────────────────
print(f'\n  Recommendation examples:')
sample_videos = eval_df.sample(5, random_state=7)
for _, row in sample_videos.iterrows():
    tags = row['tags']
    mid  = len(tags) // 2
    recs = recommend_tags(tags[:mid], n=5)
    print(f'\n  Video title : {str(row["title"])[:60]}')
    print(f'  Input tags  : {tags[:mid]}')
    print(f'  Actual tags : {tags[mid:]}')
    print(f'  GNN recommends: {recs}')

# ── Co-occurrence graph visualisation ─────────────────────────────────
print('\nGenerating co-occurrence graph visualisation...')
try:
    import networkx as nx
    top20 = vocab[:20]
    top20_idx = {t: tag2idx[t] for t in top20 if t in tag2idx}

    G = nx.Graph()
    G.add_nodes_from(top20)
    for (a,b), w in cooccur.items():
        ta, tb = idx2tag.get(a,''), idx2tag.get(b,'')
        if ta in top20_idx and tb in top20_idx and w >= 5:
            G.add_edge(ta, tb, weight=w)

    fig, ax = plt.subplots(figsize=(12, 10))
    pos    = nx.spring_layout(G, seed=42, k=2)
    sizes  = [tag_freq.get(t,1)*2 for t in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='#0D9488',
                           alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white',
                            font_weight='bold', ax=ax)
    edges  = G.edges(data=True)
    widths = [e[2]['weight']/20 for e in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4,
                           edge_color='#F59E0B', ax=ax)
    ax.set_title('Top-20 hashtag co-occurrence graph\n'
                 '(node size = frequency, edge width = co-occurrence strength)',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gnn_cooccurrence.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('  Saved → data/plots/gnn_cooccurrence.png')
except ImportError:
    print('  networkx not installed — skipping graph plot')

# Training curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(history_loss)+1), history_loss, color='#0D9488', linewidth=1.5)
ax.set_title('GNN Training Loss (Link Prediction)')
ax.set_xlabel('Epoch')
ax.set_ylabel('BPR Loss')
ax.axhline(history_loss[-1], color='#F59E0B', linestyle='--',
           label=f'Final loss={history_loss[-1]:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gnn_training.png', dpi=130, bbox_inches='tight')
plt.close()
print('  Saved → data/plots/gnn_training.png')

# ── Save model and results ─────────────────────────────────────────────
torch.save({
    'model_state': model.state_dict(),
    'tag2idx': tag2idx,
    'idx2tag': idx2tag,
    'tag_embeddings': tag_embeddings,
    'vocab': vocab,
    'embed_dim': EMBED_DIM,
    'has_pygeom': HAS_PYGEOM,
}, SAVE_DIR / 'gnn_model.pt')

results = {
    'model'              : 'GraphSAGE' if HAS_PYGEOM else 'LightGNN',
    'vocab_size'         : V,
    'n_edges'            : len(edges_raw),
    'embed_dim'          : EMBED_DIM,
    'epochs'             : EPOCHS,
    'train_time_sec'     : round(train_time, 1),
    'jaccard_gnn'        : round(mean_jaccard_gnn, 4),
    'jaccard_baseline'   : round(mean_jaccard_baseline, 4),
    'btp1_jaccard'       : BTP1_JACCARD,
    'vs_btp1'            : round(mean_jaccard_gnn - BTP1_JACCARD, 4),
    'top_n_recommend'    : TOP_N_RECOMMEND,
    'n_eval_videos'      : len(eval_df),
}
(MODELS_DIR / 'gnn_results.json').write_text(json.dumps(results, indent=2))
print(f'  Results saved → models/gnn_results.json')
print(f'  Model saved   → models/gnn_hashtag/gnn_model.pt')

print(f'\n{"═"*55}')
print(f'  GNN HASHTAG RECOMMENDER COMPLETE')
print(f'{"═"*55}')
print(f'  Vocab size     : {V:,} tags')
print(f'  Graph edges    : {len(edges_raw):,}')
print(f'  Jaccard score  : {mean_jaccard_gnn:.4f}')
print(f'  vs BTP-1       : {mean_jaccard_gnn - BTP1_JACCARD:+.4f}')
print(f'  Train time     : {train_time:.0f}s')
print(f'{"═"*55}')
print(f'\n  ALL 5 BTP-2 MODELS COMPLETE ✅')
print(f'  Next: python3 generate_report_numbers.py')
