#!/usr/bin/env bash
# =============================================================================
# setup_github.sh — BTP-2 GitHub Upload Script
# Run from inside ~/Downloads/btp2_scraper/
# Usage: bash setup_github.sh YOUR_GITHUB_USERNAME
# =============================================================================
set -e

USERNAME=${1:-"YOUR_GITHUB_USERNAME"}
REPO="btp2-youtube-predictor"
echo "Setting up GitHub repo: $USERNAME/$REPO"

# ── Step 0: Check git is installed ──────────────────────────────────────
if ! command -v git &>/dev/null; then
    echo "Installing git via Homebrew..."
    brew install git
fi

# ── Step 1: Install Git LFS (for large files) ───────────────────────────
echo ""
echo "Installing Git LFS..."
brew install git-lfs
git lfs install

# ── Step 2: Create directory structure ──────────────────────────────────
echo ""
echo "Creating directory structure..."
mkdir -p src docs .streamlit

# Move scripts to src/
for f in find_channel_ids_v3.py eda.py feature_engineering_v2.py \
          train_xgboost.py train_roberta.py train_gnn.py \
          hashtag_recommender_v2.py rebuild_recommender.py \
          test_hashtag_model.py transcript_extractor.py \
          hashtag_recommender.py seed_channels_v3.csv; do
    [ -f "$f" ] && mv "$f" src/ && echo "  moved $f → src/"
done

# Move docs
for f in BTP2_Report.docx BTP2_Final_Presentation_Template.pptx; do
    [ -f "$f" ] && mv "$f" docs/ && echo "  moved $f → docs/"
done

# ── Step 3: Write .gitattributes for Git LFS ────────────────────────────
echo ""
echo "Writing .gitattributes for Git LFS..."
cat > .gitattributes << 'EOF'
# Large model files → Git LFS
models/*.cbm filter=lfs diff=lfs merge=lfs -text
models/*.pt filter=lfs diff=lfs merge=lfs -text
models/*.bin filter=lfs diff=lfs merge=lfs -text
models/roberta_engagement/** filter=lfs diff=lfs merge=lfs -text

# Large data files → Git LFS
data/*.parquet filter=lfs diff=lfs merge=lfs -text
data/thumbnails/** filter=lfs diff=lfs merge=lfs -text

# Docs → Git LFS
docs/*.docx filter=lfs diff=lfs merge=lfs -text
docs/*.pptx filter=lfs diff=lfs merge=lfs -text
EOF
echo "  .gitattributes written"

# ── Step 4: Initialise git repo ─────────────────────────────────────────
echo ""
echo "Initialising git..."
git init
git checkout -b main 2>/dev/null || git checkout main

# ── Step 5: Add all files ───────────────────────────────────────────────
echo ""
echo "Staging files..."
git add .gitattributes .gitignore requirements.txt README.md
git add dashboard.py hashtag_recommender_v2.py rebuild_recommender.py 2>/dev/null || true
git add .streamlit/config.toml
git add models/ data/ src/ docs/ 2>/dev/null || true
git status --short

# ── Step 6: Initial commit ──────────────────────────────────────────────
echo ""
echo "Creating initial commit..."
git commit -m "BTP-2: YouTube Performance Predictor — initial release

Models: XGBoost 69.1%, CatBoost 99.1%, RoBERTa 64.1%, Ensemble 69.0%
Dataset: 20,308 videos, 157 channels, 75 features, 8 sectors
Dashboard: 6-tab Streamlit app with model-aware reasoning
Supervisor: Prof. Pabita Mitra | IIT Kharagpur | Vaidik Sharma 22MT10063"

# ── Step 7: Connect to GitHub ───────────────────────────────────────────
echo ""
echo "Connecting to GitHub..."
echo ""
echo "You need to create the repo on GitHub first:"
echo "  → https://github.com/new"
echo "  → Repository name: $REPO"
echo "  → Visibility: Public (required for free Streamlit Cloud deployment)"
echo "  → Do NOT add README/gitignore (we already have them)"
echo ""
read -p "Press Enter once you've created the repo on GitHub..."

git remote add origin "https://github.com/$USERNAME/$REPO.git" 2>/dev/null || \
git remote set-url origin "https://github.com/$USERNAME/$REPO.git"

echo ""
echo "Pushing to GitHub (this may take a few minutes due to LFS files)..."
git push -u origin main

echo ""
echo "✅ Upload complete!"
echo "   Repository: https://github.com/$USERNAME/$REPO"
echo ""
echo "Next: Deploy to Streamlit Cloud"
echo "  → https://share.streamlit.io"
echo "  → Click 'New app'"
echo "  → Repository: $USERNAME/$REPO"
echo "  → Branch: main"
echo "  → Main file: dashboard.py"
echo "  → Click Deploy!"
