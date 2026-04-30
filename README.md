# BTP-2: Multimodal YouTube Performance Prediction

**IIT Kharagpur · Bachelor's Thesis Project · 2025–26**

| | |
|---|---|
| **Student** | Vaidik Sharma (22MT10063) |
| **Supervisor** | Prof. Pabita Mitra |
| **Department** | School of Medical Science and Technology |
| **Live Dashboard** | [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app) |

---

## Overview

A complete machine learning pipeline that predicts YouTube video engagement performance (LOW / MID / HIGH) from structured metadata, text, and timing signals — without requiring video content or thumbnails.

**BTP-1 baseline → BTP-2 improvement:**

| Model | Accuracy | vs BTP-1 |
|---|---|---|
| BTP-1 Random Forest | 61.2% | baseline |
| BTP-2 RoBERTa | 64.1% | +2.9 pp |
| BTP-2 Ensemble | 69.0% | +7.8 pp |
| **BTP-2 XGBoost** | **69.1%** | **+7.9 pp** |
| Oracle ceiling | 78.2% | +17.0 pp |

---

## Dataset

- **20,308 YouTube videos** across 8 sectors
- **157 channels** scraped via YouTube Data API v3
- **75 features**: channel authority, text signals, timing, thumbnails, hashtags
- Sectors: Comedy, DIY, Education, Entertainment, Gaming, Lifestyle, Sci-Tech, Sports

---

## Dashboard Features

| Tab | Model | Description |
|---|---|---|
| 🔮 Predict & Analyze | XGBoost | Predict engagement tier + model-aware reasoning + improvement tips |
| 🏷️ Hashtag Lab | TaxonomyRecommender | Differentiated hashtag recommendations (28–92% confidence) |
| 🎯 Sector Analyzer | CatBoost (ablation) | Identify content sector from title/description only |
| 📊 Dataset EDA | — | 7 charts from 20,308-video corpus |
| ⏰ Timing | — | Sector-specific timing heatmaps + UTC→local converter |
| 🔬 Model Insights | All 5 models | Performance comparison + feature importance + real-world implications |

---

## Project Structure

```
btp2/
├── dashboard.py                  # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .streamlit/config.toml        # UI theme
│
├── models/
│   ├── xgboost_engagement.json   # XGBoost classifier (34 features)
│   ├── catboost_sector.cbm       # Sector classifier (99.1% accuracy)
│   ├── catboost_sector_nocatid.cbm
│   ├── hashtag_recommender.pkl   # TaxonomyRecommender
│   └── ensemble_results.json
│
├── data/
│   ├── btp2_v1_labeled.parquet   # Full dataset (20,308 videos)
│   ├── btp2_v2_features.parquet  # Engineered features
│   ├── dataset_meta.json         # Dataset statistics
│   └── feature_config.json       # XGBoost feature list
│
├── src/
│   ├── find_channel_ids_v3.py    # YouTube channel discovery
│   ├── eda.py                    # Exploratory data analysis
│   ├── feature_engineering_v2.py # Feature engineering pipeline
│   ├── train_xgboost.py          # XGBoost training
│   ├── train_roberta.py          # RoBERTa fine-tuning (MPS)
│   ├── train_gnn.py              # GNN hashtag model
│   ├── hashtag_recommender_v2.py # TaxonomyRecommender class
│   └── rebuild_recommender.py    # Rebuild pkl from taxonomy
│
└── docs/
    ├── BTP2_Report.docx          # Full 8-chapter thesis report
    └── BTP2_Final_Presentation.pptx
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/btp2-youtube-predictor.git
cd btp2-youtube-predictor

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for TextBlob)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Run dashboard
python3 -m streamlit run dashboard.py
```

---

## Key Findings

1. **Channel authority dominates** — `channel_total_videos`, `channel_subscribers`, `channel_age_days` account for 21.8% of XGBoost feature weight
2. **Zero hashtags wins** — videos with no hashtags achieve the highest mean engagement (3.636%), a counter-intuitive BTP-2 finding
3. **Short titles win** — word count has the strongest negative text correlation (r=−0.30)
4. **Sector is 96.5% predictable** from content features alone — no YouTube metadata needed
5. **MID class is fundamentally hard** — F1=0.57 for both XGBoost and RoBERTa; borderline videos require A/B testing

---

## Models

### XGBoost Engagement Classifier
- 34 structured features, 3 classes (LOW/MID/HIGH)
- 69.12% test accuracy, 2.3s training time on CPU
- Key features: channel authority group (21.8%), timing signals, hashtag count

### CatBoost Sector Classifier
- 99.07% accuracy with `youtube_category_id`
- **96.50% accuracy without** — sector fully predictable from content
- Key features: `youtube_category_id` (48.9%), `channel_age_days` (29.9%)

### RoBERTa Engagement Classifier
- `roberta-base` fine-tuned, 124.6M parameters
- 64.06% accuracy, trained for 3 epochs on Apple MPS (~40 min)
- Title + description concatenated, max 128 tokens

### Hashtag Recommender (TaxonomyRecommender)
- 52 curated topic clusters, 450+ hashtags
- Jaccard: 0.276 | Recall@5: 56.4%
- 5-signal scoring: title match, stem match, desc match, seed co-occurrence, sector frequency

---

## Technologies

`Python 3.9` · `XGBoost` · `CatBoost` · `HuggingFace Transformers` · `Streamlit` · `Plotly` · `Pandas` · `YouTube Data API v3` · `Apple MPS (M-series GPU)`

---

## Citation

```
Vaidik Sharma. "Multimodal YouTube Performance Prediction using
Structured Features and Language Models." BTP-2 Thesis,
IIT Kharagpur, 2026. Supervisor: Prof. Pabita Mitra.
```
