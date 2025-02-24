# 🎬 GenreMatics: Multi-label Indonesian Movie Genre Classification

An intelligent system for automatically classifying Indonesian movie genres from synopses using IndoBERT. This project achieves 79.23% accuracy in multi-label classification across five major genres.

## 🎯 Key Features

- Multi-label classification for 5 major Indonesian movie genres
- Dynamic Thresholding & Per-class Performance Tracking optimization
- Interactive web interface using Streamlit
- Fast inference time (~0.24s average)
- Comprehensive evaluation metrics and visualizations

## 📊 Dataset

The project uses a curated dataset of 1,738 Indonesian movies distributed across 5 genres:
- Drama: 510 films (29.3%)
- Comedy: 374 films (21.5%)
- Horror: 349 films (20.1%)
- Action: 297 films (17.1%)
- Romance: 208 films (12.0%)

[Access the complete dataset on Kaggle](https://www.kaggle.com/datasets/bariqkhairullah1/datasets-classificationsynopsis)

## 🚀 Performance

Best model configuration achieves:
- Accuracy: 79.23%
- Macro F1-score: 57.15%
- Macro Precision: 50.07%
- Macro Recall: 67.96%

### Genre-specific Performance Highlights:
- Horror: 88.89% accuracy (best performing)
- Romance: Significant improvement with optimization techniques
- Comedy: Stable performance across configurations (60-81% accuracy)

## 🛠 Technical Stack

- Base Model: IndoBERT-base-p1
- Optimization Techniques:
  - Dynamic Thresholding
  - Per-class Performance Tracking
  - Label Smoothing
  - Mixup Augmentation
- Web Interface: Streamlit
- Response Time: <0.3s for all synopsis lengths

## 🏃‍♂ How to Run

1. Clone the repository
bash
git clone https://github.com/yourusername/genrematics-optuna-app.git
cd genrematics-optuna-app


2. Install dependencies
bash
pip install -r requirements.txt


3. Run the Streamlit app
bash
streamlit run app.py


## 📁 Project Structure

genrematics/
├── data/                   # Dataset and preprocessing scripts
├── logs/                   # Training and evaluation logs
│   └── experiments/        # Experiment results
├── models/                 # Saved model checkpoints
├── app.py                  # Streamlit web interface
├── model.py               # Model architecture and training code
└── requirements.txt       # Project dependencies
