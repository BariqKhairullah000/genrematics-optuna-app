# 🎬 GenreMatics: Multi-label Indonesian Movie Genre Classification

An AI-powered system that classifies Indonesian movie genres from synopses using **IndoBERT**, achieving **79.23% accuracy** in multi-label classification across five major genres. This project leverages **dynamic thresholding, per-class performance tracking, and augmentation techniques** to improve accuracy and generalization.

---

## 📌 Project Overview

**GenreMatics** is designed to automatically classify movie genres based on their synopses. Given a movie's synopsis, the model predicts one or more genres from five main categories:

- 🎭 **Drama**  
- 😂 **Comedy**  
- 👻 **Horror**  
- 🔥 **Action**  
- ❤️ **Romance**  

With this system, filmmakers, streaming platforms, and researchers can efficiently analyze and categorize movies based on their content.

---

## 🎯 Features

✅ **Multi-label genre classification** for Indonesian movies  
✅ **Optimized IndoBERT model** with **dynamic thresholding**  
✅ **Interactive web app** powered by **Streamlit**  
✅ **Fast inference time (~0.24s on average)**  
✅ **Comprehensive performance metrics & visualizations**  

---

## 📊 Dataset Overview

The dataset consists of **1,738 Indonesian movies**, each labeled with one or more genres. The distribution is as follows:

| Genre   | Count | Percentage |
|---------|--------|------------|
| Drama  | 510    | 29.3% |
| Comedy  | 374    | 21.5% |
| Horror  | 349    | 20.1% |
| Action  | 297    | 17.1% |
| Romance | 208    | 12.0% |

📂 **Dataset Access**: [Kaggle Dataset](https://www.kaggle.com/datasets/bariqkhairullah1/datasets-classificationsynopsis)

---

## 🚀 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 79.23% |
| **Macro F1-score** | 57.15% |
| **Macro Precision** | 50.07% |
| **Macro Recall** | 67.96% |

### **Per-Genre Performance**:

| Genre | Accuracy |
|--------|------------|
| 👻 Horror | **88.89%** (best performing) |
| ❤️ Romance | Significant improvement with optimization techniques |
| 😂 Comedy | Stable accuracy between **60-81%** |

---

## ⚙️ Technical Stack & Optimization

**🔹 Model:** IndoBERT-base-p1  
**🔹 Optimization Techniques:**
- ✅ **Dynamic Thresholding**
- ✅ **Per-class Performance Tracking**
- ✅ **Label Smoothing**
- ✅ **Mixup Augmentation**

**🔹 Deployment:** Interactive Streamlit web app  
**🔹 Inference Speed:** < 0.3s across all synopsis lengths  

---

## 💻 How to Run the Project

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/genrematics-optuna-app.git
cd genrematics-optuna-app
```

2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit App**  
```bash
streamlit run app.py
```

---

## 📌 Example Usage

You can test the model with the following Python script:

```python
from models.indobert_model import predict_genre

synopsis = "Seorang pemuda menemukan buku ajaib yang bisa mengubah masa depannya."
print(predict_genre(synopsis))

# Output: ['Fantasy', 'Drama']
```

Or by using the web app interface:
1. Enter a movie synopsis
2. Click **Predict**
3. View the predicted genres

---

## 📊 Model Architecture

Below is a high-level overview of the IndoBERT-based architecture:

```
[ Input: Movie Synopsis ] → [ IndoBERT Tokenizer ] → [ IndoBERT Model ] → [ Classification Head ] → [ Genre Predictions ]
```

This architecture ensures **context-aware text processing** and **efficient multi-label classification**.

---

## 🔗 Resources & Links

- 📂 **Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/bariqkhairullah1/datasets-classificationsynopsis)  
- 📄 **IndoBERT Paper**: [https://arxiv.org/abs/2009.05387](https://arxiv.org/abs/2009.05387)  
- 🚀 **Hugging Face IndoBERT**: [https://huggingface.co/indobert](https://huggingface.co/indobert)  

---

## 🏆 Future Improvements & Additional Features

💡 **Possible Enhancements**:
- 🎯 **Sub-genre Classification** → Expand labels to include **Thriller, Sci-Fi, Mystery**, etc.
- 📈 **Explainability** → Visualize model attention weights for better interpretability.
- 🎬 **Movie Recommendations** → Suggest similar movies based on synopsis similarity.
- 🌐 **API Integration** → Provide a **REST API** for classification and recommendations.

✨ **What’s Next?** Let’s push this model further and make movie genre classification smarter than ever! 🚀

