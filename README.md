# Zomato Restaurant Clustering & Sentiment Analysis: Customer Insights and Restaurant Segmentation

![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Notebook: Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)
![ML: scikit-learn](https://img.shields.io/badge/ML-scikit--learn-green.svg)

## 🏢 Project Overview
This repository contains an end-to-end machine learning project on **Zomato**, India's restaurant aggregator and food delivery platform. Using two real datasets from **Hyderabad** (105 restaurants and 10,000 customer reviews), the project reads customer opinions with a **sentiment analysis model**, groups restaurants into business segments with **unsupervised clustering**, and turns both results into clear actions for Zomato.

The project covers the full data science workflow: data cleaning, feature engineering, 16 UBM charts, hypothesis testing, NLP, supervised and unsupervised modelling, cross-validation, hyperparameter tuning, explainability and model saving.

## 🚀 Key Business Findings
*   **Sentiment Model:** Logistic Regression reached **94.8% accuracy** and a **macro F1 of 0.94** on unseen reviews, finding about **95 out of every 100 negative reviews** (recall 0.95).
*   **Price and Satisfaction:** Expensive restaurants are rated higher than cheaper ones (**3.79 vs 3.43**, p = 0.0003). In the High price group **73%** of reviews are positive, against **56%** in the Low group.
*   **Pictures Matter:** Reviews with pictures are rated higher (**3.86 vs 3.54**, p = 0.003), so engagement and satisfaction go together.
*   **Critics:** No evidence that critics are harsher than regular reviewers (**3.79 vs 3.60**, p = 0.23).
*   **Three Restaurant Segments (K-Means, k = 3):**

| Segment | Restaurants | Average Cost | Average Rating | Suggested Action |
|---|---|---|---|---|
| Budget Bites | 45 | about Rs 520 | 3.37 | Quality support and menu improvement |
| Social Buzz | 29 | about Rs 914 | 3.66 | More visibility through collections and influencer events |
| Premium Favourites | 26 | about Rs 1,404 | 3.94 | Reputation protection and loyalty offers |

## 🛠️ Methodology & Statistics
*   **Data Cleaning:** Cost converted from text to number, one unreadable rating and 38 empty rows removed (10,000 to **9,961 reviews**), reviewer metadata split into reviews and followers, **269 critic reviews** flagged (100+ reviews and 1,000+ followers).
*   **Feature Engineering:** One row per restaurant, price groups (Low, Medium, High), cuisine and collection counts, value for money, IQR outlier capping, log transform and standard scaling.
*   **EDA:** **16 charts** following the UBM rule (Univariate, Bivariate, Multivariate). Every chart explains why it was chosen, the insight, the positive business impact and the risk of negative growth.
*   **Hypothesis Testing:** Three tests at the 5% level using **Welch's t-test** and **Mann-Whitney U**.
*   **NLP:** Contraction expansion, lower casing, punctuation and stopword removal (negations kept), stemming and **TF-IDF** (3,000 features with 1-2 word phrases).
*   **Sentiment Model:** **Logistic Regression** trained only on 1-2 star (negative) and 4-5 star (positive) reviews. 3-star reviews are not used for training. They get a probability score afterwards, and scores between **0.40 and 0.60** are called Neutral. Tuned with **5-fold stratified cross-validated grid search** (best C = 3).
*   **Clustering:** **K-Means, Hierarchical and Gaussian Mixture** models, each tuned and compared with silhouette and Davies-Bouldin scores. K-Means gave balanced groups (45 / 29 / 26) and was chosen as the final model.

| Model | Silhouette (after tuning) | Note |
|---|---|---|
| K-Means (k = 3) | 0.253 | Balanced, easy to explain, can place new restaurants |
| Hierarchical | about 0.255 | Very unbalanced groups (49 / 43 / 8) |
| Gaussian Mixture | 0.253 | Finds exactly the same groups as K-Means |

*   **Explainability:** Random Forest feature importance (pictures 0.18, cost 0.15, collections 0.15) shows that the segments are defined by price, reviewer engagement and visibility.
*   **Deployment Ready:** Final model saved and reloaded with **joblib**, then tested on a new restaurant.

## 📂 Repository Structure
```text
├── Zomato_Restaurant_Clustering.ipynb       # Complete analysis workflow with outputs
├── Zomato Restaurant names and Metadata.csv # Restaurant dataset (105 restaurants)
├── Zomato Restaurant reviews.csv            # Review dataset (10,000 reviews)
├── Zomato project.pptx                      # Project brief and architecture
└── README.md                                # Project documentation
```

## 💻 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/aadilchavhan/zomato-restaurant-clustering.git
   cd zomato-restaurant-clustering
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib seaborn joblib jupyter
   ```
   `nltk` is optional (used only for a part-of-speech demo).
3. **Open the notebook** (keep both CSV files in the same folder):
   ```bash
   jupyter notebook Zomato_Restaurant_Clustering.ipynb
   ```
4. Choose **Kernel > Restart & Run All**. It also runs in Google Colab: upload the CSV files first, then Runtime > Restart and run all.

## 🛡️ Limitations & Threats to Validity
*   **Observational Data:** The hypothesis tests show association, not cause. Expensive restaurants may differ from cheaper ones in many ways.
*   **Moderate Cluster Separation:** Silhouette is about 0.25, so segments overlap a little. They are broad groups for strategy, not strict boxes.
*   **3-Star Reviews:** The model never learns from them, and only about 18% fall in the neutral band, so they should be read together with their star rating.
*   **Small and Local Sample:** One city, one platform and only 100 restaurants are clustered. Five restaurants without reviews were left out.
*   **Reviewer Bias:** People who write reviews are often very happy or very unhappy, so results may not describe silent customers.

---
*Created as part of a Data Science Internship project.*
