
# 📌 Sentiment-Based Product Recommendation System  

## 📖 Project Overview  
This project implements an **end-to-end Sentiment-Based Product Recommendation System** for an e-commerce platform. The goal is to recommend products to users not only based on ratings and interactions but also by **analyzing the sentiment of product reviews**.  

The project covers the following:  
1. **Sentiment Analysis** on user reviews using ML models.  
2. **Collaborative Filtering-based Recommendation Systems** (User-based & Item-based).  
3. **Fine-tuning recommendations with sentiment scores** to output the **Top-5 best products** for a user.  
4. **Deployment of the system** as a Flask web application with a simple UI.  

---

## 🛠 Problem Statement  
You are working as a Machine Learning Engineer for an e-commerce company **Ebuss**, which competes with Amazon, Flipkart, and Snapdeal.  

The company requires a recommendation engine that not only suggests products but also considers **user sentiments** from product reviews to improve personalization.  

The tasks include:  
- Building a **sentiment analysis model** on reviews.  
- Designing **user-based and item-based collaborative filtering systems**.  
- Filtering the **top-5 product recommendations** using sentiment scores.  
- Deploying the solution using **Flask + Render/Heroku** with a web interface.  

---

## 📊 Dataset  
- **Source:** Subset inspired by a Kaggle competition.  
- **Size:** ~30,000 reviews from ~20,000 users across 200+ products.  
- **Attributes:** `reviews_username`, `reviews_rating`, `reviews_text`, `product_id`, etc.  
- **Supporting File:** `Product Dataset - Attributes.csv` (column descriptions).  

---

## 🚀 Workflow  

### **1. Data Cleaning & Preprocessing**
- Handle missing values.  
- Remove irrelevant attributes.  
- Convert datatypes.  
- Filter neutral reviews (rating = 3).  

### **2. Text Preprocessing**
- Lowercasing  
- Stopword removal  
- Tokenization & Lemmatization  
- Punctuation & special character removal  

### **3. Feature Extraction**
- Bag of Words (CountVectorizer)  
- TF-IDF Vectorizer  
- (Optional) Word Embeddings  

### **4. Sentiment Analysis**
- Build at least 3 models from:  
  - Logistic Regression  
  - Naive Bayes  
  - Random Forest  
  - XGBoost  
- Handle class imbalance (e.g., SMOTE).  
- Hyperparameter tuning.  
- Select **best-performing model** for deployment.  

### **5. Recommendation Systems**
- Build **User-based CF** and **Item-based CF**.  
- Evaluate performance (precision@k, recall@k, hit ratio).  
- Select **final recommendation model**.  

### **6. Sentiment-Aware Fine-Tuning**
- Recommend **Top-20 products** per user (CF).  
- Predict sentiment for reviews of these products.  
- Compute **% of positive reviews per product**.  
- Output **Top-5 recommendations** with highest sentiment scores.  

### **7. Deployment**
- **Flask app** with input box for `username`.  
- On submit → return **Top-5 products**.  
- Frontend: `index.html`  
- Backend: `app.py` + `model.py`  
- Deployment: **Render** (Heroku alternative).  

---

## 📂 Project Structure  

```
Sentiment_Based_Product_Recommendation_System/
│── data/                     # Raw & processed datasets
│── notebooks/                # Jupyter notebooks for EDA & modeling
│── artifacts/                # Saved models, vectorizers, mappings
│   └── *.pkl                 # Serialized ML models
│── app.py                    # Flask application
│── model.py                  # Deployment model logic
│── index.html                # Frontend UI
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---

## ✅ Evaluation Rubric (10% each, unless mentioned)  
1. Data cleaning & preprocessing ✔️  
2. Text preprocessing ✔️  
3. Feature extraction ✔️  
4. Model building (3+ ML models, select 1) **20%**  
5. Recommendation systems (User & Item-based) **20%**  
6. Recommend Top-20 products ✔️  
7. Fine-tuning to Top-5 using sentiment ✔️  
8. Deployment using Flask ✔️  

---

## 📦 Installation & Usage  

### 1. Clone Repository  
```bash
git clone https://github.com/blizwing/Sentiment_Based_Product_Recommendation_System.git
cd Sentiment_Based_Product_Recommendation_System
```

### 2. Install Requirements  
```bash
pip install -r requirements.txt
```

### 3. Run Flask App  
```bash
python app.py
```
Then open `http://127.0.0.1:5500/` in your browser.  

### 4. Deployment  
- Supported on **Heroku (paid)** or **Render (free alternative)**.  
- Ensure model `.pkl` files are included in deployment package.  

---

## 🎯 Expected Output  
- User enters a `username`.  
- System returns **Top-5 recommended products** with highest sentiment scores.  

---

## 🔮 Future Work  
- Extend to new users/products using hybrid approaches.  
- Incorporate **deep learning sentiment models** (e.g., LSTM, BERT).  
- Use **implicit feedback** (clicks, dwell time).  
- Real-time recommendations.  
