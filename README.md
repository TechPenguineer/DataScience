# Data Science Portfolio

A comprehensive collection of data science and machine learning projects demonstrating various techniques in exploratory data analysis, predictive modeling, natural language processing, and deep learning.

## 📊 Projects Overview

### Exploratory Data Analysis

- **[2016 General Election Poll Analysis](2016%20General%20Election%20Poll%20Analysis.ipynb)** - Analysis of opinion poll data from the 2016 US General Election, examining voter sentiment changes over time and debate effects
- **[911 Calls Exploratory Analysis](911%20Calls%20-%20Exploratory%20Analysis.ipynb)** - Comprehensive analysis of emergency call patterns using the Montgomery County 911 dataset from Kaggle
- **[Stock Market Analysis for Tech Stocks](Stock%20Market%20Analysis%20for%20Tech%20Stocks.ipynb)** - Financial analysis of technology stocks including price movements, daily returns, correlations, and risk assessment
- **[Titanic Dataset Exploratory Analysis](Titanic%20Dataset%20-%20Exploratory%20Analysis.ipynb)** - Classic exploratory analysis of the Titanic dataset examining passenger demographics and survival factors

### Machine Learning Projects

#### Supervised Learning

- **[Boston Housing Price Prediction](boston_housing/)** - Regression model to predict housing prices using the Boston Housing dataset with feature engineering and model evaluation
- **[Customer Segmentation](customer_segments/)** - Unsupervised learning project using clustering techniques to identify customer segments for a wholesale distributor

#### ML Micro Projects Collection

A series of focused machine learning implementations demonstrating various algorithms:

- **[Linear Regression](ML%20Micro%20Projects/Machine%20Learning%20with%20Linear%20Regression.ipynb)** - E-commerce company analysis to determine mobile app vs website focus
- **[Logistic Regression](ML%20Micro%20Projects/Machine%20Learning%20with%20Logistic%20Regression.ipynb)** - Binary classification for predicting ad click-through rates
- **[K-Nearest Neighbors](ML%20Micro%20Projects/ML%20with%20K%20Nearest%20Neighbors.ipynb)** - Classification using KNN algorithm on synthetic datasets
- **[Support Vector Machines](ML%20Micro%20Projects/ML%20with%20Support%20Vector%20Machines.ipynb)** - Iris flower classification with parameter tuning
- **[Decision Trees & Random Forests](ML%20Micro%20Projects/Machine%20Learning%20with%20Decision%20Trees%20and%20Random%20Forests.ipynb)** - Loan default prediction using LendingClub data
- **[Recommender Systems](ML%20Micro%20Projects/Recommender%20Systems%20with%20Python.ipynb)** - Movie recommendation system using collaborative filtering

### Natural Language Processing

- **[3-Way Sentiment Analysis for Tweets](3-Way%20Sentiment%20Analysis%20for%20Tweets.ipynb)** - Custom sentiment classifier for tweets using logistic regression, bag-of-words features, and polarity lexicons
- **[Cross Language Information Retrieval](Cross%20Language%20Information%20Retrieval.ipynb)** - CLIR system for German-to-English document search using machine translation and vector space models

### Deep Learning

- **[MNIST Digit Recognition](digit_recognition-mnist-sequence.ipynb)** - Deep learning model using Keras to recognize sequences of handwritten digits

## 🛠 Technologies Used

- **Python Libraries**: pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Machine Learning**: Supervised/unsupervised learning, classification, regression, clustering
- **Deep Learning**: Keras, TensorFlow
- **NLP**: NLTK, custom preprocessing pipelines
- **Data Visualization**: matplotlib, seaborn
- **Data Sources**: Kaggle, UCI ML Repository, real-world datasets

## 📁 Project Structure

```
├── ML Micro Projects/          # Collection of focused ML algorithm implementations
├── boston_housing/            # Housing price prediction project
├── customer_segments/         # Customer segmentation analysis
├── data/                      # Datasets and supporting files
│   ├── 911.csv
│   ├── advertising.csv
│   └── clir/                  # Cross-language IR corpus files
└── *.ipynb                    # Individual project notebooks
```

## 🚀 Getting Started

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter nltk keras tensorflow
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open any project notebook to explore the analysis

## 📈 Key Skills Demonstrated

- **Data Preprocessing**: Cleaning, feature engineering, handling missing values
- **Exploratory Data Analysis**: Statistical analysis, data visualization, pattern recognition
- **Machine Learning**: Model selection, hyperparameter tuning, cross-validation, performance evaluation
- **Deep Learning**: Neural network architecture design, sequence modeling
- **Natural Language Processing**: Text preprocessing, sentiment analysis, machine translation
- **Statistical Analysis**: Hypothesis testing, correlation analysis, time series analysis

## 📊 Datasets

Projects utilize diverse datasets including:

- Financial market data (Yahoo Finance)
- Government datasets (911 calls, election polls)
- Academic datasets (UCI ML Repository)
- Social media data (Twitter)
- E-commerce and business data
- Historical datasets (Titanic)

Each project includes detailed documentation of data sources, preprocessing steps, and methodology.
