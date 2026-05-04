
# Customer Feedback Analytics using NLP.

This project presents an end-to-end system for analyzing customer feedback using Natural Language Processing (NLP) and visualizing insights through Power BI dashboards. The system processes both structured and unstructured data to extract meaningful insights such as sentiment, keywords, entities, and summaries.

## Project Overview

Customer feedback data is often large and unstructured, making manual analysis inefficient. This project automates the process by:

- Cleaning and preprocessing datasets using Python
- Applying NLP techniques (VADER, BERT, TF-IDF, spaCy)
- Generating summaries and extracting key insights
- Visualizing results in an interactive Power BI dashboard

## Data Sources

The project uses Amazon-based datasets:
- **Product Summary Data**
- **Final Review Data**
These datasets are merged and validated to create a comprehensive dataset for analysis.

## Data Cleaning & Preprocessing

Performed using **Python (pandas)**:
- Merged multiple datasets
- Removed null and duplicate values
- Standardized text data
- Selected relevant columns (e.g., review_text)
Output: **Cleaned dataset ready for NLP processing**

## NLP Techniques Used

### 🔹 Sentiment Analysis
- **VADER** (fast, rule-based)
- **BERT (DistilBERT)** (context-aware deep learning model)

### 🔹 Keyword Extraction
- **TF-IDF** to identify important words and phrases

### 🔹 Named Entity Recognition (NER)
- Using **spaCy** to extract:
  - Organizations
  - Locations
  - Dates
  - Numbers

### 🔹 Text Summarization
- Extractive summarization using sentence scoring

## 🌐 Web Application

Built using **Flask**:
- Upload files (TXT, PDF, DOCX, CSV)
- Perform real-time NLP analysis
- Display:
  - Sentiment (VADER + BERT)
  - Summary
  - Keywords
  - Named Entities
## 📊 Power BI Dashboard

The processed data is visualized using Power BI to provide business insights:

### Key Features:
- Sentiment distribution
- Brand risk analysis
- Hidden dissatisfaction score
- Product performance tracking
- Negative rate distribution
- Sentiment vs rating correlation
- 
## 🔄 Workflow

1. Data Collection (Amazon datasets)  
2. Data Validation  
3. Data Cleaning (pandas)  
4. NLP Processing (VADER, BERT, TF-IDF, spaCy)  
5. Data Aggregation  
6. Visualization using Power BI  

## ▶️ How to Run the Project


## 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/Customer-Feedback-analytics-using-NLP.git
cd Customer-Feedback-analytics-using-NLP
```
## 2. Create virtual environment
```bash 
python -m venv venv
source venv/bin/activate
```
## 3. Install dependencies
``` bash
pip install -r requirements.txt
```
## 4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```
## 5. Run the application
```bash
python app.py
```
## 6. Open in browser
``` bash
http://127.0.0.1:5000
```
