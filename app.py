from flask import Flask, render_template, request
import re
import io
import pandas as pd
import spacy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from docx import Document
from PyPDF2 import PdfReader

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# VADER sentiment
sent_analyzer = SentimentIntensityAnalyzer()

# BERT sentiment pipeline
bert_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_csv(file):
    content = file.read()
    df = pd.read_csv(io.BytesIO(content))
    return df


def sentiment_vader(text: str):
    scores = sent_analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, scores


def sentiment_bert(text: str):
    short_text = text[:512]
    result = bert_sentiment(short_text)[0]
    label = result["label"]
    score = result["score"]

    if label.upper() == "POSITIVE":
        final_label = "Positive"
    elif label.upper() == "NEGATIVE":
        final_label = "Negative"
    else:
        final_label = label.title()

    return final_label, round(score, 4)


def extract_keywords_tfidf(text: str, top_k: int = 12):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    tfidf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    scored_terms = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )

    keywords = [term for term, score in scored_terms if score > 0][:top_k]
    return keywords


def extract_entities(text: str, max_per_label: int = 10):
    limited_text = text[:20000]
    doc = nlp(limited_text)
    entities_by_label = {}

    for ent in doc.ents:
        label = ent.label_
        value = ent.text.strip()

        if not value:
            continue

        entities_by_label.setdefault(label, [])
        if value not in entities_by_label[label]:
            entities_by_label[label].append(value)

    for label in list(entities_by_label.keys()):
        entities_by_label[label] = entities_by_label[label][:max_per_label]

    return entities_by_label


def summarize_extractive(text: str, max_sentences: int = 3):
    limited_text = text[:20000]
    doc = nlp(limited_text)
    sents = [s for s in doc.sents if len(s.text.strip()) > 30]

    if not sents:
        return limited_text[:300] + ("..." if len(limited_text) > 300 else "")

    words = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    ]

    if not words:
        return " ".join([s.text.strip() for s in sents[:max_sentences]])

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sent_scores = []
    for s in sents:
        score = 0
        for t in s:
            if t.is_alpha:
                score += freq.get(t.lemma_.lower(), 0)
        sent_scores.append((s.text.strip(), score))

    top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_texts = {t[0] for t in top}
    ordered = [s.text.strip() for s in sents if s.text.strip() in top_texts]

    return " ".join(ordered)


def get_csv_text_column(df):
    preferred_columns = [
        "cleaned_review_text",
        "review_text",
        "text",
        "review",
        "comment",
        "comments",
        "description"
    ]

    for col in preferred_columns:
        if col in df.columns:
            return col

    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if object_columns:
        return object_columns[0]

    return None


def analyze_csv_dataset(df):
    text_column = get_csv_text_column(df)

    if not text_column:
        return {
            "error": "No text column found in the uploaded CSV."
        }

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    texts = texts[texts != ""]

    if len(texts) == 0:
        return {
            "error": "The CSV does not contain enough readable text to analyze."
        }

    # VADER over all rows
    vader_labels = []
    for text in texts:
        label, _ = sentiment_vader(text)
        vader_labels.append(label)

    vader_counts = Counter(vader_labels)

    # BERT over a sample of rows to keep app responsive
    bert_labels = []
    bert_sample = texts.head(300)

    for text in bert_sample:
        label, _ = sentiment_bert(text[:512])
        bert_labels.append(label)

    bert_counts = Counter(bert_labels)

    # NER in chunks over a subset
    entity_counter = {}
    entity_label_map = {}

    nlp_subset = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    batch_texts = texts.head(500).tolist()

    for doc in nlp_subset.pipe(batch_texts, batch_size=32):
        for ent in doc.ents:
            value = ent.text.strip()
            label = ent.label_

            if value:
                entity_counter[value] = entity_counter.get(value, 0) + 1
                entity_label_map[value] = label

    entities_by_label = {}
    sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)

    for value, count in sorted_entities[:40]:
        label = entity_label_map[value]
        entities_by_label.setdefault(label, [])
        entities_by_label[label].append(f"{value} ({count})")

    # Keywords from a large safe combined sample
    full_text = " ".join(texts.tolist())
    keywords = extract_keywords_tfidf(full_text[:200000], top_k=15)

    # Summary from a large safe sample
    summary_source = " ".join(texts.head(200).tolist())
    summary = summarize_extractive(summary_source[:30000])

    # Preview of original text
    preview_text = "\n".join(texts.head(20).tolist())

    total_rows = len(df)
    analyzed_rows = len(texts)

    vader_majority = max(vader_counts, key=vader_counts.get) if vader_counts else "N/A"
    bert_majority = max(bert_counts, key=bert_counts.get) if bert_counts else "N/A"

    total_vader = sum(vader_counts.values()) if vader_counts else 1
    positive_ratio = vader_counts.get("Positive", 0) / total_vader
    neutral_ratio = vader_counts.get("Neutral", 0) / total_vader
    negative_ratio = vader_counts.get("Negative", 0) / total_vader

    sentiment_scores = {
        "pos": positive_ratio,
        "neu": neutral_ratio,
        "neg": negative_ratio,
        "compound": 0.0
    }

    return {
        "text": preview_text,
        "sentiment_label": vader_majority,
        "sentiment_scores": sentiment_scores,
        "bert_label": bert_majority,
        "bert_score": f"Sampled on {len(bert_sample)} rows",
        "keywords": keywords,
        "entities": entities_by_label,
        "summary": summary,
        "text_column": text_column,
        "total_rows": total_rows,
        "analyzed_rows": analyzed_rows,
        "is_csv_analysis": True
    }


def analyze_regular_text(text: str):
    text = clean_text(text)

    if len(text) < 20:
        return {
            "error": "The uploaded file does not contain enough readable text to analyze."
        }

    analysis_text = text[:30000]
    bert_text = text[:512]

    vader_label, vader_scores = sentiment_vader(analysis_text)
    bert_label, bert_score = sentiment_bert(bert_text)
    keywords = extract_keywords_tfidf(analysis_text)
    entities = extract_entities(analysis_text)
    summary = summarize_extractive(analysis_text)

    return {
        "text": text[:5000],
        "sentiment_label": vader_label,
        "sentiment_scores": vader_scores,
        "bert_label": bert_label,
        "bert_score": bert_score,
        "keywords": keywords,
        "entities": entities,
        "summary": summary,
        "is_csv_analysis": False
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "results.html",
            error="Please upload a TXT, PDF, DOCX, or CSV file."
        )

    filename = uploaded_file.filename.lower()

    try:
        if filename.endswith(".txt"):
            result = analyze_regular_text(extract_text_from_txt(uploaded_file))

        elif filename.endswith(".pdf"):
            result = analyze_regular_text(extract_text_from_pdf(uploaded_file))

        elif filename.endswith(".docx"):
            result = analyze_regular_text(extract_text_from_docx(uploaded_file))

        elif filename.endswith(".csv"):
            df = extract_text_from_csv(uploaded_file)
            result = analyze_csv_dataset(df)

        else:
            return render_template(
                "results.html",
                error="Unsupported file type. Please upload a TXT, PDF, DOCX, or CSV file."
            )

        if "error" in result:
            return render_template("results.html", error=result["error"])

        return render_template("results.html", **result)

    except Exception as e:
        return render_template(
            "results.html",
            error=f"Could not read the uploaded file: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)