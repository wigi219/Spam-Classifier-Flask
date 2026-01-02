# import library
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# load dataset
df = pd.read_csv("C:\Users\LABKOM\Videos\jumat pagi\dataset\spam.csv")
X = df["Message"]
y = df["Category"]
# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# pipeline TF-IDF +SVM (tidak perlu preprocessing manual)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ("svm", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])
# training model
pipeline.fit(X_train, y_train)
# evaluasi model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
# simpan ke svm_model.pkl
with open("svm_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("svm_model.pkl berhasil dibuat")