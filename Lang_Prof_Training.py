from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Assume you have collected and preprocessed your dataset
# X: list of preprocessed text samples
# y: list of corresponding proficiency level labels (encoded as integers, e.g., A1=0, A2=1, ..., C2=5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract features using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a support vector classifier (SVC) model
clf = SVC(kernel='linear')
clf.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Integrate the trained model into the assessment function
def assess_grammar(text):
    text_tfidf = vectorizer.transform([text])
    grammar_score = clf.predict_proba(text_tfidf)[:, -1]  # Assuming higher class labels indicate better proficiency
    return grammar_score
