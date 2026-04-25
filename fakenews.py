import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("news.csv")

# Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df["text"])

y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test prediction
user_input = input("Enter news: ")
sample = [user_input]
sample_data = cv.transform(sample)

prediction = model.predict(sample_data)

if prediction[0] == "real":
    print("This news is REAL ✅")
else:
    print("This news is FAKE ❌")
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)