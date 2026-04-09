import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset
diabetes = datasets.load_diabetes()

# convert to dataframe
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["Target"] = diabetes.target

# convert regression target to classification
df["Diabetes"] = df["Target"].apply(
    lambda x: 1 if x > df["Target"].mean() else 0
)

# split data
X = df[diabetes.feature_names]
y = df["Diabetes"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
