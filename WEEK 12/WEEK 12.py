import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------- 1. LOAD ----------
try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    df = pd.read_csv('../Iris.csv')

measurements = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[measurements + ['Species']]
df = df.drop_duplicates()

# ---------- 2. FEATURES (X) AND TARGET (y) ----------
X = df[measurements]     # the measurements the model learns from
y = df['Species']        # the species we want to predict

# ---------- 3. TRAIN / TEST SPLIT ----------
# 80% to train the model, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- 4. TRAIN THE DECISION TREE ----------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------- 5. PREDICT ----------
y_pred = model.predict(X_test)

# ---------- 6. EVALUATE ----------
print('Accuracy :', round(accuracy_score(y_test, y_pred), 4))
print('Precision:', round(precision_score(y_test, y_pred, average='macro'), 4))
print('Recall   :', round(recall_score(y_test, y_pred, average='macro'), 4))
