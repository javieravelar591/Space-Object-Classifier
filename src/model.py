from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_pickle("data\processed\X_train.pkl")
y_train = pd.read_pickle("data\processed\Y_train.pkl")
X_test = pd.read_pickle("data\processed\X_test.pkl")
y_test = pd.read_pickle("data\processed\Y_test.pkl")

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

importances = clf.feature_importances_
features = X_train.columns
print(X_train.columns.tolist())
print(features)

feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(20))  # top 20 features
plt.title('Top Feature Importances')
plt.tight_layout()
plt.show()