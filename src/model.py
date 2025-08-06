from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_loader import sc, le

X_train = pd.read_pickle("data\processed\X_train.pkl")
y_train = pd.read_pickle("data\processed\Y_train.pkl")
X_test = pd.read_pickle("data\processed\X_test.pkl")
y_test = pd.read_pickle("data\processed\Y_test.pkl")

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = clf.feature_importances_
features = X_train.columns

feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# joblib.dump(clf, "model/random_forest _model.pkl")
joblib.dump(sc, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")

# plt.figure(figsize=(10,6))
# sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(20))  # top 20 features
# plt.title('Top Feature Importances')
# plt.tight_layout()
# plt.show()