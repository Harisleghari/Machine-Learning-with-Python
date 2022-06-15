import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
Y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, Y)

joblib.dump(model, "music_recommandor.joblib")
# predictions = model.predict()