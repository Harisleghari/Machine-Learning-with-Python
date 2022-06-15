model = joblib.load("music_recommandor.joblib")
predictions = model.predict([[21, 1]])
predictions