🧠 Student Mental Health Prediction (Depression)

This project utilizes machine learning (XGBoost & Random Forest) to predict student mental health conditions based on symptoms. It also integrates LLM-based suggestions for coping strategies based on prediction confidence.

📂 Dataset Preparation

Dataset Sources:
	•	Student Depression Dataset (Kaggle : https://www.kaggle.com/datasets/hopesb/student-depression-dataset?select=Student+Depression+Dataset.csv )
 

Preprocessing Steps:

✅ Data Cleaning: Used dropna() and fillna() to handle missing values.
✅ Feature Normalization: Applied Label Encoding to categorical features.
✅ Exploratory Data Analysis (EDA):
	•	Correlation analysis for numeric features.
	•	Heatmap to visualize feature relationships.
✅ Feature Selection: Used Random Forest Classifier to select top 5 most important features.
✅ Model Interpretability: Applied SHAP analysis only on selected features.

🤖 Model Development

Developed a multi-class classification model to predict mental health conditions.

Models Used:
	•	Random Forest Classifier 🌲
	•	XGBoost ⚡

Evaluation Metrics:

📊 Accuracy, Precision, Recall, F1-score, ROC-AUC
🔍 Explainability: SHAP for feature importance analysis.

Inference Script:

A trained model is saved, and an inference script (predict_mental_health.py) is provided to make predictions based on user inputs.

🧠 LLM-Based Coping Suggestions

After predicting a mental health condition, an LLM prompt suggests coping strategies based on the model’s confidence.

How It Works:

✅ If the model is highly confident, the LLM provides personalized coping mechanisms (exercise, therapy, self-help).
✅ If the model is uncertain, it suggests general mental health resources.

🎨 UI / CLI Testing

A Streamlit UI allows users to enter symptoms and get:
✅ Predicted Mental Health Condition
✅ Confidence Score
✅ LLM-based Coping Suggestions

🚀 How to Run

1️⃣ Fork the repsoitory and install requirements and run  streamlit

pip install -r requirements.txt

2️⃣ Launch UI (Streamlit) # Best

streamlit run app.py

I have already deplyed the model here: 
