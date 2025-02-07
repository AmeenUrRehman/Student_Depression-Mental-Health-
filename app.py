import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("student_depression_model.pkl", "rb"))

# Define UI
st.title("🧠 Depression and Anxiety Symptoms")
st.write("Enter your details below to check the likelihood of depression.")

# User Inputs
degree = st.selectbox("🎓 Degree", ["Undergraduate", "Graduate", "PhD"])
age = st.number_input("📅 Age", min_value=18, max_value=60, value=25)
academic_pressure = st.slider("📚 Academic Pressure (1-10)", min_value=1, max_value=10, value=5)
cgpa = st.number_input("📊 CGPA (0-4)", min_value=0.0, max_value=4.0, value=3.0)
suicidal_thoughts = st.radio("💭 Have you ever had suicidal thoughts?", ["No", "Yes"])

# Convert inputs for the model
degree_encoded = {"Undergraduate": 0, "Graduate": 1, "PhD": 2}[degree]
suicidal_encoded = 1 if suicidal_thoughts == "Yes" else 0

# Prepare input for prediction
features = np.array([[degree_encoded, age, academic_pressure, cgpa, suicidal_encoded]])

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None

# Predict depression
if st.button("🔍 Check Mental Health Condition"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of depression
    st.session_state.prediction = prediction
    st.session_state.probability = probability

# Show results if prediction was made
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.error(f"⚠️ You may be experiencing symptoms of depression. (Confidence: {st.session_state.probability:.2f})")
        st.write("💡 **Suggested Coping Mechanisms:**")
        st.write("- Seek professional help (therapy, counseling).")
        st.write("- Practice mindfulness and stress management techniques.")
        st.write("- Reach out to trusted friends or family members.")

        # Show "Find Cure" button if depressed
        if st.button("🛠️ Find Cure"):
            st.subheader("🧑‍⚕️ Recommended Steps for Recovery")
            st.write("Here are some actionable steps you can take to improve your mental well-being:")

            st.markdown("### 🏥 **Professional Help**")
            st.write("- Consider scheduling an appointment with a mental health professional.")
            st.write("- Therapy options include **CBT (Cognitive Behavioral Therapy)** and **talk therapy**.")
            st.write("- Explore online counseling services if in-person sessions aren't accessible.")

            st.markdown("### 🧘 **Self-Care & Daily Routine**")
            st.write("- Develop a **consistent sleep schedule** (7-9 hours per night).")
            st.write("- Engage in **daily physical activity** (walking, yoga, or exercise).")
            st.write("- Maintain a **balanced diet** with essential nutrients (avoid excessive caffeine/alcohol).")
            st.write("- Try relaxation techniques such as **meditation, deep breathing, or journaling**.")

            st.markdown("### 🎯 **Daily To-Do List for Better Mental Health**")
            st.write("✅ Start your day with a positive affirmation.")
            st.write("✅ Break tasks into smaller steps to avoid feeling overwhelmed.")
            st.write("✅ Spend at least **30 minutes outdoors** in natural light.")
            st.write("✅ Stay connected—talk to a friend, family member, or support group.")
            st.write("✅ Engage in a **hobby you enjoy** (music, reading, art, etc.).")

            st.markdown("### 💡 **Emergency Resources**")
            st.write("If you're feeling overwhelmed, please reach out for immediate help:")
            st.write("📞 **Suicide Prevention Helpline:** Call a local helpline in your country.")
            st.write("🌍 [Find Global Helplines Here](https://findahelpline.com/)")

    else:
        st.success("✅ You are not showing strong signs of depression. Stay mindful of your well-being!")
