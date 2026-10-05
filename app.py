import streamlit as st
import pickle

model = pickle.load(open('npl_logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_bow.pkl', 'rb'))

# Emotion label mapping
emotion_mapping = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

st.title("NPL Sentiment Analysis")

text = st.text_area(
    'Enter your statement here:',
    placeholder='Tip : Use I based statements for better results.\nExample : I am really Happy today!'         
    )

if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter a statement.")

    else:
        # Convert text into Bag of Words representation
        text_vector = vectorizer.transform([text])

        # Predict emotion
        prediction = model.predict(text_vector)[0]

        # Get probabilities
        probabilities = model.predict_proba(text_vector)[0]

        # Convert numeric label to emotion
        emotion = emotion_mapping[prediction]

        # Get confidence
        confidence = probabilities[prediction] * 100

        st.success(f"Emotion: {emotion}")
        st.write(f"Confidence: {confidence:.2f}%")