import streamlit as st
import pickle

st.set_page_config(
    page_title="EmotionAI",
    page_icon="😊",
    layout="centered"
)

model = pickle.load(open("npl_logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_bow.pkl", "rb"))

emotion_mapping = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

emotion_emoji = {
    "Sadness": "😢",
    "Anger": "😡",
    "Love": "❤️",
    "Surprise": "😲",
    "Fear": "😨",
    "Joy": "😊"
}

st.title("EmotionAI")

st.markdown(
    "### Text Emotion Classifier"
)

st.write(
    "Enter a statement and let the machine learning model "
    "predict the emotion behind it."
)

st.divider()

text = st.text_area(
    "Enter your statement",
    placeholder="Example: I finally got the internship I wanted!",
    height=120
)

if st.button("Analyze Emotion", use_container_width=True):

    if not text.strip():

        st.warning("Please enter a statement first.")

    else:

        # Convert text to Bag of Words
        text_vector = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Probability of every emotion
        probabilities = model.predict_proba(text_vector)[0]

        # Get emotion name
        emotion = emotion_mapping[prediction]

        # Confidence
        confidence = probabilities[prediction] * 100

        st.divider()

        st.subheader("Prediction")

        st.markdown(
            f"# {emotion_emoji[emotion]} {emotion}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.divider()

        st.subheader("Emotion Probabilities")

        for index, probability in enumerate(probabilities):

            emotion_name = emotion_mapping[index]
            percentage = probability * 100

            st.write(
                f"**{emotion_name}** — {percentage:.2f}%"
            )

            st.progress(float(probability))