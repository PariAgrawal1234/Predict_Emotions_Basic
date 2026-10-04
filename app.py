import streamlit as st
import pickle

model = pickle.load(open('npl_logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_bow.pkl', 'rb'))

st.title("NPL Sentiment Analysis")

text = st.text_area(
    'Enter your statement here:',
    placeholder='Example = I am really Happy today!',         
    )

if st.button("Predict"):
    if text:
        # Transform the input text using the loaded vectorizer
        transformed_text = vectorizer.transform([text])
        
        # Make a prediction using the loaded model
        prediction = model.predict(transformed_text)[0]
        
        st.success(f'The sentiment of the text is: {prediction}')
    else:
        st.warning("Please enter some text to analyze.")