import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np


st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🤖 AI Dashboard")

tab1, tab2, tab3 = st.tabs(["Food Sentiment", "Spam Classifier", "Customer Churn Prediction"])

with tab1:

    def mycleaning(doc):
        return re.sub("[^a-zA-Z ]","",doc).lower()
    
    model=joblib.load("sentiment_model.pkl")

    st.markdown("""
        <div style="
            background-image: linear-gradient(rgba(0,0,0,0.60), rgba(0,0,0,0.70)),
                          url('https://images.unsplash.com/photo-1517248135467-4c7edcad34c4');
            background-size: cover;
            background-position: center;
            padding: 10px;
            border-radius: 20px;
            text-align: center;
        ">
            <h1 style="
                color: #ffffff;
                font-size: 30px;
                margin-bottom: 4px;
                letter-spacing: 0px;
            ">
                🍽️ Food Sentiment Analysis
            </h1>
            <p style="
                color: #dddddd;
                ont-size: 15px;
            ">
                Discover • Order • Enjoy
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <h2 style='
        text-align: center;
        background: linear-gradient(90deg, #000080, #5cd65c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: bold;'>
        💫 Control Panel
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.image("restraunt.jpg")


    st.sidebar.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #000080, #5cd65c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: bold;'>
        About Project 💻
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.write("Decode customer emotions from restaurant reviews using AI.")


    st.sidebar.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #000080, #5cd65c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: bold;'>
        Libraries 🐍
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.write("Scikit Learn")
    st.sidebar.write("Pandas")
    st.sidebar.write("Numpy")
    st.sidebar.write("Joblib")
    st.sidebar.write("Streamlit")

    st.sidebar.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #000080, #5cd65c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: bold;'>
        Contact us 📞
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.write("639761XXXX")

    st.sidebar.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #000080, #5cd65c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: bold;'>
        About us 👥
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.write("Enter your review and let AI predict the sentiment")

    st.write("\n")

    st.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #1E90FF, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px;
        font-weight: bold;'>
        💫 Enter Review
    </h2>
    """, unsafe_allow_html=True)

    sample=st.text_input("")
    if st.button("Predict"):
        pred=model.predict([sample])
        prob=model.predict_proba([sample])
        if pred[0]==0:
            st.write("Neg 😔")
            st.write(f"Confidence Score : {prob[0][0]:.2f}")
        else:
            st.write("Pos 🥰")
            st.write(f"Confidence Score : {prob[0][1]:.2f}")
            st.balloons()

    st.markdown("""
    <h2 style='
        text-align: left;
        background: linear-gradient(90deg, #1E90FF, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px;
        font-weight: bold;'>
        💫 Bulk Prediction
    </h2>
    """, unsafe_allow_html=True)

    file=st.file_uploader("select file",type=["csv","txt"])
    if file:
        df=pd.read_csv(file,names=["Review"])
        placeholder=st.empty()
        placeholder.dataframe(df)
    
        if st.button("Predict",key="b2"):
            corpus=df.Review
            pred=model.predict(corpus)
            prob=np.max(model.predict_proba(corpus),axis=1)
            df['Sentiment']=pred
            df['Confidence']=prob
            df['Sentiment']=df['Sentiment'].map({0:"Neg 😔",1:"Pos 😍"})
            placeholder.dataframe(df)

with tab2:
    def mycleaning(doc):
        return re.sub("[^a-zA-Z ]","",doc).lower()
        
    model=joblib.load("spam_model.pkl")

    st.markdown(
        """
        <div style="
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 10px;
        ">
            <h1 style="color: #00FFFF;">📩 Spam Classifier</h1>
            <p style="color: white; font-size:18px;">
                Detect whether a message is Spam or Not Spam using NLP 🚀
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.image("msg.jpg")

    st.sidebar.title("About Project🧠")
    st.sidebar.write("""This project is an AI-based Spam Message Classifier that uses 🧠 Natural Language Processing (NLP) techniques 
                    to identify whether a message is Spam 🚫 or Not Spam ✅""")

    st.sidebar.title("Contact us📞")
    st.sidebar.write("639761xxxx")


    st.write("\n")
    st.write("### Enter Msg")
    sample=st.text_area("")
    if st.button("Check Spam"):
        pred=model.predict([sample])
        prob=model.predict_proba([sample])
        if pred[0]=='ham':
            st.write("Valid")
            st.write(f"Confidence Score : {prob[0][0]:.2f}")
        else:
            st.write("SPAM")
            st.write(f"Confidence Score : {prob[0][1]:.2f}")
            st.balloons()

    st.write("### Bulk Prediction")
    file=st.file_uploader("open file",type=["csv","txt"])
    if file:
        df=pd.read_csv(file,names=["MSG"])
        placeholder=st.empty()
        placeholder.dataframe(df)
        
        if st.button("Check Spam",key="b3"):
            corpus=df.MSG
            pred=model.predict(corpus)
            prob=np.max(model.predict_proba(corpus),axis=1)
            df['Msg Type']=pred
            df['Confidence']=prob
            placeholder.dataframe(df)

with tab3:
    st.markdown("""
    <style>
    .banner {
        background: linear-gradient(90deg, #000000, #434343);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        font-size: 18px;
        color: #cccccc;
    }
    </style>

    <div class="banner">
        <div class="title">🧑‍💼Customer Churn Prediction</div>
    </div>
    """, unsafe_allow_html=True)



    st.sidebar.image("customer.jpg")

    st.sidebar.title("About Project")

    st.sidebar.write("""Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company’s service based on customer behavior, account details, and usage patterns.
    The model helps businesses identify at-risk customers and improve customer retention strategies.
    """)
    
    model = joblib.load(open("model.pkl", "rb"))
    columns = joblib.load(open("columns.pkl", "rb"))

    st.title("Customer Churn Prediction")

    # Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    tenure = st.slider("Tenure", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    # Create input dictionary
    input_dict = {
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Add categorical encoding
    categorical_inputs = [
        ("gender", gender),
        ("Partner", Partner),
        ("Dependents", Dependents),
        ("PhoneService", PhoneService),
        ("MultipleLines", MultipleLines),
        ("InternetService", InternetService),
        ("OnlineSecurity", OnlineSecurity),
        ("OnlineBackup", OnlineBackup),
        ("DeviceProtection", DeviceProtection),
        ("TechSupport", TechSupport),
        ("StreamingTV", StreamingTV),
        ("StreamingMovies", StreamingMovies),
        ("Contract", Contract),
        ("PaperlessBilling", PaperlessBilling),
        ("PaymentMethod", PaymentMethod)
    ]

    for col, val in categorical_inputs:
        input_dict[f"{col}_{val}"] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    if st.button("Predict Churn"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"High Risk ({prob*100:.2f}%)")
    
        else:
            st.success(f" Low Risk ({prob*100:.2f}%)")
