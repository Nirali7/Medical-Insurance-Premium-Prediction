import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Medical Insurance Premium Prediction",
    page_icon="🏥",
    layout="wide"
)

model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")

categorical_columns = ["smoker", "region", "plan_type"]
numerical_columns = ["age", "bmi"]


st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #d9ecff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #12304a;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #527086;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.82);
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 8px 30px rgba(30, 80, 120, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.8);
}

.prediction-card {
    background: linear-gradient(135deg, #0f9d58, #34c759);
    padding: 32px;
    border-radius: 22px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(15, 157, 88, 0.25);
    margin-top: 25px;
}

.prediction-label {
    font-size: 18px;
    font-weight: 500;
}

.prediction-value {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #12304a;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    background: rgba(255, 255, 255, 0.75);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.8);
}

.footer {
    text-align: center;
    color: #527086;
    margin-top: 30px;
    padding: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


with st.sidebar:

    st.markdown("## 🏥 Medical Insurance")

    st.markdown("---")

    st.markdown("### 🤖 Machine Learning")
    st.write("Annual medical insurance premium prediction.")

    st.markdown("### 📦 Model Status")
    st.success("Model Loaded")

    st.markdown("### 🔐 Encoder Status")
    st.success("Encoder Loaded")

    st.markdown("---")

    st.markdown("### 👩‍💻 Developer")
    st.write("Nirali Banubakode")

    st.markdown("---")

    st.caption("Medical Insurance Premium Prediction System")


st.markdown(
    '<div class="main-title">🏥 Medical Insurance Premium Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict annual medical insurance premium using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=35,
        step=1
    )

    bmi = st.slider(
        "BMI",
        min_value=15.0,
        max_value=45.0,
        value=27.5,
        step=0.1
    )

    smoker = st.radio(
        "Smoking Status",
        encoder.categories_[0],
        horizontal=True
    )


with col2:

    region = st.selectbox(
        "Region",
        encoder.categories_[1]
    )

    plan_type = st.selectbox(
        "Insurance Plan",
        encoder.categories_[2]
    )


st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")


predict = st.button("🔮 Predict Insurance Premium")


if predict:

    sample_input = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "smoker": [smoker],
        "region": [region],
        "plan_type": [plan_type]
    })

    sample_encoded = encoder.transform(
        sample_input[categorical_columns]
    )

    sample_encoded_df = pd.DataFrame(
        sample_encoded,
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    sample_processed = pd.concat(
        [
            sample_input[numerical_columns].reset_index(drop=True),
            sample_encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    prediction = model.predict(sample_processed)[0]

    st.session_state["prediction"] = prediction
    st.session_state["age"] = age
    st.session_state["bmi"] = bmi
    st.session_state["smoker"] = smoker
    st.session_state["region"] = region
    st.session_state["plan_type"] = plan_type


if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]


    st.markdown(
        '<div class="prediction-card">'
        '<div class="prediction-label">'
        'Estimated Annual Insurance Premium'
        '</div>'
        f'<div class="prediction-value">₹ {prediction:,.2f}</div>'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title">📋 Prediction Summary</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:
        st.markdown(
            f'''
            <div class="info-box">
                <b>Age</b><br>
                {st.session_state["age"]}
            </div>
            ''',
            unsafe_allow_html=True
        )


    with c2:
        st.markdown(
            f'''
            <div class="info-box">
                <b>BMI</b><br>
                {st.session_state["bmi"]:.1f}
            </div>
            ''',
            unsafe_allow_html=True
        )


    with c3:
        st.markdown(
            f'''
            <div class="info-box">
                <b>Smoker</b><br>
                {st.session_state["smoker"]}
            </div>
            ''',
            unsafe_allow_html=True
        )


    with c4:
        st.markdown(
            f'''
            <div class="info-box">
                <b>Region</b><br>
                {st.session_state["region"]}
            </div>
            ''',
            unsafe_allow_html=True
        )


    with c5:
        st.markdown(
            f'''
            <div class="info-box">
                <b>Plan</b><br>
                {st.session_state["plan_type"]}
            </div>
            ''',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">💡 Prediction Information</div>',
        unsafe_allow_html=True
    )


    with st.expander("🔎 How was this prediction generated?"):

        st.write(
            "The trained machine learning model uses age, BMI, "
            "smoking status, region and insurance plan to estimate "
            "the annual medical insurance premium."
        )

        st.write(
            "The categorical features are converted using the same "
            "OneHotEncoder that was used during model training."
        )

        st.write(
            "The processed input is then passed to the saved best "
            "regression model to generate the premium prediction."
        )


    st.balloons()


st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Developed by <b>Nirali Banubakode</b>'
    '</div>',
    unsafe_allow_html=True
)