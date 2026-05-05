import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stMetric { background: #f0f4ff; border-radius: 10px; padding: 1rem; }
    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Load and train model ---
@st.cache_resource
def load_model():
    df = pd.read_csv('student_data.csv')
    cat_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df_encoded.drop('G3', axis=1)
    y = df_encoded['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, df

model, X_train, df = load_model()

# --- Title ---
st.markdown("""
<div class="title-box">
    <h1>Student Performance Predictor</h1>
    <p style="margin:0; opacity:0.8;">Predict a student's final grade based on their academic profile</p>
</div>
""", unsafe_allow_html=True)

# --- Input fields ---
st.subheader("Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Academic Grades**")
    G1 = st.number_input("First Term Grade G1 (out of 15)", min_value=0, max_value=15, value=8,
                         help="Enter a number between 0 and 15")
    G2 = st.number_input("Second Term Grade G2 (out of 25)", min_value=0, max_value=25, value=13,
                         help="Enter a number between 0 and 25")
    failures = st.number_input("Past Class Failures", min_value=0, max_value=3, value=0,
                               help="Number of times the student failed a class before")
    st.caption(f"G1: {G1}/15 = {G1/15*100:.0f}% | G2: {G2}/25 = {G2/25*100:.0f}%")

with col2:
    st.markdown("**Study Habits**")
    studytime = st.number_input("Study Time (1=low, 4=high)", min_value=1, max_value=4, value=2,
                                help="1=less than 2hrs, 2=2-5hrs, 3=5-10hrs, 4=more than 10hrs per week")
    absences = st.number_input("Number of Absences", min_value=0, max_value=93, value=5,
                               help="Total school absences this year")
    goout = st.number_input("Going Out with Friends (1=low, 5=high)", min_value=1, max_value=5, value=3,
                            help="How often the student goes out socially")

with col3:
    st.markdown("**Lifestyle**")
    Dalc = st.number_input("Weekday Alcohol Consumption (1=low, 5=high)", min_value=1, max_value=5, value=1,
                           help="Alcohol consumption on school days")
    Walc = st.number_input("Weekend Alcohol Consumption (1=low, 5=high)", min_value=1, max_value=5, value=1,
                           help="Alcohol consumption on weekends")
    health = st.number_input("Health Status (1=bad, 5=excellent)", min_value=1, max_value=5, value=3,
                             help="Current health status of the student")

st.markdown("---")

# --- Predict button ---
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_clicked = st.button("Predict Final Grade", use_container_width=True)

# --- Convert G1/G2 to 0-20 scale for model ---
G1_scaled = round((G1 / 15) * 20, 1)
G2_scaled = round((G2 / 25) * 20, 1)

# --- Results ---
if predict_clicked:
    input_row = pd.DataFrame([X_train.mean()], columns=X_train.columns)
    for col, val in [('G1', G1_scaled), ('G2', G2_scaled),
                     ('studytime', studytime), ('failures', failures),
                     ('absences', absences), ('goout', goout),
                     ('Dalc', Dalc), ('Walc', Walc), ('health', health)]:
        if col in input_row.columns:
            input_row[col] = val

    # Raw prediction is on 0-20 scale, convert to 0-35
    raw_pred = model.predict(input_row)[0]
    prediction_35 = round((raw_pred / 20) * 35, 1)
    prediction_35 = max(0, min(35, prediction_35))
    percentage = (prediction_35 / 35) * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Predicted Final Grade (G3)", f"{prediction_35} / 35",
                  help="Final grade scaled to 35 marks")
    with res_col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with res_col3:
        if percentage >= 80:
            grade_letter = "A"
        elif percentage >= 65:
            grade_letter = "B"
        elif percentage >= 50:
            grade_letter = "C"
        elif percentage >= 35:
            grade_letter = "D"
        else:
            grade_letter = "F"
        st.metric("Grade", grade_letter)

    # Status message
    if percentage >= 80:
        st.success("Excellent performance expected! Keep it up!")
    elif percentage >= 65:
        st.info("Good performance expected. A little more effort can get you to A!")
    elif percentage >= 50:
        st.warning("Passing, but there is room to improve. Focus on study time and attendance.")
    else:
        st.error("At risk of failing. Needs immediate attention and support.")

    # --- Progress bar ---
    st.markdown(f"**Score Progress: {prediction_35}/35**")
    st.progress(prediction_35 / 35)

    # --- Comparison table ---
    st.markdown("---")
    st.subheader("How This Student Compares to Class Averages")

    avg = df[['G1', 'G2', 'studytime', 'failures', 'absences']].mean()

    compare_df = pd.DataFrame({
        'Your Student': [f"{G1}/15", f"{G2}/25", studytime, failures, absences],
        'Class Average (original scale)': [
            f"{avg['G1']:.1f}/20",
            f"{avg['G2']:.1f}/20",
            f"{avg['studytime']:.1f}/4",
            f"{avg['failures']:.1f}",
            f"{avg['absences']:.1f}"
        ]
    }, index=['G1 (Term 1)', 'G2 (Term 2)', 'Study Time', 'Failures', 'Absences'])

    st.dataframe(compare_df, use_container_width=True)

    # --- Key insight ---
    st.markdown("---")
    st.info(f"""
    **Key Insight:** Based on G1 ({G1}/15) and G2 ({G2}/25), 
    the model predicts a final G3 of **{prediction_35}/35** ({percentage:.1f}%).  
    """)