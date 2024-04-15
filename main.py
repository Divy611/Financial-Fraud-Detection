import sqlite3
import pandas as pd
import streamlit as st
from fraud_detection import *

st.set_page_config(page_title='Financial Fraud Detection System', layout='wide',
                   initial_sidebar_state='auto')


css = """
<style>
:root {
  --bg-color: #22222218;
  --text-color: rgb(45, 49, 45);
}

body {
  color: var(--text-color);
  background-color: var(--bg-color);
}


.stButton>button {
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background-color: rgb(255, 255, 255);
  color: black;
  padding: 0.25rem 0.75rem;
  position: relative;
  text-decoration: none;
  border-radius: 4px;
  border-width: 1px;
  border-style: solid;
  border-color: aquamarine;
  border-image: initial;
}

.stButton>button:hover {
  border-color: rgba(9, 223, 38, 0.626);
  color: rgba(9, 223, 38, 0.626);
}

.stButton>button:active {
  box-shadow: none;
  background-color: rgba(9, 223, 38, 0.626);
  color: white;
}

.highlight {
  border-radius: 0.4rem;
  color: white;
  padding: 0.5rem;
  margin-bottom: 1rem;
}

.bold {
  padding-left: 1rem;
  font-weight: 700;
}

.blue {
  background-color: rgba(9, 223, 38, 0.626);
}

.red {
  background-color: lightblue;
}
</style>
"""


def team_page():
    st.title('Our Team')
    st.header("Aditya Pandey (AI-A) (Registration Number: 225890264)")
    st.write("")
    st.header("Suraj Prasanna (AI-B) (Registration Number: 225890296)")
    st.write("")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def database_page():
    st.title('View Database')
    conn = sqlite3.connect('fraud_data.db')
    query = "SELECT * FROM transaction_data"
    df = pd.read_sql(query, conn)
    st.write(df)
    conn.close()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def main():
    st.markdown(css, unsafe_allow_html=True)
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'main'
    with st.sidebar:
        st.info('**Financial Fraud Detection System**')
        team_button = st.button("Our Team")
        database_button = st.button('View Database')
        st.session_state.log_holder = st.empty()
        if team_button:
            st.session_state.app_mode = 'team'
            # st.session_state['page'] = 'team'
        if database_button:
            st.session_state.app_mode = 'database'
            # st.session_state['page'] = 'database'
    st.title('Financial Fraud Detection System')
    if st.session_state.app_mode == 'team':
        team_page()
    if st.session_state.app_mode == 'database':
        database_page()
    # model = load_model()

    st.write("Enter the transaction details:")
    transaction_time = st.text_input("Time:")
    user_name = st.text_input("User:")
    recipient_name = st.text_input("Recipient:")
    transaction_amount = st.number_input("Transaction Amount:")

    # clf, X_test, y_test = train_model(df)
    if st.button("Predict"):
        st.write("Evaluating the model...")
        # accuracy, precision, recall, f1 = evaluate_model(
        #     clf, X_test, y_test)
        st.write("Model evaluation complete!")
        # prediction = predict(model, input_data)
        # st.write(f"Predicted Transaction Action: {prediction}")
        st.write("Predicted Transaction Action:")

    input_data = {
        'transaction_time': transaction_time,
        'user_name': user_name,
        'recipient_name': recipient_name,
        'transaction_amount': transaction_amount,
    }
    # df = load_data()
    # df = preprocess_data(df)


# Run the app
if __name__ == '__main__':
    main()
