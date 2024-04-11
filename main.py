import sqlite3
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def load_data():
    conn = sqlite3.connect('fraud_data.db')
    # query = "SELECT * FROM json_transactions"
    # df = pd.read_sql(query, conn)
    # conn.close()
    # return df
    return


def sidebar():
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


def preprocess_data(df):
    # Convert time_id to datetime
    df['time_id'] = pd.to_datetime(df['time_id'])
    df['trans_amount'] = pd.to_numeric(df['trans_amount'])

    return df


def train_model(df):
    # Split the data into features and target variable
    X = df.drop(columns=['event_id'])
    y = df['event_id']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    return accuracy, precision, recall, f1


def load_model():
    # Load the trained model here
    model = RandomForestClassifier()
    return model


def preprocess_input(input_data):
    df = pd.DataFrame(input_data, index=[0])
    return df


def predict(model, input_data):
    input_df = preprocess_input(input_data)
    conn = sqlite3.connect('fraud_data.db')
    query = "SELECT * FROM json_transactions"
    df = pd.read_sql(query, conn)
    conn.close()

    df = preprocess_data(df)
    X = df.drop(columns=['event_id'])
    y = df['event_id']
    model.fit(X, y)
    prediction = model.predict(input_df)

    return prediction


def team_page():
    st.title('Our Team')
    st.write("Details about our team.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def database_page():
    st.title('View Database')
    st.write("Database contents here.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def main():
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'main'
    sidebar()
    # df = load_data()
    # df = preprocess_data(df)
    st.markdown(css, unsafe_allow_html=True)
    st.title('Financial Fraud Detection System')
    if st.session_state.app_mode == 'team':
        team_page()
    if st.session_state.app_mode == 'database':
        database_page()
    model = load_model()

    # User input for transaction details
    st.write("Enter the transaction details:")
    time_id = st.text_input("Time ID:")
    user_id = st.text_input("User ID:")
    trans_amount = st.number_input("Transaction Amount:")

    input_data = {
        'time_id': time_id,
        'user_id': user_id,
        'trans_amount': trans_amount,
    }

    # clf, X_test, y_test = train_model(df)
    if st.button("Predict"):
        st.write("Evaluating the model...")
        st.write(f"Predicted Event ID:")
        # accuracy, precision, recall, f1 = evaluate_model(clf, X_test, y_test)
        # st.write("Model evaluation complete!")
        # prediction = predict(model, input_data)
        # st.write(f"Predicted Event ID: {prediction}")


# Run the app
if __name__ == '__main__':
    main()
# def main():
#     clf, X_test, y_test = train_model(df)

#     # Display evaluation metrics
#     st.write("## Evaluation Metrics")
#     st.write(f"Accuracy: {accuracy:.2f}")
#     st.write(f"Precision: {precision:.2f}")
#     st.write(f"Recall: {recall:.2f}")
#     st.write(f"F1 Score: {f1:.2f}")
