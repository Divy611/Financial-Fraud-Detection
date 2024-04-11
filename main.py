import sqlite3
import sqlalchemy
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        st.info('***Financial Fraud Detection System***')
        database_button = st.button('View Database')
        team_button = st.button("Our Team")
        st.session_state.log_holder = st.empty()
        if team_button:
            st.session_state.app_mode = 'dataset'
        if database_button:
            st.session_state.app_mode = 'recommend'


def preprocess_data(df):
    # Convert time_id to datetime
    df['time_id'] = pd.to_datetime(df['time_id'])
    df['trans_amount'] = pd.to_numeric(df['trans_amount'])

    # Perform additional preprocessing steps if needed

    return df

# Train the model


def train_model(df):
    # Split the data into features and target variable
    X = df.drop(columns=['event_id'])
    y = df['event_id']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    return clf, X_test, y_test

# Evaluate the model


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

# Function to preprocess input data


def preprocess_input(input_data):
    # Preprocess the input data here
    # For simplicity, we'll just convert the input dictionary to a DataFrame
    df = pd.DataFrame(input_data, index=[0])
    return df

# Predict function


def predict(model, input_data):
    # Preprocess input data
    input_df = preprocess_input(input_data)

    # Load and preprocess data
    # Replace 'your_database.db' with your SQLite database file path
    conn = sqlite3.connect('your_database.db')
    # Adjust the query according to your table name
    query = "SELECT * FROM json_transactions"
    df = pd.read_sql(query, conn)
    conn.close()

    # Preprocess data
    df = preprocess_data(df)

    # Select relevant features for prediction
    X = df.drop(columns=['event_id'])
    y = df['event_id']

    # Train the model
    model.fit(X, y)

    # Predict
    prediction = model.predict(input_df)

    return prediction


def main():
    sidebar()
    st.markdown(css, unsafe_allow_html=True)
    st.title('Financial Fraud Detection System')

    # Load trained model
    model = load_model()

    # User input for transaction details
    st.write("Enter transaction details:")
    time_id = st.text_input("Time ID:")
    user_id = st.text_input("User ID:")
    trans_amount = st.number_input("Transaction Amount:")
    # Add more input fields as needed

    input_data = {
        'time_id': time_id,
        'user_id': user_id,
        'trans_amount': trans_amount
        # Add more input fields as needed
    }

    # Predict
    if st.button("Predict"):
        prediction = predict(model, input_data)
        st.write(f"Predicted Event ID: {prediction}")


# Run the app
if __name__ == '__main__':
    main()
# def main():
#     sidebar()
#     st.markdown(css, unsafe_allow_html=True)
#     st.title('Financial Fraud Detection System')
#     df = load_data()
#     df = preprocess_data(df)

#     clf, X_test, y_test = train_model(df)

#     # Evaluate the model
#     st.write("Evaluating the model...")
#     accuracy, precision, recall, f1 = evaluate_model(clf, X_test, y_test)
#     st.write("Model evaluation complete!")

#     # Display evaluation metrics
#     st.write("## Evaluation Metrics")
#     st.write(f"Accuracy: {accuracy:.2f}")
#     st.write(f"Precision: {precision:.2f}")
#     st.write(f"Recall: {recall:.2f}")
#     st.write(f"F1 Score: {f1:.2f}")


# # Run the app
# if __name__ == '__main__':
#     main()
