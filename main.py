import sqlite3
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
    query = "SELECT * FROM transaction_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_data(df):
    print("Original DataFrame:")
    print(df.head())
    print("Data types before preprocessing:")
    print(df.dtypes)

    # Convert 'transaction_time' to datetime
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])

    # Perform one-hot encoding for 'user_name' if the column exists
    if 'user_name' in df.columns:
        df = pd.get_dummies(df, columns=['user_name'], drop_first=True)

    # Drop non-numeric columns before imputation
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df.drop(columns=non_numeric_cols, inplace=True)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    print("DataFrame after preprocessing:")
    print(df.head())
    print("Data types after preprocessing:")
    print(df.dtypes)

    return df


# def preprocess_data(df):
#     print("Original DataFrame:")
#     print(df.head())
#     print("Data types before preprocessing:")
#     print(df.dtypes)
#     df['transaction_time'] = pd.to_datetime(df['transaction_time'])

#     if 'user_name' in df.columns:
#         df = pd.get_dummies(df, columns=['user_name'], drop_first=True)
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     imputer = SimpleImputer(strategy='mean')
#     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#     df['social_security_number'] = df['social_security_number'].astype(
#         str).fillna('Unknown')

#     encoder = OneHotEncoder(sparse=False)
#     encoded_ss_numbers = encoder.fit_transform(df[['social_security_number']])
#     encoded_ss_df = pd.DataFrame(encoded_ss_numbers, columns=[
#                                  f'social_security_number_{i}' for i in range(encoded_ss_numbers.shape[1])])
#     df = pd.concat(
#         [df.drop(columns=['social_security_number']), encoded_ss_df], axis=1)

#     print("DataFrame after preprocessing:")
#     print(df.head())
#     print("Data types after preprocessing:")
#     print(df.dtypes)

#     return df


def train_model(df):
    target_variable = 'transaction_action'
    if target_variable not in df.columns:
        raise ValueError(
            f"Column '{target_variable}' does not exist in DataFrame. Please check your DataFrame columns.")
    df = preprocess_data(df)

    X = df.drop(columns=[target_variable, 'transaction_time'])
    y = df[target_variable]
    X.columns = X.columns.astype(str)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42)
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
    model = RandomForestClassifier()
    return model


def preprocess_input(input_data):
    df = pd.DataFrame(input_data, index=[0])
    return df


def predict(model, input_data):
    input_df = preprocess_input(input_data)
    conn = sqlite3.connect('fraud_data.db')
    query = "SELECT * FROM transaction_data"
    df = pd.read_sql(query, conn)
    conn.close()

    df = preprocess_data(df)
    X = df.drop(columns=['transaction_action'])
    y = df['transaction_action']
    model.fit(X, y)
    prediction = model.predict(input_df)
    return prediction


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
    df = load_data()
    df = preprocess_data(df)
    st.markdown(css, unsafe_allow_html=True)
    st.title('Financial Fraud Detection System')
    if st.session_state.app_mode == 'team':
        team_page()
    if st.session_state.app_mode == 'database':
        database_page()
    model = load_model()

    st.write("Enter the transaction details:")
    transaction_time = st.text_input("Time:")
    user_name = st.text_input("User:")
    recipient_name = st.text_input("Recipient:")
    transaction_amount = st.number_input("Transaction Amount:")

    input_data = {
        'transaction_time': transaction_time,
        'user_name': user_name,
        'recipient_name': recipient_name,
        'transaction_amount': transaction_amount,
    }

    clf, X_test, y_test = train_model(df)
    if st.button("Predict"):
        st.write("Evaluating the model...")
        accuracy, precision, recall, f1 = evaluate_model(clf, X_test, y_test)
        st.write("Model evaluation complete!")
        prediction = predict(model, input_data)
        st.write(f"Predicted Transaction Action: {prediction}")


# Run the app
if __name__ == '__main__':
    main()
