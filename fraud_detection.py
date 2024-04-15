import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

conn = sqlite3.connect('fraud_data.db')
query = "SELECT * FROM transaction_data"
df = pd.read_sql(query, conn)
conn.close()
print(df.head())

label_encoders = {}
for column in ['transaction_action', 'user_name', 'recipient_name']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['transaction_day'] = df['transaction_time'].dt.day
df['transaction_month'] = df['transaction_time'].dt.month
df['transaction_week'] = df['transaction_time'].dt.week

# Normalize numerical features
df['transaction_amount'] = (df['transaction_amount'] -
                            df['transaction_amount'].mean()) / df['transaction_amount'].std()

# Random Forest
X = df.drop(['social_security_number', 'transaction_time',
            'transaction_nature'], axis=1)  # Features
y = df['social_security_number']  # Target variable
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)


def predict_fraudulence(date, user_name, recipient_name, transaction_action, transaction_amount):
    input_data = pd.DataFrame({
        'user_name': [user_name],
        'transaction_amount': [transaction_amount],
        'recipient_name': [recipient_name],
        'transaction_action': [transaction_action],
        'transaction_day': [date.day],
        'transaction_month': [date.month],
        'transaction_week': [date.week]
    })
    input_data = input_data[X.columns]

    # Convert categorical variables to numerical representations
    for column in ['transaction_action', 'user_name', 'recipient_name']:
        input_data[column] = label_encoders[column].transform(
            input_data[column])
    prediction = clf.predict(input_data)
    if prediction[0] == 1:
        return "Fraudulent Transaction"
    else:
        return "Legitimate Transaction"


date = pd.to_datetime(input("Enter transaction date (YYYY-MM-DD): "))
user_name = input("Enter user name: ")
recipient_name = input("Enter recipient name: ")
transaction_action = input(
    "Enter transaction action: ")
transaction_amount = float(input("Enter transaction amount: "))

result = predict_fraudulence(
    date, user_name, recipient_name, transaction_action, transaction_amount)
print("Prediction:", result)
