import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

# Define the function to clean the data
def clean_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove non-breaking spaces and any non-ASCII characters
            df[col] = df[col].str.replace(u'\xa0', ' ', regex=False)
            df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII characters
    return df

# Define the custom decision rule-based classifier
class RandomDecisionScorer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            score = 0
            if row['DaysSinceLastPurchase'] > self.thresholds['DaysSinceLastPurchase']:
                score += 1
            if row['TotalSpent'] < self.thresholds['TotalSpent']:
                score += 1
            if row['TotalPurchases'] < self.thresholds['TotalPurchases']:
                score += 1
            predictions.append(1 if score >= 2 else 0)
        return predictions

# Step 1: File upload using Streamlit's file uploader
st.title("Customer Churn Analysis & Notification System")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Clean the dataset
        df = clean_data(df)

        st.write("Dataset Preview:")
        st.write(df.head())

    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Step 2: Algorithm Selection
if 'df' in locals():
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    algorithm_choice = st.selectbox("Choose an algorithm", list(algorithms.keys()) + ['Custom Decision Rule-Based Classifier'])

    # Step 3: Train model
    if st.button("Generate Churn Dataset"):
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
            df['IsChurned'] = df['DaysSinceLastPurchase'] > 30
            
            X = df[['TotalPurchases', 'TotalSpent', 'DaysSinceLastPurchase']]
            y = df['IsChurned']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if algorithm_choice != 'Custom Decision Rule-Based Classifier':
                model = algorithms[algorithm_choice]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                thresholds = {'DaysSinceLastPurchase': 45, 'TotalSpent': 300, 'TotalPurchases': 5}
                rds = RandomDecisionScorer(thresholds)
                y_pred = rds.predict(pd.DataFrame(X_test, columns=X.columns))

            # Show model performance
            st.subheader(f"{algorithm_choice} Results")
            st.write(confusion_matrix(y_test, y_pred))
            st.write(classification_report(y_test, y_pred, zero_division=0))
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

            # Save the churned dataset
            churned_customers = df[df['IsChurned'] == True]
            churned_customers.to_excel('churned_customers.xlsx', index=False, engine='openpyxl')
            st.download_button("Download Churn Dataset", data=open('churned_customers.xlsx', 'rb').read(), file_name='churned_customers.xlsx')

        except Exception as e:
            st.error(f"Error processing the dataset: {e}")

# Step 4: Upload churned customers dataset
st.subheader("Upload Churned Customers Excel File")

uploaded_churned_file = st.file_uploader("Upload the churned customers Excel file", type=["xlsx"], key="churned_file")
if uploaded_churned_file:
    try:
        df_churned = pd.read_excel(uploaded_churned_file, engine='openpyxl')
        # Clean the churned customers dataset
        df_churned = clean_data(df_churned)

    except Exception as e:
        st.error(f"Failed to read the Excel file. Error: {e}")

# Step 5: Email Sending
if 'df_churned' in locals():
    st.subheader("Send Emails to Churned Customers")

    col1, col2 = st.columns(2)
    with col1:
        email_input = st.text_input("Enter your email", key='email', placeholder="Your email")
    with col2:
        password_input = st.text_input("Enter your app-specific password", type='password', key='password', placeholder="App-specific password")

    if st.button("Save Email Credentials"):
        if email_input and password_input:
            st.session_state.email = email_input
            st.session_state.password = password_input
            st.success("Credentials saved. You can now send emails.")
        else:
            st.error("Please enter both email and password.")

    if st.button("Send Emails"):
        if 'email' in st.session_state and 'password' in st.session_state:
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(st.session_state.email, st.session_state.password)

                subject = "Special Offer Just for You!"
                body = """
                Dear Customer,

                We hope you are doing well! We have an exciting offer on our latest products.
                Check out our website for more details!

                Best Regards,
                Your Online Retail Team
                """

                for _, row in df_churned.head(10).iterrows():
                    msg = MIMEMultipart()
                    msg['From'] = st.session_state.email
                    msg['To'] = row['Email']
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body, 'plain', 'utf-8'))

                    server.send_message(msg)
                    st.write(f"Email sent to {row['Email']}")

                server.quit()
                st.success("Emails sent successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please save email credentials first.")

# Step 6: Pincode Analysis and Visualization
if st.button("Generate Pincode Analysis"):
    if 'df' in locals():
        try:
            pincode_analysis = df.groupby('Pincode').agg({
                'CustomerID': 'count',
                'TotalSpent': 'sum',
                'TotalPurchases': 'sum',
                'Description': lambda x: ', '.join(x.unique())
            }).reset_index()
            pincode_analysis.columns = ['Pincode', 'NumberOfChurnedCustomers', 'TotalSpentByChurnedCustomers', 'TotalPurchasesByChurnedCustomers', 'ProductNames']

            st.subheader("Pincode-wise Churn Analysis")
            st.write(pincode_analysis)

            pincode_analysis_excel = BytesIO()
            with pd.ExcelWriter(pincode_analysis_excel, engine='openpyxl') as writer:
                pincode_analysis.to_excel(writer, index=False, sheet_name='Pincode Analysis')
            st.download_button("Download Pincode Analysis", data=pincode_analysis_excel.getvalue(), file_name='pincode_wise_analysis.xlsx')

            # Visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(data=pincode_analysis, x='Pincode', y='NumberOfChurnedCustomers')
            plt.title('Number of Churned Customers by Pincode')
            plt.xticks(rotation=45)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error generating pincode analysis: {e}")
