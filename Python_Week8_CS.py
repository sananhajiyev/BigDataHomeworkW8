import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

water_data = pd.read_csv('water_potability.csv')
loan_data = pd.read_csv('loan_pred.csv')


def water_potability_model(data):
    st.subheader("Water Potability Prediction Model")
    
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write(f"Accuracy: {round(accuracy, 2)*100}%")
    st.text("Classification Report:")
    st.text(report)

def loan_prediction_model(data):
    st.subheader("Loan Approval Prediction Model")
    
    data = pd.get_dummies(data, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], 
                          drop_first=True)
    
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write(f"Accuracy: {round(accuracy, 2)*100}%")
    st.text("Classification Report:")
    st.text(report)

def main():
    st.title("Machine Learning Web App")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a Page", ["Introduction", "Water Potability Model", "Loan Prediction Model"])
    
    if page == "Introduction":
        st.subheader("Introduction")
        st.write("Welcome to my first Machine Learning Web App. This application is designed to showcase machine learning models for two different datasets: Water Potability Prediction and Loan Approval Prediction.")
        st.write("In the following pages, you can explore the functionality of these models, evaluate their performance, and learn about the datasets used.")
        st.write("Use the sidebar on the left to navigate between different sections of this web app.")
        
    elif page == "Water Potability Model":
        water_potability_model(water_data)
        
    elif page == "Loan Prediction Model":
        loan_prediction_model(loan_data)



if __name__ == '__main__':
    main()