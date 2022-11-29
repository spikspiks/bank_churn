import streamlit as st
from bank_churn import input_transformer, model_prediction
import sklearn

if __name__ == '__main__':
    st.set_page_config(page_title="Bank Customer Churn Predictor")
    st.header("Bank Customer Churn Prediction")
    
    with st.form("Inputs"):
        st.write("Enter the following details")
        geog = st.radio("Pick Country",["France","Spain","Germany"])
        
        st.write("Enter Gender")
        gender = st.radio("Pick Gender",["Male","Female"])
        
        credit_score = st.number_input("Credit Score (in range 350 to 850)",350,850)
        
        age = st.number_input("Age")
        
        tenure = st.number_input("Tenure (in range 0 to 10)",0,10)
        
        balance = st.number_input("Balance")
        
        num_of_products = st.number_input("Number of Products (in range 0 to 4)",0,4)
        
        cred_card = st.radio("Credit Card",["Yes","No"])
        if cred_card == "Yes":
            has_crcard=1
        else:
            has_crcard=0
            
        active_member = st.radio("Active Member",["Yes","No"])
        if active_member == "Yes":
            is_active_member=1
        else:
            is_active_member=0
            
        estimated_salary = st.number_input("Estimated Salary")
        
        submit = st.form_submit_button("Submit")
    
    if submit:
        Xt_input = input_transformer("France","Male",credit_score,age,tenure,balance,
                                     num_of_products,has_crcard,is_active_member,estimated_salary)
        prediction = model_prediction(Xt_input)
        
        if prediction:
            st.write("Customer Will Leave")
        else:
            st.write("Customer will stay")