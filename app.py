import streamlit as st
from bank_churn import input_transformer, model_prediction
import sklearn

if __name__ == '__main__':
    st.set_page_config(page_title="Bank Customer Churn Predictor")
    st.header("Bank Customer Churn Prediction")
    st.write("Find out if a customer will leave or not")
    
    with st.form("Inputs"):
        st.write("Enter the following details")
        geog = st.radio("Pick Country",["France","Spain","Germany"])
        
        st.write("Enter Gender")
        gender = st.radio("Pick Gender",["Male","Female"])
        
        credit_score = st.slider("Credit Score",min_value=350,max_value=850,value=650,step=1)
        
        age = st.slider("Age",min_value=18,max_value=100,value=40,step=1)
        
        tenure = st.slider("Tenure",min_value=0,max_value=10,value=5,step=1)
        
        num_of_products = st.slider("Number of Products",min_value=0,max_value=4,step=1,value=2)
        
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
            
        balance = st.number_input("Balance (in €)")
        
        estimated_salary = st.number_input("Estimated Salary (in €)")
        
        submit = st.form_submit_button("Submit")
    
    if submit:
        Xt_input = input_transformer("France","Male",credit_score,age,tenure,balance,
                                     num_of_products,has_crcard,is_active_member,estimated_salary)
        prediction = model_prediction(Xt_input)
        
        if prediction:
            st.write("Customer Will Leave")
        else:
            st.write("Customer will stay")
        st.write("""According to model, the top 3 determining features are: \n 1. Estimated salary.
                    \n 2. Whether customer is an active member or not. \n 3. Whether customer has credit card or not. \n""")
        st.markdown("""--------""")
        st.markdown("""
                    This model was built by training a neural network on a bank customer churn dataset from 
                    [kaggle](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling).\n
                    Check out the source code on my [github](https://github.com/spikspiks/bank_churn). \n
                    """)