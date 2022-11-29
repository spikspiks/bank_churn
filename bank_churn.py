import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import pickle

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.layers = nn.Sequential(nn.Linear(11,50), nn.ReLU(),
                                    nn.Linear(50,10), nn.ReLU(),
                                    nn.Linear(10,1), nn.Sigmoid())
        
    def forward(self,x):
        return self.layers(x)

def input_transformer(geog,gender,credit_score,age,tenure,balance,num_of_products,has_crcard,is_active_member,estimated_salary):
    col_tran = pickle.load(open('col_transformer.pkl','rb'))
    input_df = pd.DataFrame([[geog,gender,credit_score,age,tenure,balance,
                             num_of_products,has_crcard,is_active_member,
                             estimated_salary]],
                             columns=['Geography','Gender','CreditScore','Age','Tenure','Balance',
                                      'NumOfProducts','HasCrCard','IsActiveMember',
                                      'EstimatedSalary'])
    X_input = col_tran.transform(input_df)
    Xt_input = torch.from_numpy(X_input).to(torch.float)
    return Xt_input

def model_prediction(X):
    model = NN()
    model.load_state_dict(torch.load('model_state.pt'))
    y = model(X).detach().numpy()
    if y[0][0] > 0.5:
        return True
    else: 
        return False


     