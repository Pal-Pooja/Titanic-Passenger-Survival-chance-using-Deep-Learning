import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
st.title("Passanger survival chance in the Titanic Journey")
Pclass=st.slider("Enter Passanger class for the user ",1,3)
Sex=st.selectbox("Enter the gender",["male","female"])
SibSp=st.slider("Enter the passanger's total number of sibling and spouse",1,8)
Parch=st.slider("Enter the passaner's total number of parents and child",0,8)
Fare=st.number_input("Enter the Fare Amount")
Embarked=st.selectbox("Enter the passanger's station from where they started the journey",["Southampton","Chebourg","Queensdown"])

data=pd.DataFrame([{"Pclass":Pclass,"Sex":Sex,"SibSp":SibSp,"Parch":Parch,"Fare":Fare,"Embarked":Embarked}])
model=load_model("model.h5")

with open("Label_encoder.pkl","rb") as file:
    label=pickle.load(file)


with open("onehot_encoder.pkl","rb") as file:
    Onehot=pickle.load(file)


with open("Scaller.pkl","rb") as file:
    Scaller=pickle.load(file)

data["Sex"]=label.transform(data["Sex"])
print(data.columns)

Embarked = onehot.transform(data[["Embarked"]])

embarked = pd.DataFrame(
    Embarked,
    columns=onehot.get_feature_names_out(["Embarked"])
)

data = pd.concat(
    [data.drop(columns=["Embarked"]), embarked],
    axis=1
)
data[["Pclass","SibSp","Parch","Fare"]]=Scaller.transform(data[["Pclass","SibSp","Parch","Fare"]])
y=model.predict(data)
y=y[0][0]

def chance(y):
    if y>0.5:
        return "The Passanger will Survive the journey"
    else:
        return "The Passanger will not survive"
    
if st.button("Predict Survival chance"):
    st.write("Probability of passanger survival chance",y)
    st.write(chance(y))
