import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from apikey import apikey
import json
import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/c91d3a39-8c00-43f2-8f8e-6625bdb78cf9/Lhu7VUnlLa.json"

lottie_hello = load_lottieurl(lottie_url_hello)

st_lottie(lottie_hello,height=500)
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.sidebar.text_input("Please enter your openai api key", type='password')

# Define Streamlit app
def app():
      # Title and description
    st.title("Chat with your file")
    st.write("Upload a CSV or Excel file and get your answers.")
    file =  st.file_uploader("Upload your file",type=["csv","xlsx"])
    if not file:
        st.stop()
        try:
            data = pd.read_csv(file)
        except:
            data = pd.read_excel(file)    
        st.write("Data Preview:")
        st.dataframe(data.head()) 

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 

    query = st.text_input("Enter a query:") 

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)
if __name__ == "__main__":
    app()   