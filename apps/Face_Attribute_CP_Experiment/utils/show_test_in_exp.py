import pickle as pkl
import streamlit as st




########  INPUT ########

EXPERIMENT_NUMBER = 0
TEST_NUMBER = 2

########################


folder_name = '../experiments_data'
file_name = folder_name + '/' + str(EXPERIMENT_NUMBER) +'/'+ 'experiment_' + str(EXPERIMENT_NUMBER) + '_images.pkl'

with open(file_name, 'rb') as file:
    data = pkl.load(file)
    st.image(data[str(TEST_NUMBER)]["Initial Combined Image"])
    st.image(data[str(TEST_NUMBER)]["Middle Image"])
    st.image(data[str(TEST_NUMBER)]["Final Combined Image"]) 
    st.image(data[str(TEST_NUMBER)]["Intital Modified Image"])
    st.image(data[str(TEST_NUMBER)]["Final Modified Image"])
#     print((data[str(sample_number)]["Origin Ws"]))
#     print((data[str(sample_number)]["Final Ws"]))

  

        
