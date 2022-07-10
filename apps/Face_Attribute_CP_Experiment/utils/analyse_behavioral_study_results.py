import os

import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st



semantics_metadata = {
    "mouth": {"layers": [4, 5], "maxAbsAlpha": 3, "minAbsAlpha": 0},
    "eyebrows": {"layers": [6], "maxAbsAlpha": 4, "minAbsAlpha": 0},
    "eyes": {"layers": [6], "maxAbsAlpha": 5, "minAbsAlpha": 0},

    "nose": {"layers": [4, 5], "maxAbsAlpha": 4, "minAbsAlpha": 0},
    "hairstyle": {"layers": [4], "maxAbsAlpha": 13, "minAbsAlpha": 0},
    "hair color": {"layers": [8, 9], "maxAbsAlpha": 3, "minAbsAlpha": 0},
}


st.title("Dashboard")
st.header("Behavioral Study Results")

def precision_rate():
    """
    This method calculates the number of paticipants and
    the precsion rate of the face attribute edits.
    """
    
    res = {
    "mouth": {'Total':0, 'Success':0, 'None':0, 'False':0},
    "eyebrows": {'Total':0, 'Success':0, 'None':0, 'False':0},
    "eyes": {'Total':0, 'Success':0, 'None':0, 'False':0},

    "nose": {'Total':0, 'Success':0, 'None':0, 'False':0},
    "hairstyle": {'Total':0, 'Success':0, 'None':0, 'False':0},
    "hair color": {'Total':0, 'Success':0, 'None':0, 'False':0},
    }
    

    number_of_participants = 0
    
    for filename in os.listdir("../behavioral_study_results"):
        
        with open("../behavioral_study_results/"+filename, 'rb') as file:
            data = pkl.load(file)

        for val in data['results'].values():
            
            res[val['Attribute']]['Total'] += 1

            if val['Modified image number'] == val['user_choice']:
                res[val['Attribute']]['Success'] += 1

            if val['user_choice'] == 0:
                res[val['Attribute']]['None'] += 1

            if (val['Modified image number'] != val['user_choice']) and (val['user_choice'] != 0):
                res[val['Attribute']]['False'] += 1


        number_of_participants += 1
        
        
    st.subheader("Number of participants: " + str(number_of_participants))
    
    st.subheader("Number of appearances:")
    df = pd.DataFrame.from_dict(res, orient='index')
    st.dataframe(df)
    #df.to_csv('appearances.csv') uncomment to save csv

    #Precision rate (in %):
    rate = {
    "mouth": {},
    "eyebrows": {},
    "eyes": {},
    "nose": {},
    "hairstyle": {},
    "hair color": {},
    }
    for key in res.keys():
        if res[key]['Total'] != 0 :
            rate[key]['Success %'] = round(res[key]['Success']/res[key]['Total'] *100, 2)
            rate[key]['None %'] = round(res[key]['None'] / res[key]['Total'] *100, 2)
            rate[key]['False %'] = round(res[key]['False'] / res[key]['Total'] *100, 2)
        else:
            rate[key]['Success %'] = 0
            rate[key]['None %'] = 0
            rate[key]['False %'] = 0
        
    
    df = pd.DataFrame.from_dict(rate, orient='index')
    st.subheader("Precision rate (in %):")
    st.dataframe(df)
    #df.to_csv('precision_rate.csv') uncomment to save csv
    
    return
    
def create_figure(x, y, alpha, attribute, label, color):
    """
    Creates and show a figure.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.bar(x, y, width=0.1, label=label, color=color)
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Number of '+ label)
    ax.set_title(attribute)
    ax.legend(loc='best')

    plt.xticks(np.arange(-alpha, alpha+ 0.1, 1))
    plt.yticks(np.arange(0, 4, 1))
    st.pyplot(plt)
    
    
def compute_alphas_plot(attribute):
    """
    Show for a face attribute "attribute" the
    plots of the success and fails in relation to the alphas.
    
    If there is no appearances of that attribute, the plot is not
    shown.
    """
    
    
    abs_alpha = semantics_metadata[attribute]['maxAbsAlpha']
    result = map(lambda x: round(x, 1), np.arange(-abs_alpha, abs_alpha+ 0.1, 0.1))
    r = list(result)
    dicti = {key: {"total":0, "success":0, "fail":0} for key in r}

    for filename in os.listdir("../behavioral_study_results"):
        with open("../behavioral_study_results/" + filename, 'rb') as file:
            data = pkl.load(file)

        for val in data['results'].values():
            if val['Attribute'] == attribute:
                alpha_rounded = round(val['Alpha'],1)

                if val['Modified image number'] == val['user_choice']:
                    dicti[alpha_rounded]['success'] += 1

                else:
                    dicti[alpha_rounded]['fail'] += 1


                dicti[alpha_rounded]['total'] += 1


    result_fail = {}
    for key in dicti.keys():
        if dicti[key]['fail'] != 0:
            result_fail[key] = dicti[key]['fail']

    result_succ = {}
    for key in dicti.keys():
        if dicti[key]['success'] != 0:
            result_succ[key] = dicti[key]['success']

    lists_0 = sorted(result_fail.items())
    lists_1 = sorted(result_succ.items())
    
    if (len(lists_0) > 0 or len(lists_1) > 0):
        st.subheader("Stats for " + attribute + ":")
        
    if len(lists_0) > 0:
        x0, y0 = zip(*lists_0)
        create_figure(x0,y0,abs_alpha, attribute, "Fail", 'r')
    
    if len(lists_1) > 0:
        x1, y1 = zip(*lists_1)
        create_figure(x1,y1,abs_alpha, attribute, "Success", 'b')
    
    return




precision_rate()
for key in semantics_metadata.keys():
    compute_alphas_plot(key)












