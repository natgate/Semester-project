import os
import pickle as pkl
import random
import string
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import streamlit as st


#Behavioral study app on streamlit
#Display to a user one experiment from the experiments_data

########  INPUT ########

# Important:
NUMBER_OF_EXPERIMENTS_CREATED = 10  # Must be equal to the value of NUMBER_OF_EXPERIMENTS in experiments_generator.py
NUMBER_OF_TESTS_PER_EXPERIMENT = 3  # Must be equal to the value of NUMBER_OF_TESTS_PER_EXPERIMENT in experiments_generator.py

# App Behavior:
SECONDS_AFTER_INITIAL_IMAGE = 5
SECONDS_AFTER_MIDDLE_IMAGE = 0.5
SECONDS_BETWEEN_TESTS = 3

# Debug
SHOW_STATES = False


########################


def local_css(file_name):
    """
    This method loads a css file and apply a style in the app.
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def read_markdown_file(markdown_file):
    """
    return the text of a markdown file.
    Used for the text of the instructions, ready, end part.
    """
    return Path(markdown_file).read_text()


def id_generator(size=32, chars=string.ascii_letters + string.digits):
    """
    Creates and return a random string to be used as an id
    for a user of the behavioral study.
    """
    return ''.join(random.choice(chars) for _ in range(size))


def assign_experience_number():
    """ 
    Assign a random experiment to a user
    """
    if not "exp_number" in st.session_state:
        exp_number = random.randrange(NUMBER_OF_EXPERIMENTS_CREATED)
        st.session_state['exp_number'] = exp_number
    return


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1)
def load_experiment_metadata(exp_number):
    """
    Load the metadata (.csv) of experiment number "exp_number" 
    from the directory experiments_data/

    return the metadata as a python dictionnary
    """
    folder_name = 'experiments_data'
    file_name = folder_name + '/' + str(exp_number) + '/' + 'experiment_' + str(exp_number) + '.csv'
    df = pd.read_csv(file_name)
    df = df.drop(columns=['Unnamed: 0', 'Alpha possible range'])
    dict_records = df.to_dict(orient='records')
    return dict_records


def show_answer_buttons():
    """
    Display the buttons for the user to answer.
    """

    but_col77, but_col0, but_col1, but_col2, but_col3, but_col4 = st.columns([0.8, 1, 1, 1, 1, 1])
    global but0place, but1place, but2place, but3place, but4place

    with but_col0:
        but0place = st.empty()
        but0place.button("None", key="button0", on_click=state_change)
    with but_col1:
        but1place = st.empty()
        but1place.button("1", key="button1", on_click=state_change)
    with but_col2:
        but2place = st.empty()
        but2place.button("2", key="button2", on_click=state_change)
    with but_col3:
        but3place = st.empty()
        but3place.button("3", key="button3", on_click=state_change)
    with but_col4:
        but4place = st.empty()
        but4place.button("4", key="button4", on_click=state_change)


def show_test_images(test_number):
    """
    Show the experiment images of test number "test_number" of the experiment.
    """
    folder_name = 'experiments_data'
    exp_number = st.session_state['exp_number']
    file_name = folder_name + '/' + str(exp_number) + '/' + 'experiment_' + str(exp_number) + '_images.pkl'

    # Images:
    im_col0, im_col1, im_col2 = st.columns([1, 3.8, 1])

    with open(file_name, 'rb') as file:
        with im_col1:
            global image_box_1
            image_box_1 = st.empty()
            images_dict = pkl.load(file)
            image_box_1.image(images_dict[str(test_number)]["Initial Combined Image"], use_column_width=True)
            time.sleep(SECONDS_AFTER_INITIAL_IMAGE)
            image_box_1.image(images_dict[str(test_number)]["Middle Image"], use_column_width=True)
            time.sleep(SECONDS_AFTER_MIDDLE_IMAGE)
            image_box_1.image(images_dict[str(test_number)]["Final Combined Image"], use_column_width=True)
            global start_time
            start_time = time.time()


def show_practice_images():
    """
    Show practice images for the practice part of the behavioral study.
    """
    folder = 'behavioral_study_utils/practice_images/'
    im_col0, im_col1, im_col2 = st.columns([1, 3.8, 1])
    with im_col1:
        global image_box_0
        image_box_0 = st.empty()
        image_box_0.image(folder + 'practice_0.jpeg', use_column_width=True)
        time.sleep(SECONDS_AFTER_INITIAL_IMAGE)
        image_box_0.image(folder + 'practice_1.jpeg', use_column_width=True)
        time.sleep(SECONDS_AFTER_MIDDLE_IMAGE)
        image_box_0.image(folder + 'practice_2.jpeg', use_column_width=True)


def state_change():
    """
    This method control the states of the app when an action 
    like pressing a button is done.
    """
    if st.session_state['app_state'] == "Instructions":
        global text_place, button_pr_place
        text_place = st.empty()
        button_pr_place = st.empty()
        movie_place = st.empty()
        st.session_state['app_state'] = "Practice"

    elif st.session_state['app_state'] == "Form":
        st.session_state['age'] = st.session_state['age_input']
        st.session_state['gender'] = st.session_state['gender_input']
        st.session_state['app_state'] = "Instructions"

    elif st.session_state['app_state'] == "Practice":
        but0place.empty()
        but1place.empty()
        but2place.empty()
        but3place.empty()
        but4place.empty()
        global image_box_0
        image_box_0 = st.empty()
        st.session_state['app_state'] = "BeforeTest"

    elif st.session_state['app_state'] == "BeforeTest":
        global text_place1, button_start_place
        text_place1 = st.empty()
        button_start_place = st.empty()
        st.session_state['app_state'] = "Test"

    elif st.session_state['app_state'] == "Test":

        cur_test = st.session_state['current_test_number']

        st.session_state['results'][cur_test] = {}
        if st.session_state['button0']:
            st.session_state['results'][cur_test]['user_choice'] = 0
        elif st.session_state['button1']:
            st.session_state['results'][cur_test]['user_choice'] = 1
        elif st.session_state['button2']:
            st.session_state['results'][cur_test]['user_choice'] = 2
        elif st.session_state['button3']:
            st.session_state['results'][cur_test]['user_choice'] = 3
        elif st.session_state['button4']:
            st.session_state['results'][cur_test]['user_choice'] = 4

        global end_time
        end_time = time.time()
        time_diff = end_time - start_time
        st.session_state['results'][cur_test]['response_time'] = time_diff

        exp_metadata = load_experiment_metadata(st.session_state['exp_number'])
        test_metadata = exp_metadata[cur_test]
        st.session_state['results'][cur_test].update(test_metadata)

        but0place.empty()
        but1place.empty()
        but2place.empty()
        but3place.empty()
        but4place.empty()

        image_box_1 = st.empty()
        next_test = cur_test + 1
        st.session_state['current_test_number'] = next_test
        del st.session_state['images_shown']

        if next_test == NUMBER_OF_TESTS_PER_EXPERIMENT:
            st.session_state['app_state'] = "Save"

        time.sleep(SECONDS_BETWEEN_TESTS)



####################

#Actions depending on the states:

# Initialization
if "user_id" not in st.session_state:
    st.session_state['user_id'] = id_generator()

if "app_state" not in st.session_state:
    st.session_state['app_state'] = "Form"

# Form:
if st.session_state['app_state'] == "Form":
    st.title("Face Edits Perception Test")
    st.write('Please enter the following informations:')

    with st.form("my_form"):
        age = st.slider('Age', 0, 130, 0, key='age_input')
        gender = st.selectbox('Gender', ('Female', 'Male', 'Other', ''), key='gender_input', index=3)
        continue_but = st.form_submit_button(label="Continue", on_click=state_change)

# Instructions:
if st.session_state['app_state'] == "Instructions":
    st.title("Face Edits Perception Test")

    global text_place, button_pr_place, movie_place
    text_place = st.empty()
    button_pr_place = st.empty()
    movie_place = st.empty()

    video_file = open('behavioral_study_utils/one_test_example.mp4', 'rb')
    video_bytes = video_file.read()
    movie_place.video(video_bytes)

    instr_markdown = read_markdown_file("behavioral_study_utils/instructions.md")
    text_place.markdown(instr_markdown, unsafe_allow_html=True)
    button_pr_place.button("PRACTICE", on_click=state_change)

# Practice:
if st.session_state['app_state'] == "Practice":
    st.set_page_config(layout="wide")
    show_practice_images()
    show_answer_buttons()

# Before Test:
if st.session_state['app_state'] == "BeforeTest":
    global text_place1, button_start_place
    text_place1 = st.empty()
    button_start_place = st.empty()
    ready_markdown = read_markdown_file("behavioral_study_utils/ready.md")
    text_place1.markdown(ready_markdown, unsafe_allow_html=True)

    button_start_place.button("START", on_click=state_change)

# Test:
if st.session_state['app_state'] == "Test":

    if not 'exp_number' in st.session_state:
        assign_experience_number()
    if not 'current_test_number' in st.session_state:
        st.session_state['current_test_number'] = 0
    if not 'results' in st.session_state:
        st.session_state['results'] = {}

    #     st.subheader("test number" + str(st.session_state['current_test_number']))
    st.progress((st.session_state['current_test_number'] + 1) / NUMBER_OF_TESTS_PER_EXPERIMENT)
    if not 'images_shown' in st.session_state:
        show_test_images(st.session_state['current_test_number'])
        st.session_state['images_shown'] = True

    show_answer_buttons()

# Save results:
if st.session_state['app_state'] == "Save":
    del st.session_state['button0']
    del st.session_state['button1']
    del st.session_state['button2']
    del st.session_state['button3']
    del st.session_state['button4']

    data = {}
    for key in st.session_state:
        data[key] = st.session_state[key]

    folder = 'behavioral_study_results/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    date = datetime.now(pytz.timezone('Europe/Paris')).strftime("%Y_%m_%d-%I_%M_%S_%p")
    user_id = st.session_state['user_id']
    with open(folder + date + '_' + user_id + '.pkl', 'wb') as output_file:
        pkl.dump(data, output_file)

    st.session_state['app_state'] = "End"

# End:
if st.session_state['app_state'] == "End":
    text_place2 = st.empty()
    end_markdown = read_markdown_file("behavioral_study_utils/end.md")
    text_place2.markdown(end_markdown, unsafe_allow_html=True)

if (st.session_state['app_state'] == "Test") or (st.session_state['app_state'] == "Practice"):
    local_css("behavioral_study_utils/style.css")

if SHOW_STATES:
    "st session state: ", st.session_state
