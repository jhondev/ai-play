import openai
import streamlit as st
import os
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI


## upload csv
## use agent to process
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_response(files, prompt):
    for file in files:
        agent = create_csv_agent(OpenAI(temperature=0), file, verbose=True)
        response = agent(prompt)

        evaluation_prompt = """
        Here is the user question '{}' and here is the AI's response '{}',
        Did the AI say something along the lines of 'i don't know'?
        Return True or False with no punctuation.
        """.format(prompt, response)

        result = openai.ChatCompletition.create(
            model="gpt4",
            messages=[
                {"role": "system", "text": "You are an evaluation AI"},
                {"role": "user", "text": evaluation_prompt}
            ])

        open_ai_response = result.choices[0].text

        if open_ai_response:
            eval_result = eval(open_ai_response)
            if not eval_result:
                return response
            else:
                continue

        return 'Agent was not able to find answer'


def save_csv(files):
    for file in files:
        with open(file.name, 'wb') as csv_file:
            csv_file.write(file.read())
        st.success('Saved file: ' + str(file))

st.title("🤖 CSV chatbot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("", key="input")
    return input_text 

user_input = get_text()

with st.form(key="my_form"):
    csv_files = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=True)
    user_input = get_text() if csv_files else None
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if csv_files and user_input:
        generate_response(csv_files, user_input)
    else:
        save_csv(csv_files)

# if user_input:
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
