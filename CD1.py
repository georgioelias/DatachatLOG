import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io
import re
from collections import Counter
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import uuid
from datetime import datetime


# Constants
DB_NAME = 'data.db'
FIXED_TABLE_NAME = "uploaded_data"
csv_file_path = "Data18sep.csv"
explanation_file_path= "QueriesDescription.txt"
gcreds = st.secrets["gcp_service_account"]
# OpenAI API setup

API_KEYS = [
    st.secrets["OPENAI_API_KEY"],
    st.secrets["OPENAI_API_KEY_2"],
    st.secrets["OPENAI_API_KEY_3"],
]
    
MODELS = ["gpt-4o", "gpt-4o","gpt-4o"]


# Google Sheets API setup
def authenticate_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Use the service account credentials for authentication
    creds = ServiceAccountCredentials.from_json_keyfile_name(gcreds, scope)
    client = gspread.authorize(creds)
    
    # Open the sheet (replace 'Your Google Sheet Name' with your actual sheet name)
    sheet = client.open('ChatData LOG').sheet1
    return sheet


def append_to_google_sheet(chat_id, user_input, json_answer, sql_query, gpt_response):
    sheet = authenticate_google_sheets()
    
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the data to append
    row_data = [chat_id,current_time, user_input, json_answer, sql_query, gpt_response]
    
    # Append the row to the Google Sheet
    sheet.append_row(row_data)


# Global variables for prompts
if 'sql_generation_prompt' not in st.session_state:
    st.session_state.sql_generation_prompt = '''
    User's explanation of the CSV:
    {csv_explanation}

    A user will now chat with you. Your task is to transform the user's request into an SQL query that retrieves exactly what they are asking for.

    Rules:
    1. Return only two JSON variables: "Explanation" and "SQL".
    2. No matter how complex the user question is, return only one SQL query.
    3. Always return the SQL query in a one-line format.
    4. Consider the chat history when generating the SQL query.
    5. The query can return multiple rows if appropriate for the user's question.
    6. You shall not use functions like MONTH(),HOUR(),HOUR,YEAR(),DATEIFF(),.....
    7. Use only queries proper for sql LITE
    8.⁠ ⁠YOU CAN ONLY RETURN ONE SQL STATEMENT AT A TIME, COMBINE YOUR ANSWER IN ONLY ONE STATEMENT, NEVER 2 or MORE, Find workarounds.
    9.Ignore Null values in interpretations and calculations only consider them where they are relevent.

    Example output:
    {{
    "Explanation": "The user is asking about the top 5 users by age. To retrieve this, we need to select the name and age columns, order by age descending, and limit to 5 results.",
    "SQL": "SELECT name, age FROM {table_name} ORDER BY age DESC LIMIT 5"
    }}

    Your prompt ends here. Everything after this is the chat with the user. Remember to always return the accurate SQL query.
    '''

if 'response_generation_prompt' not in st.session_state:
    st.session_state.response_generation_prompt = '''
    User's explanation of the CSV:
    {csv_explanation}

    Now you will receive a JSON containing the SQL output that answers the user's inquiry. The output may contain multiple rows of data. Your task is to use the SQL's output to answer the user's inquiry in plain English. Consider the chat history when generating your response. If there are multiple results, summarize them appropriately.
    '''

############################################### HELPER FUNCTIONS ########################################################

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def get_data_type(values):
    if all(isinstance(val, (int, float)) for val in values if pd.notna(val)):
        return "Number"
    elif all(isinstance(val, str) for val in values if pd.notna(val)):
        return "Text"
    elif all(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in values if pd.notna(val)):
        return "Date"
    else:
        return "Mixed"

def analyze_csv(file_path, max_examples=3):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    
    prompt = "This CSV file contains the following columns:\n\n"
    
    for col in df.columns:
        values = df[col].dropna().tolist()
        data_type = get_data_type(values)
        
        unique_count = df[col].nunique()
        total_count = len(df)
        is_unique = unique_count == total_count
        
        examples = df[col].dropna().sample(min(max_examples, len(values))).tolist()
        
        prompt += f"Column: {col}\n"
        prompt += f"Data Type: {data_type}\n"
        prompt += f"Examples: {', '.join(map(str, examples))}\n"
        
        if is_unique:
            prompt += "Note: This column contains unique values for each row.\n"
        
        null_count = df[col].isnull().sum()
        if null_count > 0:
            prompt += f"Note: This column contains {null_count} NULL values.\n"
        
        if data_type == "Text":
            value_counts = Counter(values)
            most_common = value_counts.most_common(3)
            if len(most_common) < len(value_counts):
                prompt += f"Most common values: {', '.join(f'{val} ({count})' for val, count in most_common)}\n"
        
        prompt += "\n"
    
    return prompt

def reset_chat():
    st.session_state.messages = []

def display_sql_query(query):
    with st.expander("View SQL Query", expanded=False):
        st.code(query, language="sql")

def display_json_data(json_data):
    with st.expander("View JSON Data", expanded=False):
        if isinstance(json_data, list):
            for item in json_data:
                st.json(item)
        else:
            st.json(json_data)

def df_to_sqlite(df, table_name, db_name=DB_NAME):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred while creating the table: {e}")
        return False

# New function to update prompts
def update_prompt(prompt_type):
    if prompt_type == "SQL Generation":
        st.session_state.sql_generation_prompt = st.session_state.sql_generation_prompt_input
    elif prompt_type == "Response Generation":
        st.session_state.response_generation_prompt = st.session_state.response_generation_prompt_input

############################################## AI INTERACTION FUNCTIONS ######################################################

def try_api_call(func, *args, **kwargs):
    for api_key in API_KEYS:
        for model in MODELS:
            try:
                client = OpenAI(api_key=api_key)
                return func(client, model, *args, **kwargs)
            except Exception as e:
                print(f"Error with API key {api_key[:5]}... and model {model}: {str(e)}")
    return None  # If all combinations fail

def rephrase_query(user_input, attempt):
    # Slight modifications to the query based on attempt
    if attempt == 1:
        return user_input + " (simplified)"
    elif attempt == 2:
        return "Please find relevant data: " + user_input
    elif attempt == 3:
        return "Extract from data: " + user_input
    else:
        return user_input

def generate_sql_query(user_input, prompt, chat_history, retry_attempts=10):
    attempt = 0
    response = None

    while attempt < retry_attempts:
        attempt += 1
        try:
            # Limit the chat history context to a few latest messages
            reduced_chat_history = chat_history[-3:]

            def api_call(client, model, user_input, prompt, chat_history):
                messages = [
                    {"role": "system", "content": prompt},
                ]

                for message in chat_history:
                    messages.append({"role": message["role"], "content": message["content"]})

                messages.append({"role": "user", "content": user_input})

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    n=1,
                    stop=None,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()

            # Modify the input based on attempt number
            modified_input = rephrase_query(user_input, attempt)
            response = try_api_call(api_call, modified_input, prompt, reduced_chat_history)

            if response:
                try:
                    sql_data = json.loads(response)
                    if "SQL" in sql_data:
                        return response  # Success on generating valid SQL query
                except json.JSONDecodeError:
                    st.warning(f"Attempt {attempt}: Failed to generate a valid SQL query response. Retrying...")

        except Exception as e:
            st.warning(f"Attempt {attempt}: Encountered an error: {str(e)}. Retrying...")

    st.error("Failed to generate a valid SQL query after multiple attempts. Please try rephrasing your question.")
    return None


def execute_query_and_save_json(input_string, table_name, db_name=DB_NAME, retry_attempts=3):
    try:
        sql_data = json.loads(input_string)
        sql_query = sql_data["SQL"]
    except json.JSONDecodeError:
        st.error("Failed to parse SQL query response.")
        return None

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    attempt = 0
    results = None
    column_names = None

    while attempt < retry_attempts:
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            break  # If the query is successful, exit the loop
        except sqlite3.Error as e:
            attempt += 1
            st.warning(f"Query failed on attempt {attempt}. Retrying...")
            if attempt >= retry_attempts:
                st.error(f"Failed to execute the query after {retry_attempts} attempts: {e}")
                return None
        finally:
            conn.close()

    # If we successfully got the results, process and save them
    if results is not None and column_names is not None:
        result_list = []
        for row in results:
            result_dict = {column_names[i]: row[i] for i in range(len(column_names))}
            result_list.append(result_dict)

        with open('query_result.json', 'w') as json_file:
            json.dump(result_list, json_file, indent=2)
        
        return result_list
    else:
        return None

def generate_response(json_data, prompt, chat_history):
    def api_call(client, model, json_data, prompt, chat_history):
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
        
        messages.append({"role": "user", "content": f"JSON data: {json_data}"})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    
    return try_api_call(api_call, json_data, prompt, chat_history)


# Load data function (modified to work in the backend)
@st.cache_data
def load_data(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        csv_analysis = analyze_csv(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None

    if not df_to_sqlite(df, FIXED_TABLE_NAME):
        return None, None, None
    
    return df, FIXED_TABLE_NAME, csv_analysis

def load_explanation_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"An error occurred while reading the explanation file: {e}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="DataChat", page_icon="📈")

    # Generate a unique Chat ID for each session
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())

    # Reset chat button at the top of the page
    if st.button("Reset Chat", key="reset_top"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    st.title("Data Chat Application")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'csv_explanation' not in st.session_state:
        st.session_state.csv_explanation = ""

    # Load data in the backend
    if not os.path.exists(csv_file_path):
        st.error(f"CSV file not found: {csv_file_path}")
        return

    df, table_name, csv_analysis = load_data(csv_file_path)
    if df is None:
        st.error("Failed to load the CSV file. Please check the file and try again.")
        return

    # Choose explanation source in the code
    use_csv_analysis = False  # Set this to False to use the text file instead

    if use_csv_analysis:
        st.session_state.csv_explanation = csv_analysis
    else:
        if not os.path.exists(explanation_file_path):
            st.error(f"Explanation file not found: {explanation_file_path}")
        else:
            explanation_text = load_explanation_from_file(explanation_file_path)
            if explanation_text is not None:
                st.session_state.csv_explanation = explanation_text
            else:
                st.error("Failed to load the explanation text file. Please check the file and try again.")

    st.header("Chat with your data")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the data")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generating response..."):
            sql_generation_prompt = st.session_state.sql_generation_prompt.format(csv_explanation=st.session_state.csv_explanation, table_name=table_name)
            sql_query_response = generate_sql_query(user_input, sql_generation_prompt, st.session_state.messages[:-1])
            
            if sql_query_response is None:
                st.error("Failed to generate a response. Please try again later.")
            else:
                try:
                    sql_data = json.loads(sql_query_response)
                    sql_query = sql_data["SQL"]
                    
                    result_list = execute_query_and_save_json(sql_query_response, table_name)

                    if result_list:
                        response_generation_prompt = st.session_state.response_generation_prompt.format(csv_explanation=st.session_state.csv_explanation)
                        response = generate_response(json.dumps(result_list), response_generation_prompt, st.session_state.messages)

                        if response is None:
                            st.error("Failed to generate a response. Please try again later.")
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            with st.chat_message("assistant"):
                                st.markdown(response)
                           
                    else:
                        st.error("Failed to execute the SQL query after multiple attempts. Please try rephrasing your question.")

                except json.JSONDecodeError:
                    st.error("Failed to generate a valid SQL query. Please try rephrasing your question.")
        append_to_google_sheet(st.session_state.chat_id, user_input, json.dumps(result_list), sql_query, response)
if __name__ == "__main__":
    main()
