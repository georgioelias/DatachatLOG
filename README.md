
# CHAT WITH YOUR DATA

## Overview

**Chat with you Data** is a Streamlit-based application that allows users to query datasets using natural language. By leveraging GPT-4, this app transforms user input into SQL queries, executes them using SQLite, and returns the results in an intuitive format. It is designed for users who want to interact with their data without needing SQL knowledge.

## Key Features

- **Natural Language Processing (NLP):** Converts user queries into SQL commands using GPT-4.
- **SQL Execution:** Runs SQL queries against an SQLite database.
- **Streamlit Interface:** User-friendly, interactive chat interface.
- **Google Sheets Logging:** Logs all interactions and results using the Google Sheets API.

## Technology Stack

- **Streamlit** for the front-end interface.
- **GPT-4 API** for query processing.
- **SQLite** for database management.
- **Google Sheets API** for logging interactions.

## Installation

### Prerequisites

- Python 3.x
- Streamlit
- OpenAI Python client
- SQLite
- Google Sheets API credentials

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/georgioelias/DatachatLOG.git
   cd DatachatLOG
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure Google Sheets API and OpenAI API key:

   - AAdd your Google Sheets credentials and OpenAI API key in `streamlit secrets`.

4. Run the application:

   ```bash
   streamlit run CD1.py
   ```

5. Upload your dataset as a CSV file, or update the file path in the code as required.

## How It Works

1. Users input a natural language question.
2. GPT-4 processes the query and generates the appropriate SQL query.
3. The SQL query is executed on an SQLite database.
4. The result is displayed in the Streamlit interface, and the interaction is logged.
