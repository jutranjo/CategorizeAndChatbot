import openai
import pandas as pd
import json
import re
import os
from dotenv import load_dotenv
from datetime import datetime
import dateparser
from Chatbot.stats import describe_filtered_data

DATA_PATH = "merged_messages_with_categories.csv"


def setup_environment():
    """
    Loads environment variables and data, parses timestamps, 
    and prepares LLM client and available filter metadata.

    Returns:
        dict: A dictionary containing:
            - 'client': OpenAI client instance
            - 'df': Loaded and parsed DataFrame
            - 'current_time': Max timestamp in the data
            - 'categories': List of unique categories
            - 'sources': List of unique sources
    """
    load_dotenv()

    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    current_time = df['timestamp'].max()

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    categories = df['category'].dropna().unique().tolist()
    sources = df['source'].dropna().unique().tolist()

    return {
        "client": client,
        "df": df,
        "current_time": current_time,
        "categories": categories,
        "sources": sources
    }

def build_system_prompt(env):
    """
    Constructs the full system prompt to guide the LLM in parsing filter instructions.

    Args:
        env (dict): Environment dictionary from `setup_environment()`

    Returns:
        str: Prompt for the system role in the LLM chat API.
    """
    return (
        "You are a data assistant extracting structured filters from user queries about categorized customer support messages.\n\n"
        f"Valid categories (exact match only): {env['categories']}\n"
        f"Valid sources: {env['sources']}\n"
        f"The current datetime is {env['current_time'].isoformat()}.\n\n"
        "Return a strict JSON object with these keys:\n"
        "{\n"
        "  \"category\": (exact string from valid categories, or null),\n"
        "  \"source\": (exact string from valid sources, or null),\n"
        "  \"start_time_expr\": (human-readable date expression or null),\n"
        "  \"end_time_expr\": (human-readable date expression or null),\n"
        "  \"reset\": (true if this is a new query and filters should be cleared, otherwise false)\n"
        "}\n\n"
        "Time filtering rules:\n"
        "- Use expressions like \"1 day ago\", \"Monday\", or \"today\".\n"
        "- Avoid vague terms like \"past week\".\n\n"
        "Do not use phrases like this Monday. Instead use absolute or relative phrases like Monday, 7 days ago, or today.\n"
        "If the user does not specify an end time, set \"end_time_expr\": \"now\".\n"
        "Avoid paraphrasing expressions like \"this week\" into \"7 days ago\". Instead, use \"Monday\" to represent the start of the current week.\n"
        "If the user says something like \"start a new search\", \"new query\", \"start over\", or \"fresh search\", then:\n"
        "- Set all filter fields (category, source, time) explicitly\n"
        "- Set \"reset\": true\n\n"
        "If the user is refining the current query, using phrases like:\n"
        "\"make that\", \"change to\", \"now show\", \"just update\", \"let’s look at\", \"actually\", \"how about\", \"switch to\", \"only\", or \"instead\",\n"
        "- Only update the fields mentioned\n"
        "- Leave others as null\n"
        "- Set \"reset\": false\n\n"
        "Use \"reset\": false by default unless the user explicitly indicates a reset.\n"
        "Return ONLY the raw JSON object — no commentary.\n"
)

def query_LLM_for_filters(user_query, env):
    """
    Sends a user query to the LLM and extracts structured filter instructions.

    Args:
        user_query (str): Natural language input from the user.
        env (dict): Environment dictionary containing the LLM client and metadata.

    Returns:
        Optional[dict]: A dictionary with filter fields and a reset flag, or None if parsing fails.
    """
    prompt = build_system_prompt(env)
    
    response = env['client'].chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    reply_content = response.choices[0].message.content.strip()
    
    matches = re.findall(r'\{[\s\S]*?\}', reply_content)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            return None
    return None


def apply_filters(df, filters, current_time):
    """
    Applies category, source, and time-based filters to the message DataFrame.

    Args:
        df (pd.DataFrame): The full message dataset.
        filters (dict): Dictionary with 'category', 'source', 'start_time_expr', 'end_time_expr'.
        current_time (datetime): Reference time for relative date parsing.

    Returns:
        pd.DataFrame: A filtered copy of the original DataFrame.
    """

    filtered = df.copy()
    
    if filters.get("category"):
        filtered = filtered[filtered["category"].str.lower() == filters["category"].lower()]
        
    if filters.get("source"):
        filtered = filtered[filtered["source"].str.lower() == filters["source"].lower()]

    start_time = parse_expr(filters["start_time_expr"], current_time)
    end_time = parse_expr(filters["end_time_expr"], current_time) or dateparser.parse("now")

    if start_time:
        filtered = filtered[filtered['timestamp'] >= pd.to_datetime(start_time)]
    if end_time:
        filtered = filtered[filtered['timestamp'] <= pd.to_datetime(end_time)]
    return filtered

def parse_expr(expr: str, base_time: datetime) -> str:
    """
    Parses a human-readable time expression into an ISO-formatted timestamp string.

    Args:
        expr (str): Natural language time expression (e.g., 'yesterday', '7 days ago').
        base_time (datetime): Base time for relative parsing.

    Returns:
        Optional[str]: ISO-formatted timestamp string, or None if parsing fails.
    """
    if not expr:
        return None
    dt = dateparser.parse(expr, settings={
        'RELATIVE_BASE': base_time, 
        'RETURN_AS_TIMEZONE_AWARE': False
        })
    return dt.replace(microsecond=0).isoformat() if dt else None

def new_filter_context():
    """
    Initializes a fresh filter context dictionary.

    Returns:
        dict: A dictionary with keys 'category', 'source', 'start_time_expr', 'end_time_expr', all set to None.
    """
    return {k: None for k in ["category", "source", "start_time_expr", "end_time_expr"]}

def run_chatbot(env):
    """
    Runs the chatbot interface in a loop, handling natural language input,
    extracting filters, applying them, and printing results.

    Args:
        env (dict): Environment dictionary from `setup_environment()`.
    """
    df = env['df']
    current_time = env['current_time']
    current_filter = new_filter_context()

    print("Welcome to the message query assistant. Type 'reset' to clear filters or 'exit' to quit.")
    print(f"Valid message categories are: {env['categories']}")
    print(f"Valid message sources are: {env['sources']}")
    

    while True:
        user_query = input("Which messages do you want to see? (type 'exit' to quit, type 'reset' to reset all filtering): ")
        if user_query.lower() == 'exit':
            break
        elif user_query.lower() == 'reset':
            current_filter = new_filter_context()
            print("Filter context has been reset.")
            continue
        
        new_user_filters = query_LLM_for_filters(user_query, env)
    
        if not new_user_filters:
            print("Sorry, I couldn't understand your request.")
            continue

        if new_user_filters.get("reset"):
            current_filter = new_filter_context()

        for key, value in new_user_filters.items():
            if value is not None and key in current_filter:
                current_filter[key] = value

        filtered_df = apply_filters(df, current_filter, current_time)

        describe_filtered_data(filtered_df = filtered_df,entire_df=df)

if __name__ == "__main__":
    env = setup_environment()
    run_chatbot(env)