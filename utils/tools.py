import sqlite3
import ollama

from utils.llm import MODEL
from utils.db import DB

# app.py sets this to the uploaded dataframe before asking a question

df = None

def column_stats(column: str) -> str:
    """Get basic statistics (count, mean, min, max etc.) for a column in the dataset."""
    # case-insensitive match so the model doesn't have to get the casing exact
    cols = {c.lower(): c for c in df.columns}
    actual = cols.get(column.lower())
    if actual is None:
        return f"Column '{column}' not found. Available columns: {list(df.columns)}"
    return df[actual].describe().to_string()

def run_sql(query: str) -> str:
    """Run a SQL query on the run-history database. The table is called 'runs'
    with columns: dataset, target, problem_type, best_model, best_score, created_at."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(query).fetchall()
    conn.close()
    return str(rows)

# the tools the model is allowed to use
available = {"column_stats": column_stats, "run_sql": run_sql}

def ask(question):
    messages = [{"role": "user", "content": question}]
    res = ollama.chat(model=MODEL, messages=messages, tools=list(available.values()))

    # if the model didn't need a tool, just return its answer
    if not res.message.tool_calls:
        return res.message.content
    
    # run whatever tool the model picked and send the result back

    messages.append(res.message)
    for call in res.message.tool_calls:
        try:
            result = available[call.function.name](**call.function.arguments)
        except Exception as e:
            # feed the error back to the model instead of crashing the app
            result = f"Tool error: {e}"
        messages.append({"role": "tool", "content": str(result), "tool_name": call.function.name})

    final = ollama.chat(model=MODEL, messages=messages)
    return final.message.content

