import sqlite3

DB = "insightforge.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset TEXT,
        target TEXT,
        problem_type TEXT,
        best_model TEXT,
        best_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def save_run(dataset, target, problem_type, best_model, best_score):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO runs (dataset, target, problem_type, best_model, best_score) VALUES (?, ?, ?, ?, ?)",
        (dataset, target, problem_type, best_model, best_score)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT dataset, target, problem_type, best_model, best_score, created_at FROM runs ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows
