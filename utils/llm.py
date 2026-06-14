
import ollama

# change the model here once and it applies everywhere

MODEL = "llama3.1:8b"

def ask(prompt):
    res = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]
