# InsightForge AI

**A multi-agent autonomous data science platform that runs entirely on your own machine — no paid APIs, no data leaving your laptop.**

Upload a CSV, pick a target column, and InsightForge runs a full pipeline of specialized agents that assess data quality, explore the data, engineer features, train and evaluate models, generate visualizations, and finally write a plain-English business report — all orchestrated with LangGraph and powered by a local LLM through Ollama.

---

## What It Does

InsightForge automates the repetitive parts of an early-stage data science project. Each stage is handled by a dedicated agent, and the workflow routes itself dynamically based on whether the problem is classification or regression.

- **Data Quality Analysis** — missing values, types, duplicates, and basic health checks
- **Exploratory Data Analysis (EDA)** — summary statistics and distributions
- **Visualizations** — missing-values chart, correlation heatmap, and target distribution, generated early from the raw data; feature-importance plots later from the trained model
- **Feature Engineering Recommendations** — encoding, scaling, imputation, and drop candidates
- **Automated Model Training** — multiple models trained and compared
- **Model Evaluation** — full metrics for both classification and regression
- **AI-Generated Insights** — a local LLM turns the raw results into business-readable insights (and degrades gracefully if the LLM is unavailable)
- **Run Memory** — every run is stored in SQLite so history persists across sessions
- **Tool Calling & MCP** — the model can query your data and run history through tools, and read files through a standard Model Context Protocol server

---

## Architecture

A dataset flows through a chain of agents. The Planner decides the problem type, and the graph branches accordingly before converging again for evaluation and reporting.

```
            Dataset Upload (CSV)
                    │
                    ▼
              Planner Agent  ──►  determines problem type
                    │
                    ▼
            Data Quality Agent
                    │
                    ▼
                EDA Agent
                    │
                    ▼
            Visualization Agent  ──►  missing values, correlation, target dist
                    │
                    ▼
         Feature Engineering Agent
                    │
                    ▼
              Dynamic Routing
              ┌──────┴───────┐
              ▼              ▼
       Classification    Regression
        Model Agent      Model Agent
              └──────┬───────┘
                     ▼
              Evaluation Agent
                     │
                     ▼
        Model Visualization Agent  ──►  feature importance (Random Forest)
                     │
                     ▼
               Insight Agent   ──►  Local LLM (Ollama)
                     │
                     ▼
               Report Agent
```

The planner sets `problem_type` from the target's cardinality (`nunique <= 10` → classification, else regression); a LangGraph conditional edge routes to the matching model agent, and both branches rejoin at evaluation.

Three capabilities cut across the whole platform:

- **Memory layer (SQLite)** — records each run's dataset, target, best model, and score
- **Tool calling** — the local model can call functions (dataset stats, SQL over run history)
- **MCP integration** — the model can also use a standard MCP filesystem server, scoped to `data/`

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Core | Python 3.14, Pandas, NumPy, Scikit-Learn |
| Orchestration | LangGraph |
| Interface | Streamlit |
| Local AI | Ollama + Llama 3.1 (8B) |
| Charts | Matplotlib |
| Storage | SQLite (stdlib) |
| Integration | Model Context Protocol (MCP) |

Everything runs locally. There are no API keys and no per-token costs.

---

## Project Structure

```
InsightForge-AI/
│
├── app.py                          # Streamlit UI (tabs, Past Runs, Q&A boxes)
│
├── workflow/
│   ├── state.py                    # shared AgentState object
│   └── graph.py                    # LangGraph nodes, edges, and routing
│
├── agents/
│   ├── planner_agent.py            # decides classification vs regression
│   ├── data_quality_agent.py       # rows, columns, missing, duplicates
│   ├── eda_agent.py                # summary statistics
│   ├── visualization_agent.py      # missing values, correlation, target dist
│   ├── feature_engineering_agent.py# encoding/scaling/imputation recommendations
│   ├── classification_model_agent.py
│   ├── regression_model_agent.py
│   ├── evaluation_agent.py         # accuracy/precision/recall/f1 | MAE/RMSE/R²
│   ├── model_visualization_agent.py# feature importance (Random Forest)
│   ├── insight_agent.py            # local-LLM business insights
│   └── report_agent.py             # assembles the final text report
│
├── utils/
│   ├── llm.py                      # Ollama chat wrapper (configure the model here)
│   ├── db.py                       # SQLite memory layer
│   ├── tools.py                    # tool-calling Q&A (column stats + SQL)
│   └── mcp_client.py               # MCP filesystem client
│
├── assets/                         # example charts used in this README
├── data/                           # your CSVs (gitignored)
├── reports/visualizations/         # generated charts (gitignored)
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- **Python 3.14** (the project venv targets 3.14)
- **[Ollama](https://ollama.com)** — runs the local model
- **Node.js / npx** — only needed for the MCP feature (the filesystem server launches via `npx`)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/InsightForge-AI.git
cd InsightForge-AI

python3.14 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

> This project pins to a Python 3.14 virtual environment. Invoke tools via `./venv/bin/python ...` rather than relying on shell activation.

### 2. Set up the local model

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
```

The model is configured in one place — change the model name in `utils/llm.py` to swap it (e.g. `qwen2.5:14b` for more reliable tool selection on stronger hardware).

### 3. Run the app

```bash
./venv/bin/python -m streamlit run app.py
```

The app opens in your browser (default http://localhost:8501). Upload a CSV, choose the target column, and run the pipeline.

---

## Usage

1. **Upload a CSV** — sample datasets like Titanic or Ames House Prices work great.
2. **Select the target column** — a banner shows the detected problem type.
3. **Explore the tabs** — Dataset, Report, Feature Engineering, Model & Evaluation, Visualizations, AI Insights.
4. **Review Past Runs** — every analysis is saved automatically and listed.
5. **Use the Q&A boxes:**
   - *"Ask a question about your data or past runs"* → column stats + run-history SQL, via Ollama tool-calling.
   - *"Ask via MCP"* → file questions answered through the filesystem MCP server.

> Local-LLM note: insight generation and Q&A make on-device calls to `llama3.1:8b` (~4.9 GB). Expect ~30–100s per LLM response depending on dataset size, and run **one LLM request at a time**.

### Datasets Tested

| Type | Dataset | Target |
|------|---------|--------|
| Classification | Titanic | `Survived` |
| Classification | Customer Churn | `Churn` |
| Regression | House Prices | `SalePrice` |

---

## Example Output

Charts are generated automatically by the visualization agents (shown here for the House Prices dataset):

| Correlation heatmap | Feature importance (Random Forest) |
|---|---|
| ![Correlation heatmap](assets/correlation_heatmap.png) | ![Feature importance](assets/feature_importance.png) |

---

## How It Works

Each agent reads from and writes to a shared `AgentState` object, so every stage builds on the last:

- **Planner Agent** inspects the target column and sets `problem_type` to `classification` or `regression`, which drives the conditional routing.
- **Data Quality, EDA, and Feature Engineering Agents** profile the dataset and produce structured reports (missing values, distributions, encoding/scaling recommendations).
- **Visualization Agent** draws missing-values, correlation, and target-distribution charts from the raw data; the **Model Visualization Agent** later plots feature importance from the trained Random Forest.
- **Model Agents** train and compare several models (Logistic/Linear Regression, Decision Tree, Random Forest) and select the best by accuracy (classification) or R² (regression).
- **Evaluation Agent** computes the right metrics for the problem type — accuracy, precision, recall, F1 for classification; MAE, RMSE, R² for regression.
- **Insight Agent** sends the structured reports to the local LLM and gets back a short, plain-English summary aimed at a non-technical reader.
- **Report Agent** assembles everything into a final report.

The model can also reach beyond the pipeline through **tool calling** (functions defined in `utils/tools.py`) and an **MCP filesystem server** (via `utils/mcp_client.py`), letting it read files in `data/` and query the run-history database — all through one consistent, model-agnostic interface.

---

## Version History

The project was built incrementally, each version adding one capability.

| Version | Milestone | What it added |
|---------|-----------|---------------|
| **v1.0.0** | Multi-agent foundation | Planner, Data Quality, EDA, and Report agents; CSV upload and report generation |
| **v1.1.0** | Feature Engineering | Encoding, scaling, imputation, and drop-candidate recommendations |
| **v1.2.0** | Model Agent | Classification models with train/test split, comparison, and best-model selection |
| **v1.3.0** | Evaluation Agent | Full classification metrics and confusion matrix |
| **v2.0.0** | Dynamic Routing | User-selected target, regression support, and conditional classification/regression branching in LangGraph |
| **v2.1.0** | Visualization Agents | Automated plots from raw data and model results |
| **v3.0.0** | Local AI | Ollama + Llama 3.1 and the Insight Agent for natural-language business insights |
| **v3.0.1** | Resilient insights | Graceful handling when the local LLM is unavailable |
| **v4.0.0** | Memory Layer | SQLite storage of dataset history, runs, and best-model history |
| **v5.0.0** | Tool Calling | Local model can call functions (dataset stats, SQL over run history) |
| **v6.0.0** | MCP Integration | Connects to a standard MCP filesystem server through an MCP client |

---

## Notes & Limitations

- **Runs locally by design.** Because inference happens through a local Ollama model, the app can't be deployed to hosting that has no Ollama server (e.g. free Streamlit Cloud). This is intentional — it keeps data private and costs at zero. A short demo video or screenshots are the best way to share it.
- **No data is committed.** `data/*` and generated artifacts (`reports/visualizations/*.png`, `insightforge.db`) are gitignored.
- **Tool reliability scales with model size.** Small models handle single, clearly-scoped tool calls well but can struggle with complex multi-tool reasoning. Swap to a larger model in `utils/llm.py` if needed.
- **MCP server scope.** The filesystem server is scoped to the `data/` folder on purpose — agents are never given access to the whole disk. It handles file operations (read/list), not data analysis; use the data Q&A box for analytical questions.

---

## Roadmap Ideas

The core platform is complete. Possible future work:

- Lightweight unit tests around the agents
- Pinned dependency versions for reproducibility
- Additional MCP servers (Postgres, web search)
- Caching of LLM responses to avoid repeat inference on reruns

---

*Built as a from-scratch, fully local multi-agent data science platform — no paid APIs.*
