import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualization_agent(state):

    df = state["dataframe"]

    visualizations = {}

    # Create chart for missing values:

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    # Plot:

    plt.figure(figsize=(8, 4))

    missing.sort_values().plot.barh()

    plt.title("Missing Values")

    plt.tight_layout()

    # Save:

    path = "reports/visualizations/missing_values.png"

    plt.savefig(path)

    plt.close()

    # Store:

    visualizations["missing_values"] = path

    # Create Correlation Heatmap:

    numeric_df = df.select_dtypes(include=["number"])

    # Correlation:

    corr = numeric_df.corr()

    # Plot:

    plt.figure(figsize=(10, 8))

    plt.imshow(corr)

    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    # Save:

    corr_path = "reports/visualizations/correlation.png"

    plt.savefig(corr_path)

    plt.close()

    # Store:

    visualizations["correlation"] = corr_path

    # Target Distribution Plot:

    target = state["target_column"]

    plt.figure(figsize=(8, 4))

    if state["problem_type"] == "classification":
        df[target].value_counts().plot.bar()

    else:
        df[target].hist()

    plt.title("Target Distribution")

    plt.tight_layout()

    # Save:

    target_path = "reports/visualizations/target_distribution.png"

    plt.savefig(target_path)

    plt.close()

    # Store Path:

    visualizations["target"] = target_path

    state["visualization_report"] = visualizations

    return state
