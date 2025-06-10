# train a simple classification model to predict if a repo is developer tooling or not
from sqlalchemy import text
import sqlalchemy
import psycopg2
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- Database Connection and Feature Retrieval ---
cloud_sql_user = os.getenv("cloud_sql_user")
cloud_sql_password = os.getenv("cloud_sql_password")
cloud_sql_postgres_host = os.getenv("cloud_sql_postgres_host")
cloud_sql_postgres_db = os.getenv("cloud_sql_postgres_db")

# Construct the connection string
conn_str = (
    f"postgresql+psycopg2://"
    f"{cloud_sql_user}:{cloud_sql_password}@"
    f"{cloud_sql_postgres_host}/{cloud_sql_postgres_db}"
)

# Create the SQLAlchemy engine
try:
    cloud_sql_engine = create_engine(conn_str)
except Exception as e:
    raise Exception(f"Error creating database engine: {e}")


# Fetch the feature dataset from the database
try:
    with cloud_sql_engine.connect() as conn:
        query = text(
            """
            SELECT 
                repo,
                has_readme,
                is_collection_of_learnings,
                has_app_application,
                is_awesome_curated,
                has_benchmark,
                is_block_explorer,
                is_boilerplate_scaffold_template,
                is_bootcamp,
                is_bot,
                has_bounty_program,
                has_brand_icon_logo,
                is_cli_tool,
                is_library,
                is_course,
                is_demo,
                has_docs,
                is_education_related,
                is_eip_erc,
                has_examples,
                is_feature_description,
                is_starter_project,
                is_guide,
                is_hackathon_project,
                is_hello_world,
                uses_json_rpc,
                is_interview_related,
                is_learning_material,
                is_mcp_server,
                is_plugin,
                is_sample_project,
                is_sdk,
                is_security_related,
                has_tests_testing,
                has_tips,
                is_tooling,
                is_tutorial,
                is_whitepaper,
                is_workshop,
                is_wrapper
            FROM clean.project_repos_features
            """
        )
        repo_features_df = pd.read_sql(query, conn)
except Exception as e:
    raise Exception(f"Error fetching data from the database: {e}")

# --- Labeled Data Retrieval ---
gsheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSTIjEmhgSpvITvd8BdnttCmGD05bylP9PDZW0WaeahdL0C2Fxfh5dZcd1-EmhbP_M2BJydgA81aKy1/pub?gid=1690796422&single=true&output=csv'
try:
    dev_tooling_df = pd.read_csv(gsheet_url)
except Exception as e:
    raise Exception(f"Error reading data from Google Sheets: {e}")

# --- Data Preparation ---
# Merge the feature data with the labeled data
merged_df = pd.merge(repo_features_df, dev_tooling_df, on='repo')

# print info about the merged dataframe
print(f"Merged dataframe info: {merged_df.info()}")

# print the first 5 rows of the merged dataframe
print(f"Merged dataframe first 5 rows: {merged_df.head()}")

# print the number of rows where is_dev_tooling is 1
print(f"Number of rows where is_dev_tooling is 1: {merged_df[merged_df['is_dev_tooling'] == 1].shape[0]}")

# print the number of rows where is_dev_tooling is 0
print(f"Number of rows where is_dev_tooling is 0: {merged_df[merged_df['is_dev_tooling'] == 0].shape[0]}")

# print the number of rows where is_dev_tooling is null
print(f"Number of rows where is_dev_tooling is null: {merged_df[merged_df['is_dev_tooling'].isnull()].shape[0]}")

# Separate features (X) and the target variable (y)
feature_columns = [
    'has_readme',
    'is_collection_of_learnings', 
    'has_app_application', 
    'is_awesome_curated', 
    'has_benchmark', 
    'is_block_explorer', 
    'is_boilerplate_scaffold_template', 
    'is_bootcamp', 
    'is_bot', 
    'has_bounty_program', 
    'has_brand_icon_logo', 
    'is_cli_tool', 
    'is_library', 
    'is_course', 
    'is_demo', 
    'has_docs', 
    'is_education_related', 
    'is_eip_erc', 
    'has_examples', 
    'is_feature_description', 
    'is_starter_project', 
    'is_guide', 
    'is_hackathon_project', 
    'is_hello_world', 
    'uses_json_rpc', 
    'is_interview_related', 
    'is_learning_material', 
    'is_mcp_server', 
    'is_plugin', 
    'is_sample_project', 
    'is_sdk', 
    'is_security_related', 
    'has_tests_testing', 
    'has_tips', 
    'is_tooling', 
    'is_tutorial', 
    'is_whitepaper', 
    'is_workshop', 
    'is_wrapper']

X = merged_df[feature_columns]

# Ensure all feature data is numeric (booleans will be treated as 0s and 1s)
X = X.astype(float)

y = merged_df['is_dev_tooling']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model Training ---
# Initialize and train a simple classification model (Logistic Regression)
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Model Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix with Labels ---
# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print("                 Predicted")
print("                 False    True")
print("Actual False    {:<8} {:<8}".format(cm[0][0], cm[0][1]))
print("       True     {:<8} {:<8}".dformat(cm[1][0], cm[1][1]))
print("\n")

# Explanation of the terms
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn} - Correctly predicted not dev tooling")
print(f"False Positives (FP): {fp} - Incorrectly predicted as dev tooling")
print(f"False Negatives (FN): {fn} - Incorrectly predicted as not dev tooling (missed)")
print(f"True Positives (TP): {tp} - Correctly predicted as dev tooling")
