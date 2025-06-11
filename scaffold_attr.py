# train a simple classification model to predict if a repo is scaffold or not
## random forest

# run imports
from sqlalchemy import text
import sqlalchemy
import psycopg2
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
            SELECT *
            FROM clean.project_repos_features
            """
        )
        repo_features_df = pd.read_sql(query, conn)
except Exception as e:
    raise Exception(f"Error fetching data from the database: {e}")

# --- Labeled Data Retrieval ---
gsheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vShUKZQS6QFJ1SM1efqpFv-tXxbX6LFcJsc_L2MG-NtcXC-e9dGKgkbTSW39Zm6gfLIsUzkiWXa-CVE/pub?gid=1690796422&single=true&output=csv'
try:
    scaffold_df = pd.read_csv(gsheet_url)
except Exception as e:
    raise Exception(f"Error reading data from Google Sheets: {e}")

# --- Data Preparation ---
# Merge the feature data with the labeled data
merged_df = pd.merge(repo_features_df, scaffold_df, on='repo')

# drop rows where is_scaffold is null
merged_df = merged_df.dropna(subset=['is_scaffold'])

# print info about the merged dataframe
print(f"Merged dataframe info: {merged_df.info()}")

# print the first 5 rows of the merged dataframe
print(f"Merged dataframe first 5 rows: {merged_df.head()}")

# print the number of rows where is_scaffold is 1
print(f"Number of rows where is_scaffold is 1: {merged_df[merged_df['is_scaffold'] == 1].shape[0]}")

# print the number of rows where is_scaffold is 0
print(f"Number of rows where is_scaffold is 0: {merged_df[merged_df['is_scaffold'] == 0].shape[0]}")

# print the number of rows where is_scaffold is null
print(f"Number of rows where is_scaffold is null: {merged_df[merged_df['is_scaffold'].isnull()].shape[0]}")

# Separate features (X) and the target variable (y)
feature_columns = [
    'has_readme', # if false, then false
    'has_description', # if false, then false
    'is_collection_of_learnings',
    'has_app_application', 
    'is_awesome_curated', 
    'has_benchmark', 
    'is_block_explorer', 
    'is_boilerplate_scaffold_template', # if true, then true
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
    'is_fork',
    'has_tests_testing', 
    'has_tips', 
    'is_tooling', 
    'is_tutorial', 
    'is_whitepaper', 
    'is_workshop', 
    'is_wrapper', 
    'is_experiment',
    'is_research',
    'name_is_example', 
    'name_is_hello_world', 
    'name_is_whitepaper', 
    'name_is_tutorial', 
    'name_is_boilerplate', # if true, then true
    'name_is_scaffold', # if true, then true
    'name_is_template', # if true, then true 
    'name_is_kit', 
    'name_is_starter', 
    'name_is_getting_started', 
    'name_is_quickstart', 
    'name_is_guide', 
    'name_is_hackathon', 
    'name_is_bootcamp', 
    'name_is_course', 
    'name_is_workshop', 
    'name_is_interview', 
    'pm_has_main_entrypoint', 
    'pm_has_bin_script', 
    'pm_has_dependencies', 
    'pm_has_version_control', 
    'pm_has_author_cited', 
    'pm_has_license', 
    'pm_has_repository' 
]  

X = merged_df[feature_columns]

# Ensure all feature data is numeric (booleans will be treated as 0s and 1s)
X = X.astype(float)

y = merged_df['is_scaffold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create the scalar object for the training step
scaler = StandardScaler()

# fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# use the fitted scaler to transform the test data
X_test_scaled = scaler.transform(X_test)

# set the n_estimators param
n_estimators = 750

# Initialize and train a Random Forest model
# n_estimators is the number of trees in the forest
model_balanced = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=n_estimators)
model_balanced.fit(X_train_scaled, y_train)

## ----------------------------------------------------- Model Evaluation ------------------------------------------------- ##

# Make predictions on the test set
y_pred_class_weight_balanced = model_balanced.predict(X_test_scaled)

# Calculate and print the model's accuracy
accuracy = accuracy_score(y_test, y_pred_class_weight_balanced)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_class_weight_balanced))

# --- Confusion Matrix with Labels ---
# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred_class_weight_balanced)

print("Confusion Matrix (class weight = balanced):")
print("                 Predicted")
print("                 False    True")
print("Actual False    {:<8} {:<8}".format(cm[0][0], cm[0][1]))
print("       True     {:<8} {:<8}".format(cm[1][0], cm[1][1]))
print("\n")

# Explanation of the terms
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn} - Correctly predicted not scaffold")
print(f"False Positives (FP): {fp} - Incorrectly predicted as scaffold")
print(f"False Negatives (FN): {fn} - Incorrectly predicted as not scaffold (missed)")
print(f"True Positives (TP): {tp} - Correctly predicted as scaffold")

# Get importance scores
importance = model_balanced.feature_importances_

# Get the column names from training data
feature_names = X_train.columns

# Create a pandas Series to pair feature names with their importance scores
feat_importances = pd.Series(importance, index=feature_names)

# Sort the Series in descending order (most important features first)
sorted_importances = feat_importances.sort_values(ascending=False)

# 4. Print the sorted list
print("--- Feature Importances (Sorted) ---")
print(sorted_importances)

## ----------------------------------------------------- apply the model to the population ------------------------------------------------- ##

# 'repo_features_df' has the population of 300,000 repos
# 'merged_df' has labels

# Apply the EXACT SAME feature engineering to the new data
X_population = repo_features_df[feature_columns]

# Ensure the column order is identical to X_train
X_population = X_population[X_train.columns]

# Scale the full labeled dataset
# Create and fit the FINAL scaler on the ENTIRE labeled dataset
final_scaler = StandardScaler()
final_scaler.fit(X)

# Scale both datasets using a consistent scale
X_full_scaled = final_scaler.transform(X)

# Use the new scaler that was fit on the ENTIRE labeled dataset
X_population_scaled = final_scaler.transform(X_population)

## ------------------------- retrain the model on the full labeled dataset

# 'X' and 'y' are the full feature and target DataFrames from merged_df

# Define the final model with the proven parameters
final_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_estimators=n_estimators
)

# Fit the final model on ALL of the labeled data
final_model.fit(X_full_scaled, y)

print("Final model has been trained on the full labeled dataset.")

## ------------------------- apply the final model to the population
final_predictions = final_model.predict(X_population_scaled)

print(f"Generated {len(final_predictions)} predictions for the population using the default threshold of 0.5.")

## ------------------------- add the predictions to the population dataframe

# Add the predictions as a new column to the population DataFrame
repo_features_df['predicted_is_scaffold'] = final_predictions

# --- View Results ---

# See the first few rows with their new predictions
print("\n--- Population DataFrame with Predictions ---")
print(repo_features_df.head())

# See the distribution of the predictions
print("\n--- Prediction Counts ---")
print(repo_features_df['predicted_is_scaffold'].value_counts())

## --------------------------------------- write predictions back to database --------------------------------------- ##

print("\nPreparing to write predictions back to the database...")

# Create a new DataFrame with only repo and predicted_is_scaffold
predictions_to_write = repo_features_df[['repo', 'predicted_is_scaffold']]

# Use a transaction to ensure the entire operation succeeds or fails together.
try:
    with cloud_sql_engine.connect() as conn:
        with conn.begin() as transaction: # This starts a transaction.
            # Define a temporary table name
            temp_table_name = "temp_predictions_for_update"

            # Write the predictions DataFrame to the temporary table.
            # 'if_exists="replace"' ensures we start fresh if the script is re-run.
            print(f"Writing {len(predictions_to_write)} predictions to temporary table '{temp_table_name}'...")
            predictions_to_write.to_sql(
                temp_table_name,
                conn,
                schema='clean',
                if_exists='replace',
                index=False,
                dtype={'repo': sqlalchemy.types.Text, 'predicted_is_scaffold': sqlalchemy.types.Boolean}
            )
            print("Temporary table created successfully.")

            # Construct and execute the bulk UPDATE statement
            # This SQL joins the main table with the temporary one and updates the values.
            update_sql = text(f"""
                UPDATE clean.project_repos_features AS target
                SET
                    predicted_is_scaffold = source.predicted_is_scaffold
                FROM clean.{temp_table_name} AS source
                WHERE
                    target.repo = source.repo;
            """)

            print("Executing bulk UPDATE on 'clean.project_repos_features'...")
            result = conn.execute(update_sql)
            print(f"Update complete. {result.rowcount} rows were affected.")

            # explicitly drop the temporary table
            conn.execute(text(f"DROP TABLE clean.{temp_table_name}"))

    print("Predictions have been successfully written to the database.")

except Exception as e:
    print(f"An error occurred during the database update: {e}")
    # The transaction will be automatically rolled back if an error occurs.