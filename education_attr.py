# train a simple classification model to predict if a repo is educational or not
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
gsheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTbKlg5CJYNO3d2lrRbWWUy-71sr-oBApAJxCx2xmV931Y8CDrJ46SwVCUEoOT90LZsPpALVS_QixkE/pub?gid=1690796422&single=true&output=csv'
try:
    educational_df = pd.read_csv(gsheet_url)
except Exception as e:
    raise Exception(f"Error reading data from Google Sheets: {e}")

# --- Data Preparation ---
# Merge the feature data with the labeled data
merged_df = pd.merge(repo_features_df, educational_df, on='repo')

# drop rows where is_educational is null
merged_df = merged_df.dropna(subset=['is_educational'])

# print info about the merged dataframe
print(f"Merged dataframe info: {merged_df.info()}")

# print the first 5 rows of the merged dataframe
print(f"Merged dataframe first 5 rows: {merged_df.head()}")

# print the number of rows where is_dev_tooling is 1
print(f"Number of rows where is_educational is 1: {merged_df[merged_df['is_educational'] == 1].shape[0]}")

# print the number of rows where is_dev_tooling is 0
print(f"Number of rows where is_educational is 0: {merged_df[merged_df['is_educational'] == 0].shape[0]}")

# print the number of rows where is_dev_tooling is null
print(f"Number of rows where is_educational is null: {merged_df[merged_df['is_educational'].isnull()].shape[0]}")

# Separate features (X) and the target variable (y)
feature_columns = [
    'has_readme', # if false, then not education
    'is_collection_of_learnings', # if true, then is education
    'has_app_application', # if true, then not education
    'is_awesome_curated', # if true, then is education
    'has_benchmark', # if true, then not education
    'is_block_explorer', # if true, then not education
    'is_boilerplate_scaffold_template', # if true, then not education
    'is_bootcamp', # if true, then is education
    'is_bot', # if true, then not education
    'has_bounty_program', # if true, then not education
    'has_brand_icon_logo', # if true, then not education
    'is_cli_tool', # if true, then not education
    'is_library', # if true, then not education
    'is_course', # if true, then education
    'is_demo', # if true, then education
    'has_docs', # if true, then education
    'is_education_related', # if true, then education
    'is_eip_erc', # if true, then not education
    'has_examples', # if true, then education
    'is_feature_description', # if true, then education
    'is_starter_project', # if true, then education
    'is_guide', # if true, then education
    'is_hackathon_project', # if true, then education
    'is_hello_world', # if true, then education
    'uses_json_rpc', # if true, then not education
    'is_interview_related', # if true, then education
    'is_learning_material', # if true, then education
    'is_mcp_server', # if true, then not education
    'is_plugin', # if true, then not education
    'is_sample_project', # if true, then education
    'is_sdk', # if true, then not education
    'is_security_related', # if true, then not education
    'has_tests_testing', # if true, then not education
    'has_tips', # if true, then education
    'is_tooling', # if true, then not education
    'is_tutorial', # if true, then education
    'is_whitepaper', # if true, then education
    'is_workshop', # if true, then education
    'is_wrapper', # if true, then not education
    'is_experiment', # if true, then education
    'is_research', # if true, then education
    'name_is_example', # if true, then education
    'name_is_hello_world', # if true, then education
    'name_is_whitepaper', # if true, then education
    'name_is_tutorial', # if true, then education
    'name_is_boilerplate', # if true, then not education
    'name_is_scaffold', # if true, then not education
    'name_is_template', # if true, then not education
    'name_is_kit', # if true, then not education
    'name_is_starter', # if true, then education
    'name_is_getting_started', # if true, then education
    'name_is_quickstart', # if true, then education
    'name_is_guide', # if true, then education
    'name_is_hackathon', # if true, then education
    'name_is_bootcamp', # if true, then education
    'name_is_course', # if true, then education
    'name_is_workshop', # if true, then education
    'name_is_interview' # if true, then education
] 

X = merged_df[feature_columns]

# Ensure all feature data is numeric (booleans will be treated as 0s and 1s)
X = X.astype(float)

y = merged_df['is_educational']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create the scalar object for the training step
scaler = StandardScaler()

# fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# use the fitted scaler to transform the test data
X_test_scaled = scaler.transform(X_test)

# set the n_estimators param
n_estimators = 500

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
print(f"True Negatives (TN): {tn} - Correctly predicted not educational")
print(f"False Positives (FP): {fp} - Incorrectly predicted as educational")
print(f"False Negatives (FN): {fn} - Incorrectly predicted as not educational (missed)")
print(f"True Positives (TP): {tp} - Correctly predicted as educational")

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

# Get the predicted probabilities for the test set
# We only need the probability of the positive class (class 1)
y_pred_proba = model_balanced.predict_proba(X_test_scaled)
positive_class_proba = y_pred_proba[:, 1]

# Set a new, lower threshold to be more "aggressive"
new_threshold = 0.4  # Example: predict True if probability is 40% or more

# Apply the new threshold to get aggressive predictions
y_pred_aggressive = (positive_class_proba >= new_threshold).astype(int)

print("\n--- Aggressive Predictions (Threshold = 0.4) ---")
cm = confusion_matrix(y_test, y_pred_aggressive)
print(classification_report(y_test, y_pred_aggressive))

print("Confusion Matrix (Lowered threshold = 0.4):")
print("                 Predicted")
print("                 False    True")
print("Actual False    {:<8} {:<8}".format(cm[0][0], cm[0][1]))
print("       True     {:<8} {:<8}".format(cm[1][0], cm[1][1]))
print("\n")

# Explanation of the terms
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn} - Correctly predicted not educational")
print(f"False Positives (FP): {fp} - Incorrectly predicted as educational")
print(f"False Negatives (FN): {fn} - Incorrectly predicted as not educational (missed)")
print(f"True Positives (TP): {tp} - Correctly predicted as educational")

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

# Get the predicted probabilities for the population set
population_proba = final_model.predict_proba(X_population_scaled)

# Isolate the probability of the positive class (class 1)
positive_class_proba = population_proba[:, 1]

# Apply the optimal threshold to get the final predictions
final_predictions = (positive_class_proba >= new_threshold).astype(int)

print(f"Generated {len(final_predictions)} predictions for the population using a threshold of {new_threshold}.")

## ------------------------- add the predictions to the population dataframe

# Add the predictions as a new column to the population DataFrame
repo_features_df['predicted_is_educational'] = final_predictions

# --- View Results ---

# See the first few rows with their new predictions
print("\n--- Population DataFrame with Predictions ---")
print(repo_features_df.head())

# See the distribution of the predictions
print("\n--- Prediction Counts ---")
print(repo_features_df['predicted_is_educational'].value_counts())

## --------------------------------------- write predictions back to database --------------------------------------- ##

print("\nPreparing to write predictions back to the database...")

# Create a new DataFrame with only repo and predicted_is_educational
predictions_to_write = repo_features_df[['repo', 'predicted_is_educational']]

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
                dtype={'repo': sqlalchemy.types.Text, 'predicted_is_educational': sqlalchemy.types.Boolean}
            )
            print("Temporary table created successfully.")

            # Construct and execute the bulk UPDATE statement
            # This SQL joins the main table with the temporary one and updates the values.
            update_sql = text(f"""
                UPDATE clean.project_repos_features AS target
                SET
                    predicted_is_educational = source.predicted_is_educational
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