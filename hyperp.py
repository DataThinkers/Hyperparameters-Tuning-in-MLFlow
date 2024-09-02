import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 1. import the mlflow 
import mlflow
# 1.1 Let's set tracking uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load dataset
#2. let's create new experiment
mlflow.set_experiment('water_potability_rf')

data = pd.read_csv("D:\\exp_track_mlflow1\\data\\water_potability.csv")

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Function to fill missing values with median
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

# Fill missing values with median for training and test sets
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Prepare training data
X_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values

# Define the model and parameter distribution for RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40],
}

# Perform RandomizedSearchCV to find the best hyperparameters
search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)



# 3. For Mlflow tracking we have to write
# with mlflow.start_run():
#     random_search.fit(X_train, y_train)

#     # log all the children
#     for i in range(len(random_search.cv_results_['params'])):
#         with mlflow.start_run(nested=True):
            
#             mlflow.log_params(random_search.cv_results_['params'][i])
#             mlflow.log_metric("accuracy",random_search.cv_results_['mean_test_score'][i])
# Start the parent MLflow run
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    
    # Hyperparameter tuning with RandomizedSearchCV
    #search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    
    # Log all hyperparameter combinations
    for i in range(len(search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination {i+1}", nested=True) as child_run:
            params = search.cv_results_['params'][i]
            mean_test_score = search.cv_results_['mean_test_score'][i]
            
            # Log the parameters and their corresponding score
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", mean_test_score)
    
    # Print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", search.best_params_)
    
    # 4. Let's first log best parameters that we found above
     
    mlflow.log_params(search.best_params_)  

    # Train the model with the best parameters
    best_rf = search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Save the trained model to a file
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare test data
    X_test = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values

    # Load the saved model
    model = pickle.load(open('model.pkl', "rb"))

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 5. Now let's log best metrics that we found using model with best parameters
    
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1 score",f1)
    # 6. Let's log dataset as well
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")
    # 7. Also let's log source code as well

    mlflow.log_artifact(__file__) 
    # 8. Also we are going to log model as well

    mlflow.sklearn.log_model(search.best_estimator_,"Best Model")

    #  9 let's add tags as well
    mlflow.set_tag("author","datathinkers")
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
