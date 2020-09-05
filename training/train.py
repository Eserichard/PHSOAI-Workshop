import pandas as pd
import numpy as np
import pickle
import os
import helper_fns as hp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

# Make directory
directory = os.path.dirname('cv_results/')
if not os.path.exists(directory):
    os.makedirs(directory)

folder_path = "../data/processed/"
df_train = pd.read_pickle(folder_path + "df_train")
df_test = pd.read_pickle(folder_path + "df_test")
target = pd.read_pickle(folder_path + "target")


# Divide the training data into training (80%) and validation (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42, stratify=df_train[target])
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)

# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)

# Call id_checker on df
df_id = hp.id_checker(df)

# Print the first 5 rows of df_id
# df_id.head()

# Remove the identifiers from df_train, df_valid and df_test
df_train = df_train.drop(columns=df_id.columns)
df_valid = df_valid.drop(columns=df_id.columns)
df_test = df_test.drop(columns=df_id.columns)


# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)


# Call nan_checker on df
df_nan = hp.nan_checker(df)

# Print df_nan
df_nan

# Print the unique dtype of the variables with NaN
pd.DataFrame(df_nan['dtype'].unique(), columns=['dtype'])

# Get the variables with missing values, their proportion of missing values and dtype
df_miss = df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)

# Print df_miss
df_miss

# Remove rows with missing values from df_train, df_valid, df_test
df_train = df_train.dropna(subset=np.intersect1d(df_miss['var'], df_train.columns), inplace=False)
df_valid = df_valid.dropna(subset=np.intersect1d(df_miss['var'], df_valid.columns), inplace=False)
df_test = df_test.dropna(subset=np.intersect1d(df_miss['var'], df_test.columns), inplace=False)

# remove irrelevant features
to_remove = ['isbn', 'isbn13', 'authors', 'original_title', 'title', 
             'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
            'image_url', 'small_image_url', 'average_rating', 'to_read_count','work_ratings_count','work_text_reviews_count']



# Remove rows with irrelevant columns from df_train, df_valid, df_test
df_train = df_train.drop(columns=to_remove)
df_valid = df_valid.drop(columns=to_remove)
df_test = df_test.drop(columns=to_remove)

# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)

# Print the unique dtype of variables in df
pd.DataFrame(df.dtypes.unique(), columns=['dtype'])


# Call cat_var_checker on df
df_cat = hp.cat_var_checker(df)

# Print the dataframe
# df_cat


# One-hot-encode the categorical features in the combined data
df = pd.get_dummies(df, columns=np.setdiff1d(np.intersect1d(df.columns, df_cat['var']), [target]))

# Print the first 5 rows of df
# df.head()

# The LabelEncoder
le = LabelEncoder()

# Encode the categorical target in the combined data
df[target] = le.fit_transform(df[target].astype(str))

# Print the first 5 rows of df
# df.head()

# Separating the training data
df_train = df.iloc[:df_train.shape[0], :].copy(deep=True)

# Separating the validation data
df_valid = df.iloc[df_train.shape[0]:df_train.shape[0] + df_valid.shape[0], :].copy(deep=True)

# Separating the testing data
df_test = df.iloc[df_train.shape[0] + df_valid.shape[0]:, :].copy(deep=True)

# Print the dimension of df_remove_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

# getting the name of the features
features = np.setdiff1d(df.columns, [target])

# Get the feature matrix
X_train = df_train[features].to_numpy()
X_valid = df_valid[features].to_numpy()
X_test = df_test[features].to_numpy()

# Get the target vector
y_train = df_train[target].astype(int).to_numpy()
y_valid = df_valid[target].astype(int).to_numpy()


# scaling the data
# The StandardScaler
ss = StandardScaler()

# Standardize the training, validation and testing data
X_train = ss.fit_transform(X_train)
X_valid = ss.transform(X_valid)
X_test = ss.transform(X_test)


# Hyperparameter tunning
models = {'lr': LogisticRegression(class_weight='balanced', random_state=42),
          'dtc': DecisionTreeClassifier(class_weight='balanced', random_state=42),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=42),
          'mlpc': MLPClassifier(early_stopping=True, random_state=42)}

# create a dictionary of pipeline
pipes = {}
for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])

# get the predefined split crossvalidator

# Combine the feature matrix in the training and validation data
X_train_valid = np.vstack((X_train, X_valid))

# Combine the target vector in the training and validation data
y_train_valid = np.append(y_train, y_valid)

# Get the indices of training and validation data
train_valid_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_valid.shape[0], 0))

# The PredefinedSplit
ps = PredefinedSplit(train_valid_idxs)

# create dictonary of param grids
param_grids = {}

# params for logistic regression
C_grids = [10 ** i for i in range(-2, 3)]
tol_grids = [10 ** i for i in range(-6, -1)]
param_grids['lr'] = [{'model__C': C_grids,
                      'model__tol': tol_grids}]

# params for decision trees
min_samples_split_grids = [2, 30, 100]
min_samples_leaf_grids = [1, 30, 100]
max_depth_grids = range(1, 11)
param_grids['dtc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids,
                       'model__max_depth': max_depth_grids}]

# params for randomforest
min_samples_split_grids = [2, 20, 100]
min_samples_leaf_grids = [1, 20, 100]
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]


# params for multilayer perceptron
alpha_grids = [10 ** i for i in range(-6, -1)]
learning_rate_init_grids = [10 ** i for i in range(-5, 0)]
param_grids['mlpc'] = [{'model__alpha': alpha_grids,
                        'model__learning_rate_init': learning_rate_init_grids}]



# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_param_estimator_gs = []

for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_micro',
                      n_jobs=-1,
                      cv=ps,
                      return_train_score=True)
        
    # Fit the pipeline
    gs = gs.fit(X_train_valid, y_train_valid)
    
    # Update best_score_param_estimator_gs
    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(path_or_buf='cv_results/' + acronym + '.csv', index=False)

# Sort best_score_param_estimator_gs in descending order of the best_score_
best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_param_estimator_gs
pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])

# Get the best_score, best_param and best_estimator obtained by GridSearchCV
best_score_bm, best_param_bm, best_estimator_bm = best_score_param_estimator_gs[0]

# Get the dataframe of feature and importance
# df_fi_bm = pd.DataFrame(np.hstack((features.reshape(-1, 1), best_estimator_bm.named_steps['model'].feature_importances_.reshape(-1, 1))),
#                         columns=['Features', 'Importance'])

# Sort df_fi_bm in descending order of the importance
# df_fi_bm = df_fi_bm.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_
# df_fi_bm.head()

# Get the prediction on the testing data using best_model
y_test_pred = best_estimator_bm.predict(X_test)

# Transform y_test_pred back to the original class
y_test_pred = le.inverse_transform(y_test_pred)

# save model
model_out = open("../dist/model","wb")
pickle.dump(best_estimator_bm, model_out)
model_out.close()
 
