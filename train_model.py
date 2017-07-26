from data_cleaning import data, prediction_variables
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np

def split_data(data):
    # shuffle data
    data.reindex(np.random.permutation(data.index))

    # train, test
    train, test = train_test_split(data, test_size=0.3)

    # train variables
    train_X = train[prediction_variables]
    # train output
    train_Y = train.label

    # test variables
    test_X = test[prediction_variables]
    # output test variables
    test_Y = test.label

    return [train_X, train_Y, test_X, test_Y]


input_data = split_data(data)
train_X = input_data[0]
train_Y = input_data[1]
test_X  = input_data[2]
test_Y  = input_data[3]

# fit a Random Forest. n_estimators = 100
def predict_RF_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    return accuracy # 0.970557308097

# fit SVM
def predict_SVM():
    model = svm.SVC()
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    return accuracy # 0.703470031546

# cross validation of models
def classification_model(model, data, prediction_input, output):
    # fit using training set
    model.fit(data[prediction_input], data[output])
    # predict based on training set
    predictions = model.predict(data[prediction_input])
    # accuracy
    accuracy = metrics.accuracy_score(predictions, data[output])
    print ("Accuracy: ", accuracy)

    # create 5 partitions
    kf = KFold(n_splits=5)

    error = []

    # Split dataset into 5 consecutive folds
    for train, test in kf.split(data):
        # rows & columns
        train_X = (data[prediction_input].iloc[train,:])
        # rows
        train_y = data[output].iloc[train]
        model.fit(train_X, train_y)

        # test data also
        test_X = data[prediction_input].iloc[test, :]
        test_y = data[output].iloc[test]
        error.append(model.score(test_X, test_y))

        # score
        print("Cross Validation score: " , np.mean(error))

def cross_validation_Decision_Tree_Classifier():
    model = DecisionTreeClassifier()
    outcome_var = "label"
    classification_model(model, data, prediction_variables, outcome_var)

print("-------------Decision Tree Classifier------")
cross_validation_Decision_Tree_Classifier()

'''
    Accuracy:  1.0
Cross Validation score:  0.774447949527
Cross Validation score:  0.858832807571
Cross Validation score:  0.889589905363
Cross Validation score:  0.899419917174
Cross Validation score:  0.906581747325
'''

def cross_validation_SVM():
    model = svm.SVC()
    outcome_var = "label"
    classification_model(model, data, prediction_variables, outcome_var)

print("-------------SVM------")
cross_validation_SVM()
'''
    Accuracy:  0.758207070707
Cross Validation score:  0.317034700315
Cross Validation score:  0.369085173502
Cross Validation score:  0.471083070452
Cross Validation score:  0.47021593633
Cross Validation score:  0.466220142429
'''

def cross_validation_RF_Classifier():
    model = RandomForestClassifier(n_estimators=100)
    outcome_var = "label"
    classification_model(model, data, prediction_variables, outcome_var)

print("-------------RF Classifier------")
cross_validation_RF_Classifier()
'''
    Accuracy:  1.0
Cross Validation score:  0.839116719243
Cross Validation score:  0.894321766562
Cross Validation score:  0.9174553102
Cross Validation score:  0.927033030833
Cross Validation score:  0.93056797285
'''

# Use grid search to find the best parameters for prediction using SVM
data_X = data[prediction_variables]
data_Y = data["label"]
def model_grid_search_CV(model, param_grid, data_X, data_Y):
    clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    clf.fit(train_X, train_Y)
    print("Best Parameters: ")
    print(clf.best_params_)
    print("Best Estimator: ")
    print(clf.best_estimator_)
    print("Best Score: ")
    print(clf.best_score_)

def svm_grid_search():
    model = svm.SVC()
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']
         },
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']
         },
    ]
    model_grid_search_CV(model, param_grid, data_X, data_Y)

print("----GridSearchCV----")
svm_grid_search()
'''
    Best Parameters: 
{'kernel': 'linear', 'C': 10}
Best Estimator: 
SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Best Score: 
0.967072620659
'''



