from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


# training and calculate the precision   path: the location of data    rate： learing rate of adaboost
def train_test(path, rate):
    data = pd.read_csv(path)
    data_train = data.iloc[:, :-1]   #get attribute
    type = data.iloc[:, -1]   #get label
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=rate, algorithm='SAMME.R', random_state=None)
    model = adaboost.fit(data_train.iloc[:80000], type.iloc[:80000])   #the first 80000 rows as training set，the rest as testing 
    return model.score(data_train.iloc[80000:], type.iloc[80000:])   #the precision rate on testing set

#training model, input testing set, output results into files   path_train：the locatio of training set训练数据文件地址   path_test：the location of testing set
#         result：the location of output file       rate：learing rate of adaboost

def test_result(path_train, path_test, result_path, rate):
    data = pd.read_csv(path_train)
    data_train = data.iloc[:, :-1]   #get attribute
    type = data.iloc[:, -1]   #get label
    data_test = pd.read_csv(path_test)
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=rate, algorithm='SAMME.R', random_state=None)
    model = adaboost.fit(data_train, type)  #training model
    result = model.predict(data_test.iloc[:, 1:])   #results
    user_id = data_test.iloc[:, 0]   #output according to user_id
    d = pd.DataFrame({'a':user_id, 'b':result})
    d.to_csv(result_path)
