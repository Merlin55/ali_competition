from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


#ѵ��������׼ȷ�� path�������ļ���ַ    rate��adaboostѧϰ��
def train_test(path, rate):
    data = pd.read_csv(path)
    data_train = data.iloc[:, :-1]   #ȡ����
    type = data.iloc[:, -1]   #ȡ���ǩ
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=rate, algorithm='SAMME.R', random_state=None)
    model = adaboost.fit(data_train.iloc[:80000], type.iloc[:80000])   #ǰ80000����Ϊѵ������ʣ�µ���Ϊ���Լ�����ѵ����ѵ��ģ��
    return model.score(data_train.iloc[80000:], type.iloc[80000:])   #�����ڲ��Լ��ϵ�׼ȷ��

#ѵ���õ�ģ�ͣ�����������ݣ�����Ԥ������������ļ�   path_train��ѵ�������ļ���ַ   path_test�����������ļ���ַ
#         result���������ļ���ַ       rate��adaboostѧϰ��

def test_result(path_train, path_test, result_path, rate):
    data = pd.read_csv(path_train)
    data_train = data.iloc[:, :-1]   #ȡ����
    type = data.iloc[:, -1]   #ȡ���ǩ
    data_test = pd.read_csv(path_test)
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=rate, algorithm='SAMME.R', random_state=None)
    model = adaboost.fit(data_train, type)  #ѵ��ģ��
    result = model.predict(data_test.iloc[:, 1:])   #Ԥ����
    user_id = data_test.iloc[:, 0]   #����Ӧ�����Ӧ��user_id�ϣ����
    d = pd.DataFrame({'a':user_id, 'b':result})
    d.to_csv(result_path)
