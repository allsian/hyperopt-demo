from hyperopt import fmin, tpe, hp, STATUS_OK

from sklearn import datasets
# the function used to do cross validation
# change `cv` parameter to specify fold-number, default is 3-fold
# parameter:(estimator, X(feature), y(label))
from sklearn.model_selection import cross_val_score
# used to normalize and scale data
from sklearn.preprocessing import normalize, scale
from sklearn.svm import SVC

from hyperopt.mongoexp import MongoTrials

iris = datasets.load_iris()
X = iris.data
y = iris.target

# objective function
def f(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = SVC(**params)
    return {'loss': -cross_val_score(clf, X_, y).mean(), 'status': STATUS_OK}

if __name__ == "__main__":
    space4svm = {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    }
    trials = MongoTrials('mongo://localhost:1234/f_file_db/jobs', exp_key='exp1')
    best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
