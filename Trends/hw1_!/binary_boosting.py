#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.estimators_ = []

    def loss_grad(self, original_y, pred_y):
        # Вычислите градиент на кажом объекте
        ### YOUR CODE ###

        return +original_y / (1 + np.exp(-original_y * pred_y))
        #return -original_y / (1 + np.exp(-pred_y)) + (1 - original_y) / (1 + np.exp(pred_y))

    def fit(self, X, original_y):
        # Храните базовые алгоритмы тут
        self.estimators_ = []
        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))
            # Настройте базовый алгоритм на градиент, это классификация или регрессия?
            
            estimator = deepcopy(self.base_regressor)
            estimator.fit(X,grad)                     

            ### END OF YOUR CODE
            self.estimators_.append(estimator)

        self.out_ = self._outliers(grad)
        self.feature_importances_ = self._calc_feature_imps()

        return self

    def _predict(self, X):
        
        # Получите ответ композиции до применения решающего правила
        ### YOUR CODE ###
        y_pred = np.zeros(X.shape[0])
        if self.estimators_ == []:
            return y_pred
        for estimator in self.estimators_:
            y_pred += self.lr * np.array(estimator.predict(X))
        return y_pred
    
    def predict(self, X):
        # Примените к self._predict решающее правило
        ### YOUR CODE ###
        y_pred = (self._predict(X) > 0).astype(int)
        print self._predict(X)
        
        #Вообще я думал условие > 0.5,но по графику выходит наоборот
        return y_pred

    def _outliers(self, grad):
        # Топ-10 объектов с большим отступом
        ### YOUR CODE ###
        _outliers = np.argsort(np.fabs(grad))[:10]

        return _outliers

    def _calc_feature_imps(self):
        # Посчитайте self.feature_importances_ с помощью аналогичных полей у базовых алгоритмов
        f_imps = None
        ### YOUR CODE ###
        f_imps = np.sum([estimator.feature_importances_ for estimator in self.estimators_],axis = 0)

        return f_imps/len(self.estimators_)
