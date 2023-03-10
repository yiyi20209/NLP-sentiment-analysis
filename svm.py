# -*- coding: utf-8 -*-
"""nlp-svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VrejFk6AoxV3Sb1ZBpzMAtwhVul6XcsF
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

def get_detaset():
    train_df = pd.read_csv('/content/drive/MyDrive/IFT6390/kaggle2/train_data.csv')
    test_df = pd.read_csv('/content/drive/MyDrive/IFT6390/kaggle2/test_data.csv')
    train_result_df = pd.read_csv('/content/drive/MyDrive/IFT6390/kaggle2/train_results.csv')
    return train_df, test_df, train_result_df

def treat_detaset(train_df, test_df, train_result_df):
    train_df= train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])
    train_result_df = train_result_df.drop(columns=['id'])

    train_result_df.loc[train_result_df['target'] == 'negative'] = 0
    train_result_df.loc[train_result_df['target'] == 'neutral'] = 1
    train_result_df.loc[train_result_df['target'] == 'positive'] = 2
    train_result_df = train_result_df.astype('int')
    return train_df, test_df, train_result_df

train_df, test_df, train_result_df = get_detaset()
X, X_test, y = treat_detaset(train_df, test_df, train_result_df)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=6390)

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf-svm', LinearSVC(C=0.5, tol=0.001, random_state=6390, max_iter=2000))
                        ])
_ = text_clf_svm.fit(X_train.text, y_train.target)
predicted_svm = text_clf_svm.predict(X_val.text)
np.mean(predicted_svm == y_val.target)

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__tol': (1e-3, 1e-4, 1e-5),
              'clf-svm__C': (0.5, 1.0, 1.5)
             }
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X.text, y.target)
gs_clf_svm.best_score_
gs_clf_svm.best_params_

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf-svm', LinearSVC(C=0.5, tol=0.001, random_state=6390, max_iter=2000))
                        ])
_ = text_clf_svm.fit(X.text, y.target)
predicted_svm = text_clf_svm.predict(X_test.text)

output = pd.DataFrame({'id': np.arange(X_test.shape[0]), 'target': predicted_svm})
output.to_csv('submission_svm.csv', index=False)