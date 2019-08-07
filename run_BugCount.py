import numpy as np
from Processing import Processing
from configuration_file import configuration_file
from sklearn import linear_model
from openpyxl import Workbook
import os
import time
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
from rankboostYin import *
import xlrd
import shutil
from PerformanceMeasure import PerformanceMeasure
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from runWeka import *
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso
from sklearn import linear_model
from GPR import *
from skrvm import RVR
from pyearth import Earth
from LTR_New import *
from RankSVM import RankSVM


'''
需要导入skrvm包，安装方法在终端输入pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
需要导入pyearth包，安装方法:git clone git://github.com/scikit-learn-contrib/py-earth.git
cd py-earth
sudo python setup.py install

预测缺陷个数的情况下
'''


warnings.filterwarnings('ignore')


header = ["数据集", "NB", "LogR", "SL", "RBFNet","SMO", "CART", "C4.5 ", "LMT", "RF", "KNN", "Ripper",
          "Ridor", "DTR", "RFR", "LR", "NR", "LassoR", "LAR", "GP", "NNR", "SVR", "RVM",
          "MARS", "KNR", "Kstar", "Ranking SVM", "RankBoost","RankNet", "LambdaRank", "ListNet",
          "AdaRank", "Coordinate Ascent", "LTR-linear", "LTR-logistic"
          ]

header2 = ["数据集", "NaiveBayes", "LogisticRegression", "SimpleLogistic", "RBFNet",
          "SMO", "CART", "C4.5 ", "Logistic Model Tree", "Random Forest", "K-Nearest Neighbors", "Ripper",
          "RippleDownRules", "DecisionTreeRegression", "RandomForestRegression", "Linear Regression",
          "Bayesian Ridge Regression", "Lasso Regression", "Least Angle Regression ",
          "Genetic Programming", "Neural Network Regression", "Support Vector Regression", "Relevance Vector Machine",
          "Multivariate Adaptive Regression Splines", "K-nearest Neighbors Regression ", "Kstar", "Ranking SVM", "RankBoost",
          "RankNet", "LambdaRank", "ListNet", "AdaRank", "Coordinate Ascent", "LTR-linear", "LTR-log"
          ]

'''
分类方法调优参数
'''
LR_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]
DTC_tuned_parameters = [{'min_samples_split': [2, 3, 4, 5, 6]}]
BC_tuned_parameters = [{'n_estimators': [10, 20, 30, 40, 50]}]
RFC_tuned_parameters = [{'n_estimators': [10, 20, 30, 40, 50]}]
KNC_tuned_parameters = [{'n_neighbors': [1, 5, 9, 13, 17]}]
'''
回归方法的调优参数
'''
dtr_tuned_parameters = [{'min_samples_split': [2, 3, 4, 5, 6]}]
RFR_tuned_parameters = [{'n_estimators': [10, 20, 30, 40, 50]}]
lr_tuned_parameters = [{'normalize': [True, False]}]
ridge_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]

lasso_tuned_parameters = [{'normalize': [True, False]}]
lar_tuned_parameters = [{'normalize': [True, False]}]

mlpr_tuned_parameters = [{'hidden_layer_sizes': [4, 8, 16, 32, 64, 100],
                          'alpha': [0, 0.0001, 0.001, 0.01, 0.1]}]

svr_tuned_parameters = [{'C': [0.25, 0.5, 1, 2, 4]}]

knr_tuned_parameters = [{'n_neighbors': [1, 5, 9, 13, 17]}]

'''
其他可以调的参数
'''
cv_times = 3
number_of_booststrap = 10
selectedFeatureNumber = 5


def reg_method(training_data_X, training_data_y, test_data_X,test_data_y, score_func, filename, i):

    print("into reg_method..........")
    if score_func == "FPA":
        def my_fpa_score(realbug, predbug):
            return PerformanceMeasure(realbug, predbug).FPA()
        my_score = my_fpa_score
    else:
        print("没有设置除FPA之外的score_func")
        exit()
    # DTR（需要网格搜素）
    print("DTR....")
    dtr = GridSearchCV(DecisionTreeRegressor(), dtr_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    dtr.fit(training_data_X, training_data_y)
    DTR_pred = dtr.predict(test_data_X)

    # RandomForestRegression  （需要网格搜素）
    print("RandomForestRegression....")
    regr = GridSearchCV(RandomForestRegressor(), RFR_tuned_parameters, cv=cv_times,
                 scoring=make_scorer(my_score, greater_is_better=True))
    # regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators = 100)
    regr.fit(training_data_X, training_data_y)
    Regr_pred = regr.predict(test_data_X)

    # Linear Regression （需要网格搜素）
    print("Linear Regression")
    lr = GridSearchCV(linear_model.LinearRegression(), lr_tuned_parameters, cv=cv_times,
                      scoring=make_scorer(my_score, greater_is_better=True))
    lr.fit(training_data_X, training_data_y)
    Lr_pred = lr.predict(test_data_X)

    # Ridge Regression  （需要网格搜素）
    print("Ridge Regression")
    ridge = GridSearchCV(Ridge(), ridge_tuned_parameters, cv=cv_times,
                         scoring=make_scorer(my_score, greater_is_better=True))
    ridge.fit(training_data_X, training_data_y)
    Ridge_pred = ridge.predict(test_data_X)

    # Lasso Regression (需要网格搜素)  lasso_tuned_parameters
    print("Lasso Regression")
    model = GridSearchCV(Lasso(), lasso_tuned_parameters, cv=cv_times,
                         scoring=make_scorer(my_score, greater_is_better=True))
    # model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
    model.fit(training_data_X, training_data_y)  # 线性回归建模
    Lasso_pred = model.predict(test_data_X)

    # Least Angle Regression （需要网格搜素） lar_tuned_parameters
    print("Least Angle Regression")
    lar = GridSearchCV(linear_model.LassoLars(), lar_tuned_parameters, cv=cv_times,
                 scoring=make_scorer(my_score, greater_is_better=True))
    # lar = linear_model.LassoLars(alpha=0.3)
    lar.fit(training_data_X, training_data_y)
    Lar_pred = lar.predict(test_data_X)

    # Genetic Programming (需要网格搜素，已加)
    print("Genetic Programming....")
    gpr = GPR(NP=100, F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)], generation=100, len_x=20,
              value_up_range=20.0,
              value_down_range=-20.0, X=training_data_X, y=training_data_y)
    gpr_w = gpr.process()
    Gpr_pred = gpr.predict(test_data_X, gpr_w)

    # Neural Network Regression （需要网格搜素）
    print("Neural Network Regression....")
    mlpr = GridSearchCV(MLPRegressor(), mlpr_tuned_parameters, cv=cv_times,
                        scoring=make_scorer(my_score, greater_is_better=True))
    mlpr.fit(training_data_X, training_data_y)
    Mlpr_pred = mlpr.predict(test_data_X)

    # Support Vector Regression （需要网格搜素）
    print("Support Vector Regression....")
    svr = GridSearchCV(SVR(), svr_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    svr.fit(training_data_X, training_data_y)
    Svr_pred = svr.predict(test_data_X)

    # Relevance Vector Machine （不需要网格搜素）
    print("Relevance Vector Machine")
    rvr = RVR(kernel='linear')
    try:
        rvr.fit(training_data_X, training_data_y)
        Rvr_pred = rvr.predict(test_data_X)
    except Exception as e:
        Rvr_pred = [999 for i in test_data_X]

    # Multivariate Adaptive Regression Splines （不需要网格搜素）
    print("Multivariate Adaptive Regression Splines")
    mars = Earth()
    mars.fit(training_data_X, training_data_y)
    Mars_pred = mars.predict(test_data_X)

    # K-nearest Neighbors Regression    （需要网格搜素）
    print("K-nearest Neighbors Regression")
    knr = GridSearchCV(KNeighborsRegressor(), knr_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    knr.fit(training_data_X, training_data_y)
    Knr_pred = knr.predict(test_data_X)

    # Kstar （不需要网格搜素）
    print("Kstar")
    train_arff_path, test_arff_path = createWekaData("regression", training_data_X, training_data_y,
                                                     test_data_X, test_data_y, str(i + 1) + "_" + filename)
    save_pred_data_path = trainAndTest("kstar", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    Kstar_pred = get_pred_bug(save_pred_data_path)

    return [(DTR_pred, 'Decision Tree Regressor'),
            (Regr_pred, 'Random Forest Regression'),
            (Lr_pred, 'Linear Regression'),
            (Ridge_pred, 'Ridge Regression'),
            (Lasso_pred, "Lasso Regression"),
            (Lar_pred, 'Least Angle Regression'),
            (Gpr_pred, 'Genetic Programming'),
            (Mlpr_pred, 'Neural Network Regression'),
            (Svr_pred, 'Support Vector Regression'),
            (Rvr_pred, 'Relevance Vector Machine'),
            (Mars_pred, 'Multivariate Adaptive Regression Splines'),
            (Knr_pred, 'K-nearest Neighbors Regression'),
            (Kstar_pred, 'Kstar')
            ]


def ClassificationMethod(X, Y, testX, testY, score_func, filename, i):
    print("into ClassificationMethod..........")
    if score_func == "FPA":
        def my_fpa_score(realbug, predbug):
            return PerformanceMeasure(realbug, predbug).FPA()
        my_score = my_fpa_score
    else:
        print("没有设置除FPA之外的score_func")
        exit()
    # 分类算法
    # 朴素贝叶斯（不需要网格搜索）
    gnb = GaussianNB()
    gnb_pred = gnb.fit(X, Y).predict_proba(testX)
    gnb_pred = [p[1] for p in gnb_pred]

    # 逻辑回归（需要网格搜索）
    LR = GridSearchCV(LogisticRegression(), LR_tuned_parameters, cv=cv_times,
                      scoring=make_scorer(my_score, greater_is_better=True))
    LR_pred = LR.fit(X, Y).predict_proba(testX)
    LR_pred = [p[1] for p in LR_pred]

    # Simple Logistic（不需要网格搜素）
    train_arff_path, test_arff_path = createWekaData("classfication", X, Y,
                                                     testX, testY,
                                                     str(i + 1) + "_" + filename)
    save_pred_data_path = trainAndTest("sl", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    SL_pred = get_pred_bug(save_pred_data_path)

    # rbf（0.1,0.01,0.001, 0.0001, 0.000010.1,0.01,0.001, 0.0001, 0.00001）这个怎么加网格搜素？？
    save_pred_data_path = trainAndTest("rbf", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    RBF_pred = get_pred_bug(save_pred_data_path)

    # smo（不需要网格搜素）
    save_pred_data_path = trainAndTest("smo", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    SMO_pred = get_pred_bug(save_pred_data_path)

    # CART（需要网格搜素）
    DTC = GridSearchCV(DecisionTreeClassifier(), DTC_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    DTC_pred = DTC.fit(X, Y).predict_proba(testX)
    DTC_pred = [p[1] for p in DTC_pred]

    # c45（不需要网格搜素）
    save_pred_data_path = trainAndTest("c45", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    C45_pred = get_pred_bug(save_pred_data_path)

    # Logistic Model Tree（不需要网格搜素）
    save_pred_data_path = trainAndTest("lmt", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    LMT_pred = get_pred_bug(save_pred_data_path)

    # Random Forest（需要网格搜素）
    RFC = GridSearchCV(RandomForestClassifier(), RFC_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    RFC_pred = RFC.fit(X, Y).predict_proba(testX)
    RFC_pred = [p[1] for p in RFC_pred]

    # K-Nearest Neighbors（需要网格搜素）
    KNC = GridSearchCV(KNeighborsClassifier(), KNC_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=True))
    KNC_pred = KNC.fit(X, Y).predict_proba(testX)
    KNC_pred = [p[1] for p in KNC_pred]

    # Ripper（网格参数：1,2,3,4，5）使用的是Weka不知道怎么加？
    save_pred_data_path = trainAndTest("jrip", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    Ripper_pred = get_pred_bug(save_pred_data_path)

    # Ripple Down Rules (Ridor)   2,6, 10,14,18
    save_pred_data_path = trainAndTest("ridor", train_arff_path, test_arff_path, str(i + 1) + "_" + filename)
    Ridor_pred = get_pred_bug(save_pred_data_path)

    return [(gnb_pred, 'GaussianNB'),
            (LR_pred, 'LogisticRegression'),
            (SL_pred, 'Simple Logistic'),
            (RBF_pred, 'RBF'),
            (SMO_pred, 'SMO'),
            (DTC_pred, 'CART'),
            (C45_pred, 'C4.5'),
            (LMT_pred, 'Logistic Model Tree'),
            (RFC_pred, 'Random Forest'),
            (KNC_pred, 'K-Nearest Neighbors'),
            (Ripper_pred, 'Ripper'),
            (Ridor_pred, 'Ridor')
            ]


def run_jarAndLTRMethod(training_data_X, training_data_y, test_data_X,test_data_y, filename):
    pred_return  = []
    # 先跑ranking SVM
    rs_pred_y = RankSVM().fit(training_data_X, training_data_y).predict(test_data_X)
    pred_return.append((rs_pred_y, "ranking SVM"))
    # 在计算jar包中的算法
    method_index = [2, 1, 5, 7, 3, 4]
    method_name = ["RankBoost", "RankNet", "LambdaRank", "ListNet", "AdaRank", "Coordinate Ascent"]
    # 2: RankBoost 1: RankNet 5: LambdaRank 7: ListNet 3: AdaRank 4: Coordinate Ascent
    train_dat_path, test_dat_path = RankalgorithmCreatedata(training_data_X, training_data_y, test_data_X,
                                                            test_data_y, filename)
    for i in range(len(method_index)):
        print("目前的ranker为：", method_index[i])
        pred_bug = RankalgorithmTrainandtest(method_index[i], filename, train_dat_path, test_dat_path)
        pred_return.append((pred_bug, method_name[i]))
    # 最后执行两个LTR方法
    cost = [1 for i in range(len(training_data_y))]
    de = LTR(X=training_data_X, y=training_data_y, cost=cost, costflag='module', logorlinear='linear')
    Ltr_w = de.process()
    Ltr_linear_pred = de.predict(test_data_X, Ltr_w)
    pred_return.append((Ltr_linear_pred, "LTR_linear"))

    de = LTR(X=training_data_X, y=training_data_y, cost=cost, costflag='module', logorlinear='log')
    Ltr_w = de.process()
    Ltr_log_pred = de.predict(test_data_X, Ltr_w)
    pred_return.append((Ltr_log_pred, "LTR_log"))
    return pred_return
    pass


def Predictbugs(training_data_X, training_data_y, testing_data_X, testing_data_y, Cla_training_data_y, filename, i, avg_cc, testingcodeN):
    # 评价模型预测缺陷个数的性能好坏时，只用FPA,CLC,PofB20来评价
    print("into Predictbugs................................")
    fpa1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fpa13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fpa26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Precision5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recall5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F15module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F15module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F15module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Precisionx5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recallx5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F1x5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PF5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    falsealarmrate5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFLA5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFCCA5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFMA5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PMI5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PofB5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # ---------------------------------------------------

    Precision10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recall10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F110module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F110module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F110module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Precisionx10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recallx10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F1x10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PF10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    falsealarmrate10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFLA10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFCCA10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFMA10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PMI10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PofB10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # ------------------------------------

    Precision20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precision20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recall20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recall20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F120module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F120module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F120module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Precisionx20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Precisionx20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    Recallx20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Recallx20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    F1x20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F1x20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PF20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PF20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    falsealarmrate20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsealarmrate20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFLA20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFLA20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFCCA20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFCCA20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    IFMA20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    IFMA20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PMI20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PMI20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    PofB20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PofB20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # ---------------------
    normpercentpopt5module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt5module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt5module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    normpercentpopt10module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt10module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt10module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    normpercentpopt20module1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt20module13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normpercentpopt20module26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    normWholepoptmodule1_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normWholepoptmodule13_25 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    normWholepoptmodule26_34 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 分类算法（12个算法）
    classification_results = ClassificationMethod(training_data_X, Cla_training_data_y, testing_data_X,
                                                  testing_data_y, "FPA", filename, i)
    index = 0
    for pred_prob, name in classification_results:
        fpa = PerformanceMeasure(testing_data_y, pred_prob).FPA()
        print("===========0.05===============")
        Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module, PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module  = \
            PerformanceMeasure(
            testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect', 'module').getSomePerformance()
        print("all 0.05 is:", Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module, PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module )
        print("===========0.1===============")
        Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module, F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module, PofB10module =\
        PerformanceMeasure(
            testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect', 'module').getSomePerformance()
        print("all 0.1 is:", Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module, F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module, PofB10module)
        print("===========0.2===============")
        Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module, PofB20module =\
        PerformanceMeasure(
            testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect', 'module').getSomePerformance()
        print("all 0.2 is:", Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module, PofB20module)
        print("============POPT==============")
        normpercentpopt5module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect', 'module').PercentPOPT()
        normpercentpopt10module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect', 'module').PercentPOPT()
        normpercentpopt20module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect', 'module').PercentPOPT()
        normWholepoptmodule = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 1, 'defect', 'module').POPT()

        # 写到1~12的结果list中
        fpa1_12[index] = fpa
        Precision5module1_12[index] = Precision5module
        Recall5module1_12[index] = Recall5module
        F15module1_12[index] = F15module
        Precisionx5module1_12[index] = Precisionx5module
        Recallx5module1_12[index] = Recallx5module
        F1x5module1_12[index] = F1x5module
        PF5module1_12[index] = PF5module
        falsealarmrate5module1_12[index] = falsealarmrate5module
        IFLA5module1_12[index] = IFLA5module
        IFCCA5module1_12[index] = IFCCA5module
        IFMA5module1_12[index] = IFMA5module
        PMI5module1_12[index] = PMI5module
        PofB5module1_12[index] = PofB5module

        Precision10module1_12[index] = Precision10module
        Recall10module1_12[index] = Recall10module
        F110module1_12[index] = F110module
        Precisionx10module1_12[index] = Precisionx10module
        Recallx10module1_12[index] = Recallx10module
        F1x10module1_12[index] = F1x10module
        PF10module1_12[index] = PF10module
        falsealarmrate10module1_12[index] = falsealarmrate10module
        IFLA10module1_12[index] = IFLA10module
        IFCCA10module1_12[index] = IFCCA10module
        IFMA10module1_12[index] = IFMA10module
        PMI10module1_12[index] = PMI10module
        PofB10module1_12[index] = PofB10module

        Precision20module1_12[index] = Precision20module
        Recall20module1_12[index] = Recall20module
        F120module1_12[index] = F120module
        Precisionx20module1_12[index] = Precisionx20module
        Recallx20module1_12[index] = Recallx20module
        F1x20module1_12[index] = F1x20module
        PF20module1_12[index] = PF20module
        falsealarmrate20module1_12[index] = falsealarmrate20module
        IFLA20module1_12[index] = IFLA20module
        IFCCA20module1_12[index] = IFCCA20module
        IFMA20module1_12[index] = IFMA20module
        PMI20module1_12[index] = PMI20module
        PofB20module1_12[index] = PofB20module

        normpercentpopt5module1_12[index] = normpercentpopt5module
        normpercentpopt10module1_12[index] = normpercentpopt10module
        normpercentpopt20module1_12[index] = normpercentpopt20module
        normWholepoptmodule1_12[index] = normWholepoptmodule

        index += 1

    # 回归算法（13个）
    index = 0
    regression_results = reg_method(training_data_X, training_data_y, testing_data_X,
                                        testing_data_y, "FPA", filename, i)
    for pred_prob, name in regression_results:
        if name == 'Relevance Vector Machine' and min(pred_prob) == 999 and max(pred_prob) == 999:
            print("Relevance Vector Machine进入try except了，将指标置为异常值")
            fpa = Precision5module = Recall5module = F15module = Precisionx5module = Recallx5module = F1x5module = \
                PF5module = falsealarmrate5module = IFLA5module = IFCCA5module = IFMA5module = PMI5module = PofB5module = \
                Precision10module = Recall10module = F110module = Precisionx10module = Recallx10module = F1x10module = \
                PF10module = falsealarmrate10module = IFLA10module = IFCCA10module = IFMA10module = PMI10module = PofB10module = \
                Precision20module = Recall20module = F120module = Precisionx20module = Recallx20module = F1x20module = PF20module = \
                falsealarmrate20module = IFLA20module = IFCCA20module = IFMA20module = PMI20module = PofB20module = \
                normpercentpopt5module = normpercentpopt10module = normpercentpopt20module = normWholepoptmodule = 999
        else:
            fpa = PerformanceMeasure(testing_data_y, pred_prob).FPA()
            print("===========0.05===============")
            Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module, PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect', 'module').getSomePerformance()
            print("all 0.05 is:", Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module,
                  PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module)
            print("===========0.1===============")
            Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module, F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module, PofB10module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect', 'module').getSomePerformance()
            print("all 0.1 is:", Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module,
                  F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module,
                  PofB10module)
            print("===========0.2===============")
            Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module, PofB20module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect', 'module').getSomePerformance()
            print("all 0.2 is:", Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module,
                  F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module,
                  PofB20module)
            print("============POPT==============")
            normpercentpopt5module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect',
                                                        'module').PercentPOPT()
            normpercentpopt10module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect',
                                                         'module').PercentPOPT()
            normpercentpopt20module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect',
                                                         'module').PercentPOPT()
            normWholepoptmodule = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 1, 'defect',
                                                     'module').POPT()

        # 写到13~25的结果list中
        fpa13_25[index] = fpa
        Precision5module13_25[index] = Precision5module
        Recall5module13_25[index] = Recall5module
        F15module13_25[index] = F15module
        Precisionx5module13_25[index] = Precisionx5module
        Recallx5module13_25[index] = Recallx5module
        F1x5module13_25[index] = F1x5module
        PF5module13_25[index] = PF5module
        falsealarmrate5module13_25[index] = falsealarmrate5module
        IFLA5module13_25[index] = IFLA5module
        IFCCA5module13_25[index] = IFCCA5module
        IFMA5module13_25[index] = IFMA5module
        PMI5module13_25[index] = PMI5module
        PofB5module13_25[index] = PofB5module

        Precision10module13_25[index] = Precision10module
        Recall10module13_25[index] = Recall10module
        F110module13_25[index] = F110module
        Precisionx10module13_25[index] = Precisionx10module
        Recallx10module13_25[index] = Recallx10module
        F1x10module13_25[index] = F1x10module
        PF10module13_25[index] = PF10module
        falsealarmrate10module13_25[index] = falsealarmrate10module
        IFLA10module13_25[index] = IFLA10module
        IFCCA10module13_25[index] = IFCCA10module
        IFMA10module13_25[index] = IFMA10module
        PMI10module13_25[index] = PMI10module
        PofB10module13_25[index] = PofB10module

        Precision20module13_25[index] = Precision20module
        Recall20module13_25[index] = Recall20module
        F120module13_25[index] = F120module
        Precisionx20module13_25[index] = Precisionx20module
        Recallx20module13_25[index] = Recallx20module
        F1x20module13_25[index] = F1x20module
        PF20module13_25[index] = PF20module
        falsealarmrate20module13_25[index] = falsealarmrate20module
        IFLA20module13_25[index] = IFLA20module
        IFCCA20module13_25[index] = IFCCA20module
        IFMA20module13_25[index] = IFMA20module
        PMI20module13_25[index] = PMI20module
        PofB20module13_25[index] = PofB20module

        normpercentpopt5module13_25[index] = normpercentpopt5module
        normpercentpopt10module13_25[index] = normpercentpopt10module
        normpercentpopt20module13_25[index] = normpercentpopt20module
        normWholepoptmodule13_25[index] = normWholepoptmodule

        index += 1

    # Ranking svm、Jar包还两个LTR （9个算法）
    index = 0
    other_results = run_jarAndLTRMethod(training_data_X, training_data_y, testing_data_X,
                                        testing_data_y, filename)
    for pred_prob, name in other_results:
        # 对于AdaRank的异常处理
        if name == "AdaRank" and (len(pred_prob) == 0 or (len(pred_prob) > 0 and math.isnan(pred_prob[0]))):
            fpa = Precision5module=Recall5module=F15module=Precisionx5module=Recallx5module=F1x5module= \
                  PF5module=falsealarmrate5module=IFLA5module=IFCCA5module=IFMA5module= PMI5module= PofB5module=\
                Precision10module= Recall10module=F110module=Precisionx10module=Recallx10module=F1x10module=\
            PF10module=falsealarmrate10module=IFLA10module=IFCCA10module=IFMA10module=PMI10module=PofB10module=\
                Precision20module=Recall20module=F120module=Precisionx20module=Recallx20module=F1x20module=PF20module=\
                falsealarmrate20module=IFLA20module=IFCCA20module=IFMA20module=PMI20module=PofB20module=\
                normpercentpopt5module=normpercentpopt10module=normpercentpopt20module=normWholepoptmodule=999
        else:
            fpa = PerformanceMeasure(testing_data_y, pred_prob).FPA()
            print("===========0.05===============")
            Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module, PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect', 'module').getSomePerformance()
            print("all 0.05 is:", Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module,
                  PF5module, falsealarmrate5module, IFLA5module, IFCCA5module, IFMA5module, PMI5module, PofB5module)
            print("===========0.1===============")
            Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module, F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module, PofB10module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect', 'module').getSomePerformance()
            print("all 0.1 is:", Precision10module, Recall10module, F110module, Precisionx10module, Recallx10module,
                  F1x10module, PF10module, falsealarmrate10module, IFLA10module, IFCCA10module, IFMA10module, PMI10module,
                  PofB10module)
            print("===========0.2===============")
            Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module, PofB20module = \
                PerformanceMeasure(
                    testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect', 'module').getSomePerformance()
            print("all 0.2 is:", Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module,
                  F1x20module, PF20module, falsealarmrate20module, IFLA20module, IFCCA20module, IFMA20module, PMI20module,
                  PofB20module)
            print("============POPT==============")
            normpercentpopt5module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.05, 'defect',
                                                        'module').PercentPOPT()
            normpercentpopt10module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.1, 'defect',
                                                         'module').PercentPOPT()
            normpercentpopt20module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 0.2, 'defect',
                                                         'module').PercentPOPT()
            normWholepoptmodule = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, avg_cc, 1, 'defect',
                                                     'module').POPT()

        # 写到26~34的结果list中
        fpa26_34[index] = fpa
        Precision5module26_34[index] = Precision5module
        Recall5module26_34[index] = Recall5module
        F15module26_34[index] = F15module
        Precisionx5module26_34[index] = Precisionx5module
        Recallx5module26_34[index] = Recallx5module
        F1x5module26_34[index] = F1x5module
        PF5module26_34[index] = PF5module
        falsealarmrate5module26_34[index] = falsealarmrate5module
        IFLA5module26_34[index] = IFLA5module
        IFCCA5module26_34[index] = IFCCA5module
        IFMA5module26_34[index] = IFMA5module
        PMI5module26_34[index] = PMI5module
        PofB5module26_34[index] = PofB5module

        Precision10module26_34[index] = Precision10module
        Recall10module26_34[index] = Recall10module
        F110module26_34[index] = F110module
        Precisionx10module26_34[index] = Precisionx10module
        Recallx10module26_34[index] = Recallx10module
        F1x10module26_34[index] = F1x10module
        PF10module26_34[index] = PF10module
        falsealarmrate10module26_34[index] = falsealarmrate10module
        IFLA10module26_34[index] = IFLA10module
        IFCCA10module26_34[index] = IFCCA10module
        IFMA10module26_34[index] = IFMA10module
        PMI10module26_34[index] = PMI10module
        PofB10module26_34[index] = PofB10module

        Precision20module26_34[index] = Precision20module
        Recall20module26_34[index] = Recall20module
        F120module26_34[index] = F120module
        Precisionx20module26_34[index] = Precisionx20module
        Recallx20module26_34[index] = Recallx20module
        F1x20module26_34[index] = F1x20module
        PF20module26_34[index] = PF20module
        falsealarmrate20module26_34[index] = falsealarmrate20module
        IFLA20module26_34[index] = IFLA20module
        IFCCA20module26_34[index] = IFCCA20module
        IFMA20module26_34[index] = IFMA20module
        PMI20module26_34[index] = PMI20module
        PofB20module26_34[index] = PofB20module

        normpercentpopt5module26_34[index] = normpercentpopt5module
        normpercentpopt10module26_34[index] = normpercentpopt10module
        normpercentpopt20module26_34[index] = normpercentpopt20module
        normWholepoptmodule26_34[index] = normWholepoptmodule
        index += 1

    return [filename] + [inde for inde in fpa1_12 + fpa13_25 + fpa26_34],  \
           [filename] + [inde for inde in Precision5module1_12 + Precision5module13_25+Precision5module26_34], \
           [filename] + [inde for inde in Recall5module1_12 + Recall5module13_25 + Recall5module26_34], \
           [filename] + [inde for inde in F15module1_12 + F15module13_25 + F15module26_34],\
           [filename] + [inde for inde in Precisionx5module1_12 + Precisionx5module13_25 + Precisionx5module26_34], \
           [filename] + [inde for inde in Recallx5module1_12 + Recallx5module13_25 + Recallx5module26_34], \
           [filename] + [inde for inde in F1x5module1_12 + F1x5module13_25 + F1x5module26_34], \
           [filename] + [inde for inde in PF5module1_12 + PF5module13_25 + PF5module26_34], \
           [filename] + [inde for inde in falsealarmrate5module1_12 + falsealarmrate5module13_25 + falsealarmrate5module26_34], \
           [filename] + [inde for inde in IFLA5module1_12 + IFLA5module13_25 + IFLA5module26_34], \
           [filename] + [inde for inde in IFCCA5module1_12 + IFCCA5module13_25 + IFCCA5module26_34], \
           [filename] + [inde for inde in IFMA5module1_12 + IFMA5module13_25 + IFMA5module26_34], \
           [filename] + [inde for inde in PMI5module1_12 + PMI5module13_25 + PMI5module26_34], \
           [filename] + [inde for inde in PofB5module1_12 + PofB5module13_25 + PofB5module26_34],\
           [filename] + [inde for inde in Precision10module1_12 + Precision10module13_25 + Precision10module26_34], \
           [filename] + [inde for inde in Recall10module1_12 + Recall10module13_25 + Recall10module26_34], \
           [filename] + [inde for inde in F110module1_12 + F110module13_25 + F110module26_34], \
           [filename] + [inde for inde in Precisionx10module1_12 + Precisionx10module13_25 + Precisionx10module26_34], \
           [filename] + [inde for inde in Recallx10module1_12 + Recallx10module13_25 + Recallx10module26_34], \
           [filename] + [inde for inde in F1x10module1_12 + F1x10module13_25 + F1x10module26_34], \
           [filename] + [inde for inde in PF10module1_12 + PF10module13_25 + PF10module26_34], \
           [filename] + [inde for inde in
                         falsealarmrate10module1_12 + falsealarmrate10module13_25 + falsealarmrate10module26_34], \
           [filename] + [inde for inde in IFLA10module1_12 + IFLA10module13_25 + IFLA10module26_34], \
           [filename] + [inde for inde in IFCCA10module1_12 + IFCCA10module13_25 + IFCCA10module26_34], \
           [filename] + [inde for inde in IFMA10module1_12 + IFMA10module13_25 + IFMA10module26_34], \
           [filename] + [inde for inde in PMI10module1_12 + PMI10module13_25 + PMI10module26_34], \
           [filename] + [inde for inde in PofB10module1_12 + PofB10module13_25 + PofB10module26_34], \
           [filename] + [inde for inde in Precision20module1_12 + Precision20module13_25 + Precision20module26_34], \
           [filename] + [inde for inde in Recall20module1_12 + Recall20module13_25 + Recall20module26_34], \
           [filename] + [inde for inde in F120module1_12 + F120module13_25 + F120module26_34], \
           [filename] + [inde for inde in Precisionx20module1_12 + Precisionx20module13_25 + Precisionx20module26_34], \
           [filename] + [inde for inde in Recallx20module1_12 + Recallx20module13_25 + Recallx20module26_34], \
           [filename] + [inde for inde in F1x20module1_12 + F1x20module13_25 + F1x20module26_34], \
           [filename] + [inde for inde in PF20module1_12 + PF20module13_25 + PF20module26_34], \
           [filename] + [inde for inde in
                         falsealarmrate20module1_12 + falsealarmrate20module13_25 + falsealarmrate20module26_34], \
           [filename] + [inde for inde in IFLA20module1_12 + IFLA20module13_25 + IFLA20module26_34], \
           [filename] + [inde for inde in IFCCA20module1_12 + IFCCA20module13_25 + IFCCA20module26_34], \
           [filename] + [inde for inde in IFMA20module1_12 + IFMA20module13_25 + IFMA20module26_34], \
           [filename] + [inde for inde in PMI20module1_12 + PMI20module13_25 + PMI20module26_34], \
           [filename] + [inde for inde in PofB20module1_12 + PofB20module13_25 + PofB20module26_34], \
           [filename] + [inde for inde in normpercentpopt5module1_12 + normpercentpopt5module13_25 + normpercentpopt5module26_34], \
           [filename] + [inde for inde in
                         normpercentpopt10module1_12 + normpercentpopt10module13_25 + normpercentpopt10module26_34], \
           [filename] + [inde for inde in
                         normpercentpopt20module1_12 + normpercentpopt20module13_25 + normpercentpopt20module26_34], \
           [filename] + [inde for inde in
                         normWholepoptmodule1_12 + normWholepoptmodule13_25 + normWholepoptmodule26_34]


def Predictdensity(training_data_X, training_data_y, testing_data_X, testing_data_y, Cla_training_data_y, testingcodeN,
                   filename):
    # 预测缺陷密度时，评价指标为OPT,PofBS20。x为19维的向量（即去掉了loc代码行数这一维），y为缺陷密度
    opt1_6 = [0, 0, 0, 0, 0, 0]
    opt7_15 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    pofbs20_1_6 = [0, 0, 0, 0, 0, 0]
    pofbs20_7_15 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 分类算法
    classification_results = ClassificationMethod(training_data_X, Cla_training_data_y, testing_data_X, "OPT")
    index = 0
    for pred_density, name in classification_results:
        pred_bugs = pred_density * np.array(testingcodeN)
        testing_data_bugs = testing_data_y * np.array(testingcodeN)
        OPT = PerformanceMeasure(testing_data_bugs, pred_bugs).OPT(testingcodeN)
        opt1_6[index] = OPT
        index += 1


if __name__ == '__main__':
    # 首先判断configuration_file.py中的is_remain_origin_bootstrap_csv的值
    # 如果is_remain_origin_bootstrap_csv为False，说明需要重新生成bootstrap的csv文件
    if not configuration_file().is_remain_origin_bootstrap_csv:
        # 那么首先删除之前的bootstrp文件夹
        if os.path.exists(configuration_file().bootstrap_dir):
            shutil.rmtree(configuration_file().bootstrap_dir)
        # 然后重新bootstrap
        for dataset, filename in Processing().import_single_data():
            print("开始处理文件：{0}".format(filename))
            Processing().split_train_test_csv(dataset, filename)

    # 切分完之后，train和test的csv均已存在设置好的文件夹中，此时可以读取进行算法的测试了
    # 测试结果用5个列表保存起来，然后所有bootstrap数据结束后调用写Excel函数写进去

    print("开始时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()
    for train_data_x_list, train_data_y_list, test_data_x_list, test_data_y_list, filename in Processing().read_bootstrap_csv():
        fpa_list = []
        Precision5module_list = []
        Recall5module_list = []
        F15module_list = []
        Precisionx5module_list = []
        Recallx5module_list = []
        F1x5module_list = []
        PF5module_list = []
        falsealarmrate5module_list = []
        IFLA5module_list = []
        IFCCA5module_list = []
        IFMA5module_list = []
        PMI5module_list = []
        PofB5module_list = []

        Precision10module_list = []
        Recall10module_list = []
        F110module_list = []
        Precisionx10module_list = []
        Recallx10module_list = []
        F1x10module_list = []
        PF10module_list = []
        falsealarmrate10module_list = []
        IFLA10module_list = []
        IFCCA10module_list = []
        IFMA10module_list = []
        PMI10module_list = []
        PofB10module_list = []

        Precision20module_list = []
        Recall20module_list = []
        F120module_list = []
        Precisionx20module_list = []
        Recallx20module_list = []
        F1x20module_list = []
        PF20module_list = []
        falsealarmrate20module_list = []
        IFLA20module_list = []
        IFCCA20module_list = []
        IFMA20module_list = []
        PMI20module_list = []
        PofB20module_list = []

        normpercentpopt5module_list = []
        normpercentpopt10module_list = []
        normpercentpopt20module_list = []
        normWholepoptmodule_list = []

        fpa_list.append(header)
        Precision5module_list.append(header)
        Recall5module_list.append(header)
        F15module_list.append(header)
        Precisionx5module_list.append(header)
        Recallx5module_list.append(header)
        F1x5module_list.append(header)
        PF5module_list.append(header)
        falsealarmrate5module_list.append(header)
        IFLA5module_list.append(header)
        IFCCA5module_list.append(header)
        IFMA5module_list.append(header)
        PMI5module_list.append(header)
        PofB5module_list.append(header)

        Precision10module_list.append(header)
        Recall10module_list.append(header)
        F110module_list.append(header)
        Precisionx10module_list.append(header)
        Recallx10module_list.append(header)
        F1x10module_list.append(header)
        PF10module_list.append(header)
        falsealarmrate10module_list.append(header)
        IFLA10module_list.append(header)
        IFCCA10module_list.append(header)
        IFMA10module_list.append(header)
        PMI10module_list.append(header)
        PofB10module_list.append(header)

        Precision20module_list.append(header)
        Recall20module_list.append(header)
        F120module_list.append(header)
        Precisionx20module_list.append(header)
        Recallx20module_list.append(header)
        F1x20module_list.append(header)
        PF20module_list.append(header)
        falsealarmrate20module_list.append(header)
        IFLA20module_list.append(header)
        IFCCA20module_list.append(header)
        IFMA20module_list.append(header)
        PMI20module_list.append(header)
        PofB20module_list.append(header)

        normpercentpopt5module_list.append(header)
        normpercentpopt10module_list.append(header)
        normpercentpopt20module_list.append(header)
        normWholepoptmodule_list.append(header)
        print(len(train_data_x_list), len(train_data_y_list), len(test_data_x_list), len(test_data_y_list))
        print("================================")
        for i in range(len(train_data_x_list)):
            # 分别取出每一次bootstrap的数据
            training_data_X = np.array(train_data_x_list[i])
            training_data_y = np.array(train_data_y_list[i])
            testing_data_X = np.array(test_data_x_list[i])
            testing_data_y = np.array(test_data_y_list[i])
            Cla_training_data_y = [1 if y > 0 else 0 for y in training_data_y]


            print("********************************************")
            print(training_data_X.shape)
            print(training_data_y.shape)
            print(testing_data_X.shape)
            print(testing_data_y.shape)
            print("********************************************")
            # 还要得到loc 和 cc 对应的list才行
            # 在之前的数据集，也就是ant-1.3这样的数据集中，avg_cc是倒数第二列，loc是正数第11列
            avg_cc = [i[-1] for i in testing_data_X]
            testingcodeN = [i[10] for i in testing_data_X]

            Predictbugs(training_data_X, training_data_y, testing_data_X,
                        testing_data_y, Cla_training_data_y, filename, i, avg_cc, testingcodeN)

            fpa, Precision5module, Recall5module, F15module, Precisionx5module, Recallx5module, F1x5module, \
            PF5module, falsealarmrate5module, IFLA5module,  IFCCA5module, \
            IFMA5module, PMI5module, PofB5module, Precision10module, Recall10module, F110module,\
            Precisionx10module, Recallx10module, F1x10module, PF10module, falsealarmrate10module, \
            IFLA10module,  IFCCA10module, IFMA10module, PMI10module, PofB10module, Precision20module,\
            Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module,\
            falsealarmrate20module, IFLA20module,  IFCCA20module, IFMA20module, PMI20module, PofB20module,\
            normpercentpopt5module, normpercentpopt10module, normpercentpopt20module, normWholepoptmodule   \
                = Predictbugs(training_data_X, training_data_y, testing_data_X,
                                                   testing_data_y, Cla_training_data_y, filename, i, avg_cc, testingcodeN)

            fpa_list.append(fpa)
            Precision5module_list.append(Precision5module)
            Recall5module_list.append(Recall5module)
            F15module_list.append(F15module)
            Precisionx5module_list.append(Precisionx5module)
            Recallx5module_list.append(Recallx5module)
            F1x5module_list.append(F1x5module)
            PF5module_list.append(PF5module)
            falsealarmrate5module_list.append(falsealarmrate5module)
            IFLA5module_list.append(IFLA5module)
            IFCCA5module_list.append(IFCCA5module)
            IFMA5module_list.append(IFMA5module)
            PMI5module_list.append(PMI5module)
            PofB5module_list.append(PofB5module)

            Precision10module_list.append(Precision10module)
            Recall10module_list.append(Recall10module)
            F110module_list.append(F110module)
            Precisionx10module_list.append(Precisionx10module)
            Recallx10module_list.append(Recallx10module)
            F1x10module_list.append(F1x10module)
            PF10module_list.append(PF10module)
            falsealarmrate10module_list.append(falsealarmrate10module)
            IFLA10module_list.append(IFLA10module)
            IFCCA10module_list.append(IFCCA10module)
            IFMA10module_list.append(IFMA10module)
            PMI10module_list.append(PMI10module)
            PofB10module_list.append(PofB10module)

            Precision20module_list.append(Precision20module)
            Recall20module_list.append(Recall20module)
            F120module_list.append(F120module)
            Precisionx20module_list.append(Precisionx20module)
            Recallx20module_list.append(Recallx20module)
            F1x20module_list.append(F1x20module)
            PF20module_list.append(PF20module)
            falsealarmrate20module_list.append(falsealarmrate20module)
            IFLA20module_list.append(IFLA20module)
            IFCCA20module_list.append(IFCCA20module)
            IFMA20module_list.append(IFMA20module)
            PMI20module_list.append(PMI20module)
            PofB20module_list.append(PofB20module)

            normpercentpopt5module_list.append(normpercentpopt5module)
            normpercentpopt10module_list.append(normpercentpopt10module)
            normpercentpopt20module_list.append(normpercentpopt20module)
            normWholepoptmodule_list.append(normWholepoptmodule)

        result_path = configuration_file().save_PredBugCountResult_dir

        fpa_csv_name = filename + "_fpa.xlsx"
        fpa_result_path = os.path.join(result_path, fpa_csv_name)
        Processing().write_excel(fpa_result_path, fpa_list)

        Precision5module_csv_name = filename + "_Precision5module.xlsx"
        Precision5module_result_path = os.path.join(result_path, Precision5module_csv_name)
        Processing().write_excel(Precision5module_result_path, Precision5module_list)

        Recall5module_csv_name = filename + "_Recall5module.xlsx"
        Recall5module_result_path = os.path.join(result_path, Recall5module_csv_name)
        Processing().write_excel(Recall5module_result_path, Recall5module_list)

        F15module_csv_name = filename + "_F15module.xlsx"
        F15module_result_path = os.path.join(result_path, F15module_csv_name)
        Processing().write_excel(F15module_result_path, F15module_list)

        Precisionx5module_csv_name = filename + "_Precisionx5module.xlsx"
        Precisionx5module_result_path = os.path.join(result_path, Precisionx5module_csv_name)
        Processing().write_excel(Precisionx5module_result_path, Precisionx5module_list)

        Recallx5module_csv_name = filename + "_Recallx5module.xlsx"
        Recallx5module_result_path = os.path.join(result_path, Recallx5module_csv_name)
        Processing().write_excel(Recallx5module_result_path, Recallx5module_list)

        F1x5module_csv_name = filename + "_F1x5module.xlsx"
        F1x5module_result_path = os.path.join(result_path, F1x5module_csv_name)
        Processing().write_excel(F1x5module_result_path, F1x5module_list)

        PF5module_csv_name = filename + "_PF5module.xlsx"
        PF5module_result_path = os.path.join(result_path, PF5module_csv_name)
        Processing().write_excel(PF5module_result_path, PF5module_list)

        falsealarmrate5module_csv_name = filename + "_falsealarmrate5module.xlsx"
        falsealarmrate5module_result_path = os.path.join(result_path, falsealarmrate5module_csv_name)
        Processing().write_excel(falsealarmrate5module_result_path, falsealarmrate5module_list)

        IFLA5module_csv_name = filename + "_IFLA5module.xlsx"
        IFLA5module_result_path = os.path.join(result_path, IFLA5module_csv_name)
        Processing().write_excel(IFLA5module_result_path, IFLA5module_list)

        IFCCA5module_csv_name = filename + "_IFCCA5module.xlsx"
        IFCCA5module_result_path = os.path.join(result_path, IFCCA5module_csv_name)
        Processing().write_excel(IFCCA5module_result_path, IFCCA5module_list)

        IFMA5module_csv_name = filename + "_IFMA5module.xlsx"
        IFMA5module_result_path = os.path.join(result_path, IFMA5module_csv_name)
        Processing().write_excel(IFMA5module_result_path, IFMA5module_list)

        PMI5module_csv_name = filename + "_PMI5module.xlsx"
        PMI5module_result_path = os.path.join(result_path, PMI5module_csv_name)
        Processing().write_excel(PMI5module_result_path, PMI5module_list)

        PofB5module_csv_name = filename + "_PofB5module.xlsx"
        PofB5module_result_path = os.path.join(result_path, PofB5module_csv_name)
        Processing().write_excel(PofB5module_result_path, PofB5module_list)


        Precision10module_csv_name = filename + "_Precision10module.xlsx"
        Precision10module_result_path = os.path.join(result_path, Precision10module_csv_name)
        Processing().write_excel(Precision10module_result_path, Precision10module_list)

        Recall10module_csv_name = filename + "_Recall10module.xlsx"
        Recall10module_result_path = os.path.join(result_path, Recall10module_csv_name)
        Processing().write_excel(Recall10module_result_path, Recall10module_list)

        F110module_csv_name = filename + "_F110module.xlsx"
        F110module_result_path = os.path.join(result_path, F110module_csv_name)
        Processing().write_excel(F110module_result_path, F110module_list)

        Precisionx10module_csv_name = filename + "_Precisionx10module.xlsx"
        Precisionx10module_result_path = os.path.join(result_path, Precisionx10module_csv_name)
        Processing().write_excel(Precisionx10module_result_path, Precisionx10module_list)

        Recallx10module_csv_name = filename + "_Recallx10module.xlsx"
        Recallx10module_result_path = os.path.join(result_path, Recallx10module_csv_name)
        Processing().write_excel(Recallx10module_result_path, Recallx10module_list)

        F1x10module_csv_name = filename + "_F1x10module.xlsx"
        F1x10module_result_path = os.path.join(result_path, F1x10module_csv_name)
        Processing().write_excel(F1x10module_result_path, F1x10module_list)

        PF10module_csv_name = filename + "_PF10module.xlsx"
        PF10module_result_path = os.path.join(result_path, PF10module_csv_name)
        Processing().write_excel(PF10module_result_path, PF10module_list)

        falsealarmrate10module_csv_name = filename + "_falsealarmrate10module.xlsx"
        falsealarmrate10module_result_path = os.path.join(result_path, falsealarmrate10module_csv_name)
        Processing().write_excel(falsealarmrate10module_result_path, falsealarmrate10module_list)

        IFLA10module_csv_name = filename + "_IFLA10module.xlsx"
        IFLA10module_result_path = os.path.join(result_path, IFLA10module_csv_name)
        Processing().write_excel(IFLA10module_result_path, IFLA10module_list)

        IFCCA10module_csv_name = filename + "_IFCCA10module.xlsx"
        IFCCA10module_result_path = os.path.join(result_path, IFCCA10module_csv_name)
        Processing().write_excel(IFCCA10module_result_path, IFCCA10module_list)

        IFMA10module_csv_name = filename + "_IFMA10module.xlsx"
        IFMA10module_result_path = os.path.join(result_path, IFMA10module_csv_name)
        Processing().write_excel(IFMA10module_result_path, IFMA10module_list)

        PMI10module_csv_name = filename + "_PMI10module.xlsx"
        PMI10module_result_path = os.path.join(result_path, PMI10module_csv_name)
        Processing().write_excel(PMI10module_result_path, PMI10module_list)

        PofB10module_csv_name = filename + "_PofB10module.xlsx"
        PofB10module_result_path = os.path.join(result_path, PofB10module_csv_name)
        Processing().write_excel(PofB10module_result_path, PofB10module_list)

        Precision20module_csv_name = filename + "_Precision20module.xlsx"
        Precision20module_result_path = os.path.join(result_path, Precision20module_csv_name)
        Processing().write_excel(Precision20module_result_path, Precision20module_list)

        Recall20module_csv_name = filename + "_Recall20module.xlsx"
        Recall20module_result_path = os.path.join(result_path, Recall20module_csv_name)
        Processing().write_excel(Recall20module_result_path, Recall20module_list)

        F120module_csv_name = filename + "_F120module.xlsx"
        F120module_result_path = os.path.join(result_path, F120module_csv_name)
        Processing().write_excel(F120module_result_path, F120module_list)

        Precisionx20module_csv_name = filename + "_Precisionx20module.xlsx"
        Precisionx20module_result_path = os.path.join(result_path, Precisionx20module_csv_name)
        Processing().write_excel(Precisionx20module_result_path, Precisionx20module_list)

        Recallx20module_csv_name = filename + "_Recallx20module.xlsx"
        Recallx20module_result_path = os.path.join(result_path, Recallx20module_csv_name)
        Processing().write_excel(Recallx20module_result_path, Recallx20module_list)

        F1x20module_csv_name = filename + "_F1x20module.xlsx"
        F1x20module_result_path = os.path.join(result_path, F1x20module_csv_name)
        Processing().write_excel(F1x20module_result_path, F1x20module_list)

        PF20module_csv_name = filename + "_PF20module.xlsx"
        PF20module_result_path = os.path.join(result_path, PF20module_csv_name)
        Processing().write_excel(PF20module_result_path, PF20module_list)

        falsealarmrate20module_csv_name = filename + "_falsealarmrate20module.xlsx"
        falsealarmrate20module_result_path = os.path.join(result_path, falsealarmrate20module_csv_name)
        Processing().write_excel(falsealarmrate20module_result_path, falsealarmrate20module_list)

        IFLA20module_csv_name = filename + "_IFLA20module.xlsx"
        IFLA20module_result_path = os.path.join(result_path, IFLA20module_csv_name)
        Processing().write_excel(IFLA20module_result_path, IFLA20module_list)

        IFCCA20module_csv_name = filename + "_IFCCA20module.xlsx"
        IFCCA20module_result_path = os.path.join(result_path, IFCCA20module_csv_name)
        Processing().write_excel(IFCCA20module_result_path, IFCCA20module_list)

        IFMA20module_csv_name = filename + "_IFMA20module.xlsx"
        IFMA20module_result_path = os.path.join(result_path, IFMA20module_csv_name)
        Processing().write_excel(IFMA20module_result_path, IFMA20module_list)

        PMI20module_csv_name = filename + "_PMI20module.xlsx"
        PMI20module_result_path = os.path.join(result_path, PMI20module_csv_name)
        Processing().write_excel(PMI20module_result_path, PMI20module_list)

        PofB20module_csv_name = filename + "_PofB20module.xlsx"
        PofB20module_result_path = os.path.join(result_path, PofB20module_csv_name)
        Processing().write_excel(PofB20module_result_path, PofB20module_list)

        normpercentpopt5module_csv_name = filename + "_normpercentpopt5module.xlsx"
        normpercentpopt5module_result_path = os.path.join(result_path, normpercentpopt5module_csv_name)
        Processing().write_excel(normpercentpopt5module_result_path, normpercentpopt5module_list)

        normpercentpopt10module_csv_name = filename + "_normpercentpopt10module.xlsx"
        normpercentpopt10module_result_path = os.path.join(result_path, normpercentpopt10module_csv_name)
        Processing().write_excel(normpercentpopt10module_result_path, normpercentpopt10module_list)

        normpercentpopt20module_csv_name = filename + "_normpercentpopt20module.xlsx"
        normpercentpopt20module_result_path = os.path.join(result_path, normpercentpopt20module_csv_name)
        Processing().write_excel(normpercentpopt20module_result_path, normpercentpopt20module_list)

        normWholepoptmodule_csv_name = filename + "_normWholepoptmodule.xlsx"
        normWholepoptmodule_result_path = os.path.join(result_path, normWholepoptmodule_csv_name)
        Processing().write_excel(normWholepoptmodule_result_path, normWholepoptmodule_list)


    print("结束时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    endtime = time.time()
    print('耗用时间:', endtime - starttime, '秒')
    # pass
