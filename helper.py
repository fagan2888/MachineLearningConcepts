import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import seaborn as sns

from sklearn import svm
from sklearn import cross_validation as cv
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve


def get_number_of_flights(df):
    """
    Computes the total number of flights on each day.

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame.
    Each day is grouped, and the number of flights is in a column named "Flights".
    """

    result = df.copy()
    result['Flights'] = 1
    result = result.groupby(['Month', 'DayofMonth']).sum()

    result = result.drop(['Cancelled'], axis=1)

    return result


def get_cancellations(df):
    """
    Computes the total number of cancellations on each day.

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame.
    Each day is grouped, and the number of cancellations is in a column named "Cancelled".
    """

    result = df.copy()
    result = result.groupby(['Month', 'DayofMonth']).sum()

    return result


def plot_outliers(df, column, bins):
    """
    Finds and visualizes outliers.

    Parameters
    ----------
    df: A pandas DataFrame.
    column: A string.
    bins: A numpy array. Histogram bins.

    Returns
    -------
    A Matplotlib Axes instance.
    """

    x = df[column]

    mu = x.mean()
    sig = x.std()
    lb = mu - 3 * sig
    ub = mu + 3 * sig

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(column)
    ax.hist(x, bins=bins, color=sns.xkcd_rgb["denim blue"], alpha=0.5, label='Inliers')
    ax.hist(x[(x < lb) | (x > ub)], bins=bins, color='r', alpha=0.5, label='Outliers')
    ax.legend(loc='best')

    return ax


def plot_2d(df_x, df_y, col_x, col_y):
    """
    Creates a two-diemnsional plot of bivariate distribution.

    Parameters
    ----------
    df_x: A pandas.DataFrame.
    df_y: A pandas.DataFrame.
    col_x: A string. The column in "df_x" that will be used as the x variable.
    col_y: A string. The column in "df_y" that will be used as the x variable.

    Returns
    -------
    A matplotlib.Axes instance.
    """

    x = df_x[col_x]
    y = df_y[col_y]
    mu_x = x.mean()
    sig_x = x.std()
    mu_y = y.mean()
    sig_y = y.std()
    lb_x = mu_x - 3 * sig_x
    ub_x = mu_x + 3 * sig_x
    lb_y = mu_y - 3 * sig_y
    ub_y = mu_y + 3 * sig_y

    x_i = x[~((x < lb_x) | (x > ub_x) | (y < lb_y) | (y > ub_y))]
    y_i = y[~((x < lb_x) | (x > ub_x) | (y < lb_y) | (y > ub_y))]
    x_o = x[(x < lb_x) | (x > ub_x) | (y < lb_y) | (y > ub_y)]
    y_o = y[(x < lb_x) | (x > ub_x) | (y < lb_y) | (y > ub_y)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    ax.scatter(x_i, y_i, color='b', s=60, alpha=.5, label='Inliers')
    ax.scatter(x_o, y_o, color='r', s=60, marker='*', label='Outliers', alpha=.5)

    ax.legend(loc='upper left')

    return ax


def dbscan_outliers(df):
    """
    Find outliers (noise points) using DBSCAN.

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A tuple of (a sklearn.DBSCAN instance, a pandas.DataFrame)
    """

    scaler = StandardScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)

    dbs = DBSCAN()

    db = dbs.fit(scaled)
    outliers = dbs.fit_predict(scaled)

    df_o = df.ix[np.nonzero(outliers)]

    return db, df_o


def select_features(X, y, random_state, kernel='linear', C=1.0, num_attributes=3):
    """
    Uses Support Vector Classifier as the estimator to rank features
    with Recursive Feature Eliminatin.

    Parameters
    ----------
    X: A pandas.DataFrame. Attributes.
    y: A pandas.DataFrame. Labels.
    random_state: A RandomState instance. Used in SVC().
    kernel: A string. Used in SVC(). Default: "linear".
    C: A float. Used in SVC(). Default: 1.0.
    num_attributes: An int. The number of features to select in RFE. Default: 3.

    Returns
    -------
    A 3-tuple of (RFE, np.ndarray, np.ndarray)
    model: An RFE instance.
    columns: Selected features.
    ranking: The feature ranking. Selected features are assigned rank 1.
    """

    rfe = RFE(svm.SVC(C, kernel, random_state=random_state), num_attributes)
    model = rfe.fit(X, y.values.ravel())
    columns = list()

    for idx, label in enumerate(X):
        if rfe.support_[idx]:
            columns.append(label)

    ranking = rfe.ranking_

    return model, columns, ranking


def pipeline_anova_svm(X, y, random_state, k=3, kernel='linear'):
    """
    Selects top k=3 features with a pipeline that uses ANOVA F-value
    and a Support Vector Classifier.

    Parameters
    ----------
    X: A pandas.DataFrame. Attributes.
    y: A pandas.DataFrame. Labels.
    random_state: A RandomState instance. Used in SVC().
    k: An int. The number of features to select. Default: 3
    kernel: A string. Used by SVC(). Default: 'linear'

    Returns
    -------
    A 2-tuple of (Pipeline, np.ndarray)
    model: An ANOVA SVM-C pipeline.
    predictions: Classifications predicted by the pipeline.
    """

    anova = SelectKBest(f_regression, k=k)
    svc = svm.SVC(kernel=kernel, random_state=random_state)
    anova_svm = Pipeline([('anova', anova), ('svc', svc)])
    model = anova_svm.fit(X, y.values.ravel())
    predictions = anova_svm.predict(X)

    return model, predictions


def grid_search_c(X, y, split_rs, svc_rs, c_vals, test_size=0.2, kernel='linear'):
    """

    Parameters
    ----------
    X: A pandas.DataFrame. Attributes.
    y: A pandas.DataFrame. Labels.
    split_rs: A RandomState instance. Used in train_test_split().
    svc_rs: A RandomState instance. Used in SVC().
    c_vals: A np.array. A list of parameter settings to try as vlues.
    test_size: A float. Used in train_test_split(). Default: 0.2
    kernel: A string. Used by SVC(). Default: 'linear'

    Returns
    -------
    A 3-tuple of (GridSearchCV, float, float)
    model: A GridSearchCV instance.
    best_C: The value of C that gave the highest score.
    best_cv_score: Score of best_C on the hold out data.
    """

    (x_trn, x_tst, y_trn, y_tst) = cv.train_test_split(X, y.values.ravel(), test_size=test_size, random_state=split_rs)

    svc = svm.SVC(kernel=kernel, random_state=svc_rs)

    clf = GridSearchCV(estimator=svc, param_grid=dict(C=c_vals))

    model = clf.fit(x_trn, y_trn)

    best_C = clf.best_estimator_.C

    best_cv_score = clf.best_score_

    return model, best_C, best_cv_score


def plot_validation_curve(X, y, param_range):
    """
    Computes and displays the validation curve for SVC.

    Parameters
    ----------
    X: A pandas.DataFrame. Attributes.
    y: A pandas.DataFrame. Labels.
    param_range: The values of the parameter that will be evaluated.

    Returns
    -------
    A maplotlib.Axes instance.
    """

    trn_scr, tst_scr = validation_curve(svm.SVC(), X, y.values.ravel(), param_name="gamma",
                                        param_range=param_range, cv=5, scoring="accuracy")

    trn_scr_mu = np.mean(trn_scr, axis=1)
    tst_scr_mu = np.mean(tst_scr, axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))

    trn_color = sns.xkcd_rgb["denim blue"]
    ax.semilogx(param_range, trn_scr_mu, label="Training Score", marker='d', lw=2, color=trn_color)

    tst_color = sns.xkcd_rgb["medium green"]
    ax.semilogx(param_range, tst_scr_mu, label="CV Score", marker='d', lw=2, color=tst_color)

    ax.set_title("Validation Curve with SVM", fontsize=18)
    ax.set_xlabel('$\gamma$', fontsize=18)
    ax.set_ylabel("Score", fontsize=18)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="best", fontsize=18)

    return ax
