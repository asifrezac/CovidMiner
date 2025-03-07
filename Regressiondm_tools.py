import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz


def data_prep():
    # read the veteran dataset
    df = pd.read_csv('D3.csv')
    
    # one-hot encoding
    df = pd.get_dummies(df)
    
    y = df['covid19_positive']
    X = df.drop(['covid19_positive'], axis=1)

    # setting random state
    rs = 10

    X_mat = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    return df,X,y,X_train, X_test, y_train, y_test