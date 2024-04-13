import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance


df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
df.Traffic.value_counts()
df_minor = df[(df['Traffic']==3)|(df['Traffic']==1)|(df['Traffic']==0)]
df_major = df.drop(df_minor.index)
X = df_major.drop(['Traffic'],axis=1)
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)

klabel=kmeans.labels_
df_major['klabel']=klabel
df_major['klabel'].value_counts()
cols = list(df_major)
cols.insert(44, cols.pop(cols.index('Traffic')))
df_major = df_major.loc[:, cols]

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)
result['Traffic'].value_counts()
result = result.drop(['klabel'],axis=1)
result = result.append(df_minor)
result.to_csv('./dataset/wustl_iiot_sample_km.csv',index=0)
df=pd.read_csv('./dataset/wustl_iiot_sample_km.csv')
X = df.drop(['Traffic'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2>=0.9:
        break
X_fs = df[fs].values
# -*- coding: utf-8 -*-
import numpy as np

def count_vals(x):
    vals = np.unique(x)
    occ = np.zeros(shape = vals.shape)
    for i in range(vals.size):
        occ[i] = np.sum(x == vals[i])
    return occ

def entropy(x):
    n = float(x.shape[0])
    ocurrence = count_vals(x)
    px = ocurrence / n
    return -1* np.sum(px*np.log2(px))

def symmetricalUncertain(x,y):
    n = float(y.shape[0])
    vals = np.unique(y)
    # Computing Entropy for the feature x.
    Hx = entropy(x)
    # Computing Entropy for the feature y.
    Hy = entropy(y)
    #Computing Joint entropy between x and y.
    partial = np.zeros(shape = (vals.shape[0]))
    for i in range(vals.shape[0]):
       partial[i] = entropy(x[y == vals[i]])

    partial[np.isnan(partial)==1] = 0
    py = count_vals(y).astype(dtype = 'float64') / n
    Hxy = np.sum(py[py > 0]*partial)
    IG = Hx-Hxy
    return 2*IG/(Hx+Hy)

def suGroup(x, n):
    m = x.shape[0]
    x = np.reshape(x, (n,m/n)).T
    m = x.shape[1]
    SU_matrix = np.zeros(shape = (m,m))
    for j in range(m-1):
        x2 = x[:,j+1::]
        y = x[:,j]
        temp = np.apply_along_axis(symmetricalUncertain, 0, x2, y)
        for k in range(temp.shape[0]):
            SU_matrix[j,j+1::] = temp
            SU_matrix[j+1::,j] = temp

    return 1/float(m-1)*np.sum(SU_matrix, axis = 1)

def isprime(a):
    return all(a % i for i in xrange(2, a))


"""
get
"""

def get_i(a):
    if isprime(a):
        a -= 1
    return filter(lambda x: a % x == 0, range(2,a))



class FCBF:

    idx_sel = []


    def __init__(self, th = 0.01):
        '''
        Parameters
        ---------------
            th = The initial threshold
        '''
        self.th = th


    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)
        SU_list = SU_vec[SU_vec > self.th]
        SU_list[::-1].sort()

        m = x[:,SU_vec > self.th].shape
        x_sorted = np.zeros(shape = m)

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:,i] = x[:,ind].copy()
            self.idx_sel.append(ind)

        """
        Second Stage: Identify relationships between feature to remove redundancy.
        """
        j = 0
        while True:
            """
            Stopping Criteria:The search finishes
            """
            if j >= x_sorted.shape[1]: break
            y = x_sorted[:,j].copy()
            x_list = x_sorted[:,j+1:].copy()
            if x_list.shape[1] == 0: break


            SU_list_2 = SU_list[j+1:]
            SU_x = np.apply_along_axis(symmetricalUncertain, 0,
                                       x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1
            if to_remove.size > 0:
                x_sorted = np.delete(x_sorted, to_remove, axis = 1)
                SU_list = np.delete(SU_list, to_remove, axis = 0)
                to_remove.sort()
                for r in reversed(to_remove):
                    self.idx_sel.remove(self.idx_sel[r])
            j = j + 1

    def fit_transform(self, x, y):
        '''
        This function fits the feature selection
        algorithm and returns the resulting subset.

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.fit(x, y)
        return x[:,self.idx_sel]

    def transform(self, x):
        '''
        This function applies the selection
        to the vector x.

        Parameters
        ---------------
            x = dataset  [NxM]
        '''
        return x[:, self.idx_sel]


class FCBFK(FCBF):

    idx_sel = []


    def __init__(self, k = 10):
        '''
        Parameters
        ---------------
            k = Number of features to include in the
            subset.
        '''
        self.k = k


    def fit(self, x, y):
        '''
        This function executes FCBFK algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)

        SU_list = SU_vec[SU_vec > 0]
        SU_list[::-1].sort()

        m = x[:,SU_vec > 0].shape
        x_sorted = np.zeros(shape = m)

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:,i] = x[:,ind].copy()
            self.idx_sel.append(ind)

        """
        Second Stage: Identify relationships between features to remove redundancy with stopping
        criteria (features in x_best == k).
        """
        j = 0
        while True:
            y = x_sorted[:,j].copy()
            SU_list_2 = SU_list[j+1:]
            x_list = x_sorted[:,j+1:].copy()

            """
            Stopping Criteria:The search finishes
            """
            if x_list.shape[1] == 0: break


            SU_x = np.apply_along_axis(symmetricalUncertain, 0,
                                       x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1
            if to_remove.size > 0 and x.shape[1] > self.k:

                for i in reversed(to_remove):

                    x_sorted = np.delete(x_sorted, i, axis = 1)
                    SU_list = np.delete(SU_list, i, axis = 0)
                    self.idx_sel.remove(self.idx_sel[i])
                    if x_sorted.shape[1] == self.k: break

            if x_list.shape[1] == 1 or x_sorted.shape[1] == self.k:
                break
            j = j + 1

        if len(self.idx_sel) > self.k:
            self.idx_sel = self.idx_sel[:self.k]



"""
FCBFiP - Fast Correlation Based Filter in Pieces
"""

class FCBFiP(FCBF):

    idx_sel = []


    def __init__(self, k = 10, npieces = 2):
        '''
        Parameters
        ---------------
            k = Number of features to include in the
            subset.
            npieces = Number of pieces to divide the
            feature space.
        '''
        self.k = k
        self.npieces = npieces

    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''

        """
        First Stage: Computing the SU for each feature with the response. We sort the
        features. When we have a prime number of features we remove the last one from the
        sorted features list.
        """
        m = x.shape
        nfeaturesPieces = int(m[1] / float(self.npieces))
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)

        x_sorted = np.zeros(shape = m, dtype = 'float64')
        idx_sorted = np.zeros(shape = m[1], dtype = 'int64')
        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = -1
            idx_sorted[i]= ind
            x_sorted[:,i] = x[:,ind].copy()

        if isprime(m[1]):
            x_sorted = np.delete(x_sorted, m[1]-1, axis = 1 )
            ind_prime = idx_sorted[m[1]-1]
            idx_sorted = np.delete(idx_sorted, m[1]-1)
            #m = x_sorted.shape
        """
        Second Stage: Identify relationships between features into its vecinity
        to remove redundancy with stopping criteria (features in x_best == k).
        """

        x_2d = np.reshape(x_sorted.T, (self.npieces, nfeaturesPieces*m[0])).T

        SU_x =  np.apply_along_axis(suGroup, 0, x_2d, nfeaturesPieces)
        SU_x = np.reshape(SU_x.T, (self.npieces*nfeaturesPieces,))
        idx_sorted2 = np.zeros(shape = idx_sorted.shape, dtype = 'int64')
        SU_x[np.isnan(SU_x)] = 1

        for i in range(idx_sorted.shape[0]):
            ind =  np.argmin(SU_x)
            idx_sorted2[i] = idx_sorted[ind]
            SU_x[ind] = 10

        """
        Scoring step
        """
        self.scores = np.zeros(shape = m[1], dtype = 'int64')

        for i in range(m[1]):
            if i in idx_sorted:
                self.scores[i] = np.argwhere(i == idx_sorted) + np.argwhere(i == idx_sorted2)
        if isprime(m[1]):
            self.scores[ind_prime] = 2*m[1]
        self.set_k(self.k)


    def set_k(self, k):
        self.k = k
        scores_temp = -1*self.scores

        self.idx_sel = np.zeros(shape = self.k, dtype = 'int64')
        for i in range(self.k):
            ind =  np.argmax(scores_temp)
            scores_temp[ind] = -100000000
            self.idx_sel[i] = ind
fcbf = FCBFK(k = 20)
X_fss = fcbf.fit_transform(X_fs,y)
X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
pd.Series(y_train).value_counts()
X_train, y_train = smote.fit_resample(X_train, y_train)
from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000})
X_train, y_train = smote.fit_resample(X_train, y_train)
pd.Series(y_train).value_counts()
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train)
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
rf_train=rf.predict(X_train)
rf_test=rf.predict(X_test)
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X_train,y_train)
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
dt_train=dt.predict(X_train)
dt_test=dt.predict(X_test)
from sklearn.ensemble import AdaBoostClassifier
et = AdaBoostClassifier(random_state = 0)
et.fit(X_train,y_train)
et_score=et.score(X_test,y_test)
y_predict=et.predict(X_test)
y_true=y_test
from sklearn.naive_bayes import BernoulliNB
mnb = BernoulliNB()
#mnb.fit(X_train, Y_train)
mnb.fit(X_train,y_train)
nb_train=mnb.predict(X_train)
nb_test=mnb.predict(X_test)
from sklearn.neural_network import MLPClassifier
clf_Classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(39, 30, 2), random_state=1)
clf_Classifier.fit(X_train,y_train)

myclf_Classifier=clf_Classifier.score(X_test,y_test)

y_predict=clf_Classifier.predict(X_test)
y_true=y_test
mlp_train=clf_Classifier.predict(X_train)
mlp_test=clf_Classifier.predict(X_test)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

import tensorflow as tf

import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

nn = Sequential()


#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 37))
nn.add(Dense(20, activation='relu', kernel_initializer='glorot_uniform'))
nn.add(Dense(20, activation='relu', kernel_initializer='glorot_uniform'))
nn.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
nn.fit(X_train,y_train, batch_size = 256, epochs = 10)
X_test = X_test.astype(int)

y_pred = nn.predict(X_test)

y_pred = y_pred.astype(int)
nn_train=nn.predict(X_train)
nn_test=nn.predict(X_test)

base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'adaboost': et_train.ravel(),
     'NN': nn_train.ravel(),
       'mlp': mlp_train.ravel(),
       'nb': nb_train.ravel()
    })
base_predictions_train.head(5)

dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
nb_train=nb_train.reshape(-1, 1)
nn_train=nn_train.reshape(-1, 1)
mlp_train=mlp_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
nn_test=nn_test.reshape(-1, 1)
nb_test=nb_test.reshape(-1, 1)
mlp_test=mlp_test.reshape(-1, 1)

x_train = np.concatenate(( dt_train, et_train, rf_train, nn_train,nb_train,mlp_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, nn_test,nb_test,mlp_test), axis=1)
stk = xgb.XGBClassifier().fit(x_train, y_train)
y_predict=stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)

xg = xgb.XGBClassifier(learning_rate= 0.6530839813071385, n_estimators = 65, max_depth = 50)
xg.fit(x_train,y_train)
xg_score=xg.score(x_test,y_test)
y_predict=xg.predict(x_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


