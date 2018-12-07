import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


path = r'C:\Users\Vinay\Desktop\Priyanka\ML'
filename_hr1 = path+'\\'+'HR_bill_status_contemporary.json'
filename_hr2 = path+'\\'+'HR_bill_status_modern.json'
f_hm = path+'\\'+'Majority Party Chairmanships - House.csv'
f_lc = path+'\\'+'legislators-current.csv'
f_lh = path+'\\'+'legislators-historical.csv'

df_hm = pd.read_csv(f_hm)
df_hm = df_hm[['Congress', 'MajParty']]
df_hm = df_hm.drop_duplicates()

df_lc = pd.read_csv(f_lc)
df_lh = pd.read_csv(f_lh)
df_lh['full_name'] = df_lh['last_name'].fillna('')+', '+df_lh['first_name'].fillna('')
df_lh = df_lh[df_lh['type'] == 'rep']
df_lh = df_lh[['full_name', 'party']]
party = ["Republican", "Democrat"]
df_lh = df_lh[df_lh['party'].isin(party)]

df_hr1 = pd.read_json(filename_hr1)
df_hr2 = pd.read_json(filename_hr2)
df_hr = df_hr1.append(df_hr2, ignore_index=True)
df_hr = df_hr.fillna(np.nan)

df_hr['cosponsor_flg'] = np.where(df_hr['cosponsors'].apply(len)==0, 0, 1)
df_hr['amendments_flg'] = np.where(df_hr['amendments'].apply(len)==0, 0, 1)
df_hr['popular_flg'] = np.where(df_hr['popular_title'].isnull(), 0, 1)
df_hr['related_bills_flg'] = np.where(df_hr['related_bills'].apply(len)==0, 0, 1)
df_hr['sponsor_nm'] = [d.get('name') for d in df_hr.sponsor]
df_hr['full_name'] = df_hr['sponsor_nm'].str.split(' ',expand = True)[0]+' '+df_hr['sponsor_nm'].str.split(' ',expand = True)[1] 
df_hr['Congress'] = pd.to_numeric(df_hr['congress'], errors='coerce')

df_hr1 = pd.merge(df_hr, df_lh, how='inner', on=['full_name'])
df_hr2 = pd.merge(df_hr1, df_hm, how='inner', on=['Congress'])

df_hr2 = df_hr2[df_hr2['party'].isin(party)]
df_hr2['party_flg'] = np.where(df_hr2['party'] == "Republican", 0, 1)
df_hr2['majority_flg'] = np.where(df_hr2['MajParty'] == "R", 0, 1)
statuses = ["PASSED:BILL", "ENACTED:SIGNED"]
df_hr2['Class'] = np.where(df_hr2['status'].isin(statuses), 1, 0)

df_hr = df_hr2[['bill_id', 'cosponsor_flg','amendments_flg', 'popular_flg', 'related_bills_flg', 'party_flg', 'majority_flg', 'Class']]
df_hr = df_hr.reset_index(drop = True)

features = list(df_hr.columns[1:7])
traindatasize = int(len(df_hr)*.8)
testdatasize = len(df_hr) - traindatasize

traindata = df_hr[0:traindatasize]
testdata = df_hr[traindatasize:len(df_hr)+1]

df2 = traindata
y_train = df2["Class"]
X_train = df2[features]

df3 = testdata
y_test = df3["Class"]
X_test = df3[features]

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_predicted_rf = rf.predict(X_test)
y_score_rf = rf.score(X_test,y_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_predicted = dt.predict(X_test)
y_score = dt.score(X_test,y_test)

d_svm = svm.SVC()
d_svm.fit(X_train, y_train)
y_predicted_svm = d_svm.predict(X_test)
y_score_svm = d_svm.score(X_test,y_test)
