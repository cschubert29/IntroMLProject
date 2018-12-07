import pandas as pd
import numpy as np
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
filename_sen1 = path+'\\'+'Sen_bill_status_contemporary.json'
filename_sen2 = path+'\\'+'Sen_bill_status_modern.json'
f_sm = path+'\\'+'Majority Party Chairmanships - Senate.csv'
f_lc = path+'\\'+'legislators-current.csv'
f_lh = path+'\\'+'legislators-historical.csv'

df_sm = pd.read_csv(f_sm)
df_sm = df_sm[['Congress', 'MajParty']]
df_sm = df_sm.drop_duplicates()

df_lc = pd.read_csv(f_lc)
df_lh = pd.read_csv(f_lh)
df_lh['full_name'] = df_lh['last_name'].fillna('')+', '+df_lh['first_name'].fillna('')
df_lh = df_lh[df_lh['type'] == 'sen']
df_lh = df_lh[['full_name', 'party']]
party = ["Republican", "Democrat"]
df_lh = df_lh[df_lh['party'].isin(party)]

df1 = pd.read_json(filename_sen1)
df2 = pd.read_json(filename_sen2)
df = df1.append(df2, ignore_index=True)
df = df.fillna(np.nan)

df['cosponsor_flg'] = np.where(df['cosponsors'].apply(len)==0, 0, 1)
df['amendments_flg'] = np.where(df['amendments'].apply(len)==0, 0, 1)
df['popular_flg'] = np.where(df['popular_title'].isnull(), 0, 1)
df['related_bills_flg'] = np.where(df['related_bills'].apply(len)==0, 0, 1)
df['sponsor_nm'] = [d.get('name') for d in df.sponsor]
df['full_name'] = df['sponsor_nm'].str.split(' ',expand = True)[0]+' '+df['sponsor_nm'].str.split(' ',expand = True)[1] 
df['Congress'] = pd.to_numeric(df['congress'], errors='coerce')

df_sen1 = pd.merge(df, df_lh, how='inner', on=['full_name'])
df_sen2 = pd.merge(df_sen1, df_sm, how='inner', on=['Congress'])
df_sen3 = df_sen2[df_sen2['party'].isin(party)]
df_sen3['party_flg'] = np.where(df_sen3['party'] == "Republican", 0, 1)
df_sen3['majority_flg'] = np.where(df_sen3['MajParty'] == "R", 0, 1)
statuses = ["PASSED:BILL", "ENACTED:SIGNED"]
df_sen3['Class'] = np.where(df_sen3['status'].isin(statuses), 1, 0)

df_sen = df_sen3[['bill_id', 'cosponsor_flg','amendments_flg', 'popular_flg', 'related_bills_flg', 'party_flg', 'majority_flg', 'Class']]
df_sen = df_sen.reset_index(drop = True)

features = list(df_sen.columns[1:7])
traindatasize = int(len(df_sen)*.8)
testdatasize = len(df_sen) - traindatasize

traindata = df_sen[0:traindatasize]
testdata = df_sen[traindatasize:len(df_sen)+1]

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

#dd = df_sen3['status'].unique().tolist()
#df_sen3.groupby(df['status']).count()
#df_hr.groupby(df_hr['status']).count()
#result = pd.merge(df, df_hr, how='inner', on=['bill_id'])
#df['sponsor_nm'] = [d.get('name') for d in df.sponsor]