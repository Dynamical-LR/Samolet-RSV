import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

PATH_TO_TRAIN = "datasets/raw_data/dataset.csv"
PATH_TO_TEST = "datasets/raw_data/test.csv"

text_cols=['col49', 'col50', 'col51', 'col52', 'col53', 'col54', 'col55', 'col56', 'col204', 'col221', 'col222', 'col223', 'col224', 'col225', 'col226', 'col227', 'col228', 'col237', 'col238', 'col239', 'col240', 'col241', 'col242', 'col243', 'col244', 'col457', 'col458', 'col459', 'col460', 'col461', 'col462', 'col463', 'col464', 'col545', 'col546', 'col547', 'col548', 'col549', 'col550', 'col551', 'col552', 'col553', 'col554', 'col555', 'col556', 'col557', 'col558', 'col559', 'col560', 'col1046', 'col1047', 'col1048', 'col1049', 'col1050', 'col1051', 'col1052']
cat_cols=['год', 'сезон', 'месяц', 'col57', 'col58', 'col59', 'col60', 'col61', 'col62', 'col63', 'col64', 'col85', 'col86', 'col87', 'col88', 'col89', 'col90', 'col91', 'col92', 'col93', 'col94', 'col95', 'col96', 'col97', 'col98', 'col99', 'col100', 'col129', 'col130', 'col131', 'col132', 'col133', 'col134', 'col135', 'col136', 'col137', 'col138', 'col139', 'col140', 'col141', 'col142', 'col143', 'col144', 'col145', 'col146', 'col147', 'col148', 'col149', 'col150', 'col151', 'col152', 'col153', 'col154', 'col155', 'col156', 'col157', 'col158', 'col159', 'col160', 'col161', 'col162', 'col163', 'col164', 'col165', 'col166', 'col167', 'col168', 'col169', 'col170', 'col171', 'col172', 'col173', 'col174', 'col175', 'col176', 'col181', 'col182', 'col183', 'col184', 'col185', 'col186', 'col187', 'col188', 'col189', 'col190', 'col191', 'col192', 'col193', 'col194', 'col195', 'col196', 'col201', 'col202', 'col203', 'col205', 'col206', 'col207', 'col208', 'col209', 'col210', 'col211', 'col212', 'col213', 'col214', 'col215', 'col216', 'col465', 'col466', 'col467', 'col468', 'col469', 'col470', 'col471', 'col472', 'col513', 'col514', 'col515', 'col516', 'col517', 'col518', 'col519', 'col520', 'col521', 'col522', 'col523', 'col524', 'col525', 'col526', 'col527', 'col528', 'col529', 'col530', 'col531', 'col532', 'col533', 'col534', 'col535', 'col536', 'col537', 'col538', 'col539', 'col540', 'col541', 'col542', 'col543', 'col544', 'col561', 'col562', 'col563', 'col564', 'col565', 'col566', 'col567', 'col568', 'col569', 'col570', 'col571', 'col572', 'col573', 'col574', 'col575', 'col576', 'col577', 'col578', 'col579', 'col580', 'col581', 'col582', 'col583', 'col584', 'col585', 'col586', 'col587', 'col588', 'col589', 'col590', 'col591', 'col592', 'col593', 'col594', 'col595', 'col596', 'col597', 'col598', 'col599', 'col600', 'col601', 'col602', 'col603', 'col604', 'col605', 'col606', 'col607', 'col608', 'col609', 'col610', 'col611', 'col612', 'col613', 'col614', 'col615', 'col616', 'col793', 'col794', 'col795', 'col796', 'col797', 'col798', 'col799', 'col800', 'col1029', 'col1030', 'col1031', 'col1032', 'col1033', 'col1034', 'col1035', 'col1036', 'col1037', 'col1038', 'col1039', 'col1040', 'col1041', 'col1042', 'col1043', 'col1044', 'col1045', 'col1053', 'col1054', 'col1055', 'col1056', 'col1057', 'col1058', 'col1059', 'col1060', 'col1061', 'col1062', 'col1063', 'col1064', 'col1065', 'col1066', 'col1067', 'col1068', 'col1070', 'col1071', 'col1072', 'col1073', 'col1074', 'col1075', 'col1076', 'col1170', 'col1171', 'col1172', 'col1174', 'col1175', 'col1176', 'col1177', 'col1178', 'col1179', 'col1180', 'col1273', 'col1274', 'col1275', 'col1276', 'col1277', 'col1278', 'col1279', 'col1280', 'col1445', 'col1446', 'col1447', 'col1448', 'col1449', 'col1450', 'col1451', 'col1452', 'col1647', 'col1648', 'col1649', 'col1650', 'col1651', 'col1652', 'col1653', 'col1654', 'col2191', 'col2192', 'col2193', 'col2194', 'col2195', 'col2196', 'col2197', 'col2198']
# новые категории в тесте
cat_cols += ['col137', 'col141', 'col169', 'col819', 'col820', 'col823', 'col824',
       'col1069', 'col1170', 'col1171', 'col1172', 'col1173', 'col1273',
       'col1274', 'col1277', 'col1278', 'col1647', 'col1648', 'col1649',
       'col1650', 'col1653']


train = pd.read_csv(PATH_TO_TRAIN, low_memory=False)
test = pd.read_csv(PATH_TO_TEST, sep=';', low_memory=False)
test['target'] = 0


def handle_df(df: pd.DataFrame):
    '''Add time features, encode categories and get embedings of the text
    '''
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['год'] = df['report_date'].dt.year
    df['сезон'] = (df['report_date'].dt.quarter % 4) + 1
    df['месяц'] = df['report_date'].dt.month
    df.fillna(-1, inplace=True)
    df.drop(columns=['report_date'], inplace=True)

    for col in text_cols:
        df[col] = df[col].astype(str)
    for col in cat_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['target','client_id','col1454'])
    y = df['target']

    # Convert categorical columns to 'category' type
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # Transform text columns using TF-IDF
    tfidf = TfidfVectorizer(max_features=15)
    for col in text_cols:
        try:
            tfidf_matrix = tfidf.fit_transform(X[col].fillna(''))
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_{i}" for i in range(tfidf_matrix.shape[1])])
            X = pd.concat([X, tfidf_df], axis=1)
        except:
            print('skipped', col)
        X = X.drop(col, axis=1)

    for col in ['col819', 'col820', 'col823', 'col824']:
        X[col] = X[col].replace('None', -1).astype(int)

    return X, y


X, y = handle_df(pd.concat((train, test), axis=0, ignore_index=True))
spl = train.shape[0]

X_tr, y_tr = X.iloc[:spl], y.iloc[:spl]
X_ts, y_ts = X.iloc[spl:], y.iloc[spl:]

# CatBoost
model_cb = CatBoostClassifier(verbose=0, cat_features=cat_cols, depth=3)
model_cb.fit(X_tr, y_tr)
# pred_ts = model_cb.predict_proba(X_ts)[:,1]

# EASE
mms = MinMaxScaler((-2,2))
X_tr_ = mms.fit_transform(np.column_stack((y_tr, X_tr)))
X_ts_ = mms.transform(np.column_stack((y_ts, X_ts)))
X_ = np.vstack((X_tr_, X_ts_))
gram = X_.T @ X_
diag_indices = np.diag_indices(gram.shape[0])
presicion = np.linalg.inv(gram + 3e3*np.eye(gram.shape[0]))
B = presicion / (-np.diag(presicion))
B[diag_indices] = 0
pred_tr, pred_ts = X_tr_@B[:,0], X_ts_@B[:,0]

# head
sX_tr = np.column_stack((model_cb.predict_proba(X_tr)[:,1], pred_tr))
sX_ts = np.column_stack((model_cb.predict_proba(X_ts)[:,1], pred_ts))
head = LogisticRegression().fit(sX_tr, y_tr)
pred_ts = head.predict_proba(sX_tr)[:,1]

# sumbission
test['target'] = pred_ts
submission = test.loc[:, ['id', 'target']]
submission.to_csv('submission.csv',index=False,sep=';')