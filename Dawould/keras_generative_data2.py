import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# importing library for plotting
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras import layers
import keras.backend as K
from tensorflow.keras import metrics

# dataset doesn't have column names, so we have to provide it
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

# changing attack labels to their respective attack class
def change_label(df):
  df['labels'].replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
  df['labels'].replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
       'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
  df['labels'].replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
  df['labels'].replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)


def normalization(df, col, std_scaler):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df

def format_ds(data):

    # distribution of attack classes
    # print(data['labels'].value_counts())

    # selecting numeric attributes columns from data
    numeric_col = data.select_dtypes(include='number').columns

    # using standard scaler for normalizing
    std_scaler = StandardScaler()

    #print('FORMAT DATASET')
    #print(numeric_col)

    # calling the normalization() function
    #data = normalization(data.copy(),numeric_col, std_scaler)

    # selecting categorical data attributes
    cat_col = ['protocol_type','service','flag']

    categorical = data[cat_col]
    categorical.head()

    # one-hot-encoding categorical attributes using pandas.get_dummies() function
    categorical = pd.get_dummies(categorical,columns=cat_col)
    categorical.head()

    NOT_DOS = ['normal', 'ipsweep', 'satan', 'portsweep', 'nmap', 'saint', 'mscan', 'buffer_overflow',
                'rootkit', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm',
                'guess_passwd', 'warezmaster', 'imap', 'multihop', 'phf', 'ftp_write', 'spy',
                'warezclient', 'named', 'sendmail', 'xlock', 'xsnoop', 'worm', 'snmpgetattack', 'snmpguess']

    DOS = ['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', 'apache2', 'mailbomb', 'processtable', 'udpstorm']

    # # Converting labels column to not_DoS and DoS attacks
    # for label in NOT_DOS:
    #     data['labels'] = data['labels'].replace(label, 'not_dos')
    #
    # for label in DOS:
    #     data['labels'] = data['labels'].replace(label, 'dos')

    # changing attack labels into two categories 'normal' and 'abnormal'
    data['labels'] = data['labels'].map(lambda x:'dos' if x in DOS else 'not_dos')


    # np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

    # # one-hot-encoding attack label
    # bin_data = pd.get_dummies(bin_data,columns=['labels'],prefix="",prefix_sep="")
    # bin_data['labels'] = bin_label


    # # pie chart distribution of normal and abnormal labels
    # plt.figure(figsize=(8,8))
    # plt.pie(bin_data['labels'].value_counts(),labels=bin_data['labels'].unique(),autopct='%0.2f%%')
    # plt.title("Pie chart distribution of normal and abnormal labels")
    # plt.legend()
    # plt.show()

    # converting type of service column to
    # 'category'
    data['service'] = data['service'].astype('category')
    data['service'] = data['service'].cat.codes

    # converting type of flag column to 'category'
    data['flag'] = data['flag'].astype('category')
    data['flag'] = data['flag'].cat.codes

    # converting type of protocol_type column to 'category'
    data['protocol_type'] = data['protocol_type'].astype('category')
    data['protocol_type'] = data['protocol_type'].cat.codes

    data['labels'] = data['labels'].astype('category')
    data['labels'] = data['labels'].cat.codes
    # labels = data['labels']
    # data = data.drop(['labels'], axis=1)

    print(data.head())

    return data


# importing dataset
X_train = pd.read_csv('kdd_train.csv')
X_test = pd.read_csv('kdd_test.csv')

X_train = format_ds(X_train)
X_test = format_ds(X_test)

print(X_train)
labels_store = X_train['labels']
X_train = X_train.drop(columns='labels')

print(labels_store)

# Needs changing
X_train = tf.keras.utils.normalize(X_train)

X_train['labels'] = labels_store
print(X_train.head())

