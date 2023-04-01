import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn import preprocessing
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf

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

    # changing attack labels into two categories 'normal' and 'abnormal'
    data['labels'] = data['labels'].map(lambda x:'dos' if x in DOS else 'not_dos')

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

Y_train = X_train['labels']
Y_test = X_test['labels']

X_train = X_train.drop(columns='labels')
X_test = X_test.drop(columns='labels')

# Reshape the data to feed into the RNNs
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM-based network architecture
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(1, X_train.shape[2])),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the LSTM-based model
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the LSTM-based model
lstm_model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test))

# Evaluate the performance of the LSTM-based model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, Y_test)
print("LSTM Test Loss:", lstm_loss)
print("LSTM Test Accuracy:", lstm_accuracy)

# Define the Gated Reccurent Unit-based network architecture
gru_model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, input_shape=(1, X_train.shape[2])),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the GRU-based model
gru_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the GRU-based model
gru_model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test))

# Evaluate the performance of the GRU-based model
gru_loss, gru_accuracy = gru_model.evaluate(X_test, Y_test)
print("GRU Test Loss:", gru_loss)
print("GRU Test Accuracy:", gru_accuracy)