import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn import preprocessing

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
    #calling change_label() function
    change_label(data)

    # distribution of attack classes
    print(data['labels'].value_counts())

    # selecting numeric attributes columns from data
    numeric_col = data.select_dtypes(include='number').columns

    # using standard scaler for normalizing
    std_scaler = StandardScaler()

    # calling the normalization() function
    data = normalization(data.copy(),numeric_col, std_scaler)

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

    print(data['labels'].value_counts())

    # changing attack labels into two categories 'normal' and 'abnormal'
    bin_label = pd.DataFrame(data['labels'].map(lambda x:'dos' if x == 'Dos' else 'not_dos'))
    print(bin_label.value_counts())

    # creating a dataframe with binary labels (normal,abnormal)
    bin_data = data.copy()
    bin_data['labels'] = bin_label

    # label encoding (0,1) binary labels (abnormal,normal)
    le1 = preprocessing.LabelEncoder()
    print(le1.fit_transform)
    enc_label = bin_label.apply(le1.fit_transform)
    bin_data['intrusion'] = enc_label

    np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

    # one-hot-encoding attack label
    bin_data = pd.get_dummies(bin_data,columns=['labels'],prefix="",prefix_sep="")
    bin_data['labels'] = bin_label

    print(bin_data['labels'].value_counts())

    # # pie chart distribution of normal and abnormal labels
    # plt.figure(figsize=(8,8))
    # plt.pie(bin_data['labels'].value_counts(),labels=bin_data['labels'].unique(),autopct='%0.2f%%')
    # plt.title("Pie chart distribution of normal and abnormal labels")
    # plt.legend()
    # plt.show()

    # converting type of service column to 'category'
    bin_data['service'] = bin_data['service'].astype('category')
    bin_data['service'] = bin_data['service'].cat.codes

    # converting type of flag column to 'category'
    bin_data['flag'] = bin_data['flag'].astype('category')
    bin_data['flag'] = bin_data['flag'].cat.codes

    # converting type of protocol_type column to 'category'
    bin_data['protocol_type'] = bin_data['protocol_type'].astype('category')
    bin_data['protocol_type'] = bin_data['protocol_type'].cat.codes

    # labels = bin_data['labels'].copy()
    # bin_data = bin_data.drop()

    return bin_data


# importing dataset
X_train = pd.read_csv('kdd_train.csv')
X_test = pd.read_csv('kdd_test.csv')

X_train = format_ds(X_train)
X_test = format_ds(X_test)

# AE Model

# splitting the dataset 75% for training and 25% testing
#X_train, X_test = train_test_split(bin_data, test_size=0.25, random_state=42)

# dataset excluding target attribute (encoded, one-hot-encoded,original)
X_train = X_train.drop(['intrusion','dos','not_dos','labels'],axis=1)

y_test = X_test['intrusion'] # target attribute

# dataset excluding target attribute (encoded, one-hot-encoded,original)
X_test = X_test.drop(['intrusion','dos','not_dos','labels'],axis=1)


# Define the stacked autoencoder model
input_layer = Input(shape=(X_train.shape[1],))
encoder1 = Dense(32, activation='relu')(input_layer)
encoder2 = Dense(16, activation='relu')(encoder1)
decoder1 = Dense(32, activation='relu')(encoder2)
output_layer = Dense(X_train.shape[1], activation='linear')(decoder1)
stacked_autoencoder = Model(inputs=input_layer,
                            outputs=output_layer)

# Compile the stacked autoencoder model
stacked_autoencoder.compile(loss='mse', optimizer=Adam(lr=0.001))

# Train the stacked autoencoder model
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
stacked_autoencoder.fit(X_train,
                        X_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop])

# Define the multilayer perceptron model
mlp = Sequential()
mlp.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))

# Compile the multilayer perceptron model
mlp.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the multilayer perceptron model
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mlp.fit(X_train, np.ones(X_train.shape[0]), epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Extract the hidden layer representation from the stacked autoencoder
encoder = Model(inputs=input_layer, outputs=encoder2)
hidden_layer_train = encoder.predict(X_train)
hidden_layer_test = encoder.predict(X_test.iloc[:, :41])

# Concatenate the hidden layer representation with the original input
combined_train_df = np.concatenate((X_train, hidden_layer_train), axis=1)
combined_test_df = np.concatenate((X_test.iloc[:, :41], hidden_layer_test), axis=1)

# Define the stacked RBM model
rbm = Sequential()
rbm.add(Dense(64, input_dim=combined_train_df.shape[1], activation='relu'))
rbm.add(Dense(32, activation='relu'))
rbm.add(Dense(16, activation='relu'))
rbm.add(Dense(1, activation='sigmoid'))

# Compile the stacked RBM model
rbm