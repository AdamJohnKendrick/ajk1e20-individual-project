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
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


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

# X_train = X_train.values
# X_test = X_test.values
# y_test = y_test.values

input_dim = X_train.shape[1]
encoding_dim = 50

#input layer
input_layer = Input(shape=(input_dim, ))
input_shape = X_train.shape[1:]

#encoding layer with 50 neurons
inputs = Input(shape=input_shape)
encoder = Dense(128, activation="relu")(inputs)
encoder = Dense(64, activation="relu")(encoder)
latent_space = Dense(32)(encoder)

#decoding and output layer
decoder = Dense(64, activation='relu')(latent_space)
decoder = Dense(128, activation='relu')(decoder)
outputs = Dense(input_shape[0], activation='linear')(decoder)

# Define the special neuron
special = Lambda(lambda x: K.ones_like(x))(latent_space)

# creating model with input, encoding, decoding, output layers
#autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Define the model with the additional constraint
lambda_val = 0.1
model = Model(inputs=[inputs], outputs=[outputs])
model.add_loss(lambda_val * K.sum(K.square(special - latent_space)))

# defining loss function, optimizer, metrics and then compiling model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

print(X_train)

# training the model on training dataset
history = model.fit(X_train, X_train, epochs=100,batch_size=500,validation_data=(X_test, X_test)).history

# predicting target attribute on testing dataset
test_results = model.evaluate(X_test, X_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

# Plot of accuracy vs epoch of train and test dataset
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# representation of model layers
#plot_model(autoencoder, to_file='plots/ae_binary.png', show_shapes=True,)

