from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MeanShift
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Not used but show taxonomy of attacks which aren't DoS
U2R = ['buffer_overflow','rootkit','loadmodule', 'perl' 'httptunnel', 'ps', 'sqlattack', 'xterm']
R2L = ['guess_passwd','warezmaster', 'imap', 'multihop', 'phf', 'ftp_write', 'spy', 'warezclient','named',
       'sendmail', 'xlock', 'xsnoop', 'worm', 'snmpgetattack', 'snmpguess']
PROBE = ['ipsweep', 'satan', 'portsweep', 'nmap',  'saint', 'mscan']


NOT_DOS = ['normal', 'ipsweep', 'satan', 'portsweep', 'nmap', 'saint', 'mscan', 'buffer_overflow',
            'rootkit', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm',
            'guess_passwd', 'warezmaster', 'imap', 'multihop', 'phf', 'ftp_write', 'spy',
            'warezclient', 'named', 'sendmail', 'xlock', 'xsnoop', 'worm', 'snmpgetattack', 'snmpguess']
NOT_DOS_ENCODING = 1

DOS = ['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
DOS_ENCODING = 0

def format_dataset(dataset):
    # converting type of service column to 'category'
    dataset['service'] = dataset['service'].astype('category')
    dataset['service'] = dataset['service'].cat.codes

    # converting type of flag column to 'category'
    dataset['flag'] = dataset['flag'].astype('category')
    dataset['flag'] = dataset['flag'].cat.codes

    # converting type of protocol_type column to 'category'
    dataset['protocol_type'] = dataset['protocol_type'].astype('category')
    dataset['protocol_type'] = dataset['protocol_type'].cat.codes

    # Converting labels column to not_DoS and DoS attacks
    for label in NOT_DOS:
        dataset['labels'] = dataset['labels'].replace(label, 'not_dos')

    for label in DOS:
        dataset['labels'] = dataset['labels'].replace(label, 'dos')

    dataset['labels'] = dataset['labels'].apply(lambda x: 1 if x == 'dos' else 0)

    x = dataset.drop('labels', axis=1)
    y = dataset['labels']

    return x, y

print("Reading CSVs...\n")
# Read csv files
kDD_train = pd.read_csv(
    "kdd_train.csv")
kDD_test = pd.read_csv(
    "kdd_test.csv")

print("Setting up encoder...\n")

x_train, y_train,  = format_dataset(kDD_train)
x_test, y_test = format_dataset(kDD_test)

input_dim = x_train.shape[1]

# Encoder - 2 hidden layers
input_data = keras.Input(shape=(input_dim,))
encoded = layers.Dense(21, activation='relu')(input_data)
encoded = layers.Dense(30, activation='relu')(encoded)

print("Setting up decoder...\n")
# Decoder - 2 hidden layers
decoded = layers.Dense(10, activation='relu')(encoded)
decoded = layers.Dense(30, activation='relu')(decoded)
decoded = layers.Dense(41, activation='relu')(decoded)

# Turn layers into autoencoder features can be used on
autoencoder = keras.Model(input_data, decoded)
encoder = keras.Model(input_data, encoded)

encoded_input = keras.Input(shape=(30,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# Config the model with loss and optimizer
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# String summary of network
autoencoder.summary()

epochs = 100

# Provides fit statistics calculated across all models
history = autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, y_test)
                )

encoder_train = encoder.predict(x_train)
encoder_test = encoder.predict(x_test)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(encoder_train)

predicted_labels = kmeans.predict(encoder_test)

accuracy = accuracy_score(predicted_labels,y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))
