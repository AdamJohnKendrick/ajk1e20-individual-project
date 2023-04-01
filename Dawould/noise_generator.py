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

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

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
   # data = normalization(data.copy(),numeric_col, std_scaler)

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
    bin_label = pd.DataFrame(data['labels'].map(lambda x:'dos' if x == 'Dos' else 'not_dos'))

    # creating a dataframe with binary labels (normal,abnormal)
    bin_data = data.copy()
    bin_data['labels'] = bin_label

    # label encoding (0,1) binary labels (abnormal,normal)
    le1 = preprocessing.LabelEncoder()
    enc_label = bin_label.apply(le1.fit_transform)
    bin_data['intrusion'] = enc_label

    le1.classes_

    np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

    # one-hot-encoding attack label
    bin_data = pd.get_dummies(bin_data,columns=['labels'],prefix="",prefix_sep="")
    bin_data['labels'] = bin_label


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

    #bin_data = bin_data.drop('label')

    bin_data = bin_data.drop(columns=['labels'])

    return bin_data


# importing dataset
x_train = pd.read_csv('kdd_train.csv')
x_test = pd.read_csv('kdd_test.csv')

x_train = format_ds(x_train)
x_test = format_ds(x_test)

num_features = 44

# Define the VAE model
latent_dim = 10

# Encoder
encoder_inputs = tf.keras.Input(shape=(x_train.shape[1],))
encoder = layers.Dense(21, activation='relu')(encoder_inputs)
#x = layers.Dense(64, activation='relu')(x)

# Mean and log variance for Gaussian Distribution
z_mean = layers.Dense(latent_dim)(encoder)
z_log_var = layers.Dense(latent_dim)(encoder)

# Generate new data samples that are similar to the original data but exhibit variation and novelty.
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Apply sampling function to z_mean and z_log_var
z = layers.Lambda(sampling)([z_mean, z_log_var])

# Define the AE model
encoder_model = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])

latent_inputs = Input(shape=(latent_dim,))
x = layers.Dense(21, activation="relu")(latent_inputs)
outputs = layers.Dense(44, activation="sigmoid")(x)
decoder = Model(latent_inputs, outputs, name="decoder")

encoder_output = encoder_model(encoder_inputs)[2]
output = decoder(encoder_output)

vae = Model(encoder_inputs, output)

def kl_loss(inp, out):
    reconstruction_loss = encoder_inputs[0] * metrics.binary_crossentropy(inp, out)

    # Kullback-Leibler to check the learned distributions is close to a prior distribution
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    return K.mean(reconstruction_loss + kl_loss)

# Compile the model
vae.compile(optimizer='adam', loss=kl_loss)

# Train the model
vae.fit(x_train, x_train,
        epochs=10,
        batch_size=44)

# # Generate new data
# z_sample = tf.keras.backend.random_normal(shape=(100, latent_dim))
# x_decoded = decoder(z_sample).numpy()

# print(x_decoded)

# Generate data using the trained VAE model
n_samples = 1000
z = np.random.normal(size=(n_samples, latent_dim))
generated_data = decoder.predict(z)

# Save generated data to a CSV file
generated_df = pd.DataFrame(generated_data)
generated_df.to_csv('generated_data.csv', index=False)

    # # Kullback-Leibler to check the learned distributions is close to a prior distribution
    # kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #
    # print(tf.exp(z_log_var))
    # print(tf.square(z_mean))
    #
    # kl_loss = tf.reduce_mean(kl_loss, axis=-1)
    # kl_loss *= -0.5

# Total loss of Variational AutoEncoder + Kullback-Leibler
# vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

# # Generate new data
# latent_points = np.random.normal(size=(1000, latent_dim))
# generated_data = decoder.predict(latent_points)
# generated_data = generated_data * train_std.values + train_mean.values

# How well reconstruction is doing from binary loss entropy
# reconstruction_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, outputs)
# reconstruction_loss *= num_features