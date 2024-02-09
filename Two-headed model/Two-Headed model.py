import tensorflow as tf
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import random

# --- Disable eager execution
tf.compat.v1.disable_eager_execution()


n_train = 10
n_test = 10
x_min = -np.pi
x_max = np.pi
x_train=np.linspace(x_min, x_max, n_train)
n1_train=[]
n2_train=[]
for i in x_train:
    if i >=0:
        n1_train.append(i)
    else:
        n2_train.append(i)

y_train11=np.sin(0.5*np.asarray(n1_train))+np.random.normal(0,0.5,(np.asarray(n1_train).shape))
y_train22=np.sin(0.5*np.asarray(n2_train))+np.random.normal(0,0.2,(np.asarray(n2_train).shape))
y_train=list(y_train11)+list(y_train22)
plt.plot(y_train)
plt.show()
print(y_train)

x_test=np.linspace(-(np.pi), (np.pi), n_test)
n1_test=[]
n2_test=[]
for i in x_test:
    if i >=0:
        n1_test.append(i)
    else:
        n2_test.append(i)

test_y11=np.sin(0.5*np.asarray(n1_test))+np.random.normal(0,0.5,(np.asarray(n1_test).shape))
test_y22=np.sin(0.5*np.asarray(n2_test))+np.random.normal(0,0.2,(np.asarray(n2_test).shape))
y_test=list(test_y11)+list(test_y22)

# --- Create model
model = keras.Sequential()

model.add(keras.layers.Dense(64, activation="relu", input_dim=1))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(2, activation="relu"))

def grad(input_tensor, output_tensor):
    return keras.layers.Lambda( lambda z: keras.backend.gradients( z[ 0 ], z[ 1 ] ), output_shape = [1] )( [ output_tensor, input_tensor ] )

def custom_loss_wrapper(input_tensor, output_tensor):

    def custom_loss(y_true, y_pred):
        dim = int(y_pred.shape[1])
        n_dims = 1 / (2 * dim)
        n_outs = int(y_pred.shape[1] / 2)
        mu = y_pred[:, 0:n_outs]
        sigma = K.exp(y_pred[:, n_outs:])
        sigma_sq = pow(sigma, 2)
        logsigma = y_pred[:, n_outs:]
        log_likelihood = n_dims * (K.sum(logsigma + (K.square((y_true - mu) / sigma_sq))))
        return log_likelihood

    return custom_loss

# --- Configure learning process
model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=custom_loss_wrapper(model.input, model.output),
        metrics=['MeanSquaredError'])
model.summary()
# --- Train from dataset
ypp=model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
# --- Evaluate model
yqq=model.evaluate(x_test, y_test)
predictMean = model.predict(y_test)[:,0:1]
print(np.shape(predictMean))
predictStdDev = np.sqrt(model.predict(y_test)[:,1:2])
pred_standdev = pow(predictStdDev, 2)
pred_standdev_avg=np.average(pred_standdev)
print(pred_standdev_avg)
pred_mean_avg=np.average(predictMean)
print(pred_mean_avg)
plt.plot(x_train,y_train,label='sine wave with noise', color='b')
plt.scatter(x_train,predictMean,label='mean', color='r')
plt.scatter(x_train,pred_standdev,label='variance')
plt.xlabel("y")
plt.ylabel("x")
plt.legend()

plt.show()

