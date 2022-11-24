import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow import keras
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import time
def fun(x1,y1):
 i= 0
 n=np.size(x1)
 while i<n:
   z=x1[i]
   y1[i]=math.cos(z**0.5+2)+z
   i=i+1   
 return y1
start_time = time.time()
h=0.005
x1=np.arange(0.5,1+h,h)
y1=np.arange(0.5,1+h,h)
y1=fun(x1,y1)
h=0.0001
x2=np.arange(0,2+h,h)
y2=np.arange(0,2+h,h)
y2=fun(x2,y2)   

model = keras.Sequential(
[
        Dense(64, activation="relu", input_shape=(1,), name="hidden_dense_1"),
        Dense(32, activation="tanh", name="hidden_dense_2"),
        Dense(16, activation="relu", name="hidden_dense_3"),
        Dense(8, activation="tanh", name="hidden_dense_4"),
        Dense(1, activation='linear', name="output"),
]
)

model.compile(loss='mse', optimizer='RMSProp', metrics=['mse'])
callbacks = [ keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001, patience=7, verbose=1)]
history=model.fit(x1,y1,epochs=1000,verbose=1,callbacks=callbacks)
plt.figure(1)
plt.plot(x1,y1) 
print(model.summary())
plt.plot(x1,y1)
plt.plot(x1,model.predict(x1))
plt.grid(True)

plt.figure(2)
plt.plot(x2,y2) 

print(model.summary())
plt.plot(x2,model.predict(x2))
plt.grid(True)
plt.figure(3)
plt.plot(history.history['loss'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.grid(True)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()