import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data=pd.read_csv(r"C:\Users\njain12\Desktop\FullyHack2023_Project\Stress_Reduction\static\SaYoPillow.csv")

data.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']

data=data.drop(['snoring_rate','body_temperature','eye_movement'],axis=1)

X_train, X_test, y_train, y_test=train_test_split(data.iloc[:, :5], data['stress_level'], \
                                                  test_size=0.2, random_state=8)

num_classes=5
y_train=to_categorical(y_train, num_classes)
y_test=to_categorical(y_test, num_classes)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

model=Sequential()
model.add(Dense(125, activation="relu"))
model.add(Dense(125, activation="relu"))
model.add(Dense(5, "softmax"))

epochs=10
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
stats=model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

_, accuracy=model.evaluate(X_test, y_test)

np.argmax(model.predict([[24.8,17.75,93.6,6.5,90.4]]))

model.save(r'C:\Users\njain12\Desktop\FullyHack2023_Project\Stress_Reduction\static\my_model.h5')