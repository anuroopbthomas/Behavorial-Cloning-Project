from keras.models import Sequential

#Creating the sequential models
model = Sequential()

#first layer - Add a flatten layers
model.add(Flatten(input_shape=(32,32,3)))

#second layer - Add a fully connected layers
model.add(Dense(100))

#third layer - Add a ReLU activation layers
model.add(Activation('relu'))

#fourth layer - Add a fully connected layers
model.add(Dense(60))

#fifth layer - Add a ReLU activation layers
model.add(Activation('relu'))
