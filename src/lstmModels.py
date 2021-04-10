import importlibs
import constants


def getOneHiddenLayer():
    model = Sequential()
    model.add(LSTM(1))
    model.compile(optimizer='adam', loss='mse')
    return model
