import importlibs
from utility import *
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


def run_LSTM(regional_ISO_name):
    input = getDatasetForDataCenterInRegion(regional_ISO_name)
    print('Dataset total size: ', input.shape)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = input.astype('float32')

    # split into train and test sets
    train_size = int(len(dataset) * TRAIN_FRAC)
    train, test = dataset.iloc[0:train_size, :], dataset.iloc[train_size:len(dataset), :]

    train = pd.DataFrame(scaler.fit_transform(train.values.reshape(-1, 1)), index=train.index)

    trainX, trainY = create_dataset(train.values, LOOK_BACK)  # trainSize * lookbackSize, trainSize * 1
    # reshape into [samples, timesteps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    trainY = np.reshape(trainY, (trainY.shape[0], 1))

    # create and fit the LSTM network
    directoryPath = '../output/' + regional_ISO_name + '/'
    model = Sequential()
    history = None
    if not lstmModelFileExists(regional_ISO_name):
        model.add(LSTM(288, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], 1)))
        model.add(LSTM(288, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse', 'accuracy'])

        # Early stopping
        es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=5)
        #  Checkpointing
        mc = ModelCheckpoint(directoryPath + FILENAME_CHECKPOINT,
                             monitor='accuracy',
                             mode='min', verbose=1,
                             save_best_only=True)

        history = model.fit(trainX, trainY,
                            epochs=EPOCH_SIZE,
                            batch_size=NUM_BATCHES,
                            validation_split=0.3,
                            # validation_data=(testX, testY),
                            verbose=2,
                            callbacks=[es, mc])
        print(model.summary())
        path = getPath(os.path.join('output', regional_ISO_name, FILE_NAME_LSTM_MODEL))
        model.save(path)
        pickle.dump(history, open(directoryPath + FILENAME_HISTORY, "wb"))
    else:
        # returns a compiled model identical to the previous one
        model = load_model(directoryPath + FILE_NAME_LSTM_MODEL)
        history = pickle.load(open(directoryPath + FILENAME_HISTORY, "rb"))
    # ------------------------------------------------------------

    # normalize label_dataset
    test = pd.DataFrame(scaler.fit_transform(test.values.reshape(-1, 1)), index=test.index)
    # reshape into X=t and Y=t+1
    testX, testY = create_dataset(test.values, LOOK_BACK)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # make predictions
    predictions = model.predict(testX)

    predictions = scaler.inverse_transform(predictions)
    testY = scaler.inverse_transform(testY)
    predY = predictions[:, 0]

    # accuracy results of predictions VS testY
    mse = mean_squared_error(testY, predY)
    rmse = math.sqrt(mse)
    print('--- RMSE: ', rmse)

    mae = mean_absolute_error(testY, predY)
    print('--- MAE: ', mae)

    # Initialize plots
    figure, axis = plt.subplots(1,2)

    # history plots
    if history is not None:
        print(history.history.keys())
        # summarize history for accuracy
        axis[0,0].plot(history.history['accuracy'])
        # axis[0,0].plot(history.history['val_accuracy'])
        axis[0,0].title('model accuracy')
        axis[0,0].ylabel('accuracy')
        axis[0,0].xlabel('epoch')
        axis[0,0].legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        axis[0,1].plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        axis[0,1].title('model loss')
        axis[0,1].ylabel('loss')
        axis[0,1].xlabel('epoch')
        axis[0,1].legend(['train', 'test'], loc='upper left')
        axis[0,1].show()

    # ------
    # deleteLstmModelFileFor(regional_ISO_name)

    # plot baseline and predictions
    plt.plot(range(0, len(testY)), testY, label='Test data', color='green')
    plt.plot(range(0, len(testY)), predY[0:len(testY)], label='Predicted', color='red')
    plt.legend()
    plt.show()


run_LSTM('CAISO')
