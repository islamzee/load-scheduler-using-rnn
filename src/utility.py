from importlibs import *
from enum_switch import Switch
import lstmModels


def getPath(path):
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    return Path(str(projectPath) + '/' + path)


# convert an array of values into a dataset matrix
def create_dataset(dataArray, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataArray) - look_back - 1):
        a = dataArray[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataArray[i + look_back])
    return np.array(dataX), np.array(dataY)


def getDatasetForDataCenterInRegion(regionalISOName):
    df_raw = pd.read_csv(
        os.path.join(getPath('dataset/' + regionalISOName), DATASET_NAMES[regionalISOName]),
        header=None
    )

    return df_raw.T


def lstmModelFileExists(regional_ISO_name):
    filePathStr = 'output/' + regional_ISO_name + '/' + FILE_NAME_LSTM_MODEL
    outputPath = getPath(filePathStr)
    return os.path.exists(outputPath)


def deleteLstmModelFileFor(regional_ISO_name):
    outputDirPath = getPath('output/' + regional_ISO_name)
    for file in os.scandir(outputDirPath):
        os.remove(file.path)


def getLstmModel(modelType):
    switcher = {
        LSTM_MODELS.ONE_HIDDEN_LAYER: getOneHiddenLayer
    }

    return switcher[modelType]