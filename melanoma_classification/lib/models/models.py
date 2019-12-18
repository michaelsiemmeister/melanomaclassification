from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess, )
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import (Model, layers)


def trainDataGen(preprocessing_fn):
    return ImageDataGenerator(preprocessing_function=preprocessing_fn,
                              rotation_range=359,
                              horizontal_flip=True)


def testDataGen(preprocessing_fn):
    return ImageDataGenerator(preprocessing_function=preprocessing_fn)


def resnet50_trainDataGen_flow_df(df,
                                  directory_path,
                                  batch_size=20,
                                  class_mode='binary'):
    return trainDataGen(resnet50_preprocess).flow_from_dataframe(
        df,
        directory=directory_path,
        x_col='id',
        y_col='category',
        target_size=(224, 224),
        batch_size=batch_size,
        seed=0,
        class_mode='binary')


def resnet50_testDataGen_flow_df(df,
                                 directory_path,
                                 batch_size=20,
                                 class_mode='binary'):
    return testDataGen(resnet50_preprocess).flow_from_dataframe(
        df,
        directory=directory_path,
        x_col='id',
        y_col='category',
        target_size=(224, 224),
        batch_size=batch_size,
        seed=0,
        class_mode='binary',
        shuffle=False)


def resnet50_pretrained_simple():
    '''
    returns a pretrained resnet50 model
    - input shape: (224, 224, 3)
    - global-avg_pooling
    - then 1 fully connected layer with  1 neuron
     = classification layer
    - only the classification layer is trainable
    '''

    resnet50_model_pretrained = ResNet50(include_top=False,
                                         input_shape=(224, 224, 3),
                                         pooling='avg')
    # set the pretrained layers as not trainable
    for layer in resnet50_model_pretrained.layers:
        layer.trainable = False
    pretrained_output = resnet50_model_pretrained.layers[-1].output
    model_with_classifier = layers.Dense(
        1, activation='sigmoid')(pretrained_output)

    # create keras model
    this_model = Model(resnet50_model_pretrained.input, model_with_classifier)
    return this_model


def resnet50_pretrained_twoLayer():
    '''
    returns a pretrained resnet50 model
    - input shape: (224, 224, 3)
    - global-avg_pooling
    - then 1 fully connected layer with  1024 neurons
    - then 1 fully connected layer with 1 neuron
     = classification layer
    - only the last 2 layers are trainable
    '''

    resnet50_model_pretrained = ResNet50(include_top=False,
                                         input_shape=(224, 224, 3),
                                         pooling='avg')
    # set the pretrained layers as not trainable
    for layer in resnet50_model_pretrained.layers:
        layer.trainable = False
    pretrained_output = resnet50_model_pretrained.layers[-1].output
    fully_connected = layers.Dense(1024, activation='relu')(pretrained_output)
    model_with_classifier = layers.Dense(1,
                                         activation='sigmoid')(fully_connected)

    # create keras model
    this_model = Model(resnet50_model_pretrained.input, model_with_classifier)
    return this_model