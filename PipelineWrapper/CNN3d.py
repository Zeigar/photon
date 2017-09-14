from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Sequential
import numpy as np
import keras
import tensorflow as tf
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


class CNN3d(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=0.001, model=None, batch_size=16, nb_epochs=200, target_dimension=2,
                 optimizer='sgd', momentum=0.9, loss='binary_crossentropy', metrics=['mean_absolute_error'], learningRateDecay=0.01,
                 conv1Filters=32, conv1KernelSize=10, conv1Strides=5,
                 gpu_device='/gpu:0'):
        self.learning_rate = learning_rate
        self.model = model
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.target_dimension = target_dimension
        self.momentum = momentum
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.learningRateDecay = learningRateDecay

        self.conv1Filters = conv1Filters
        self.conv1KernelSize = conv1KernelSize
        self.conv1Strides = conv1Strides



        self.gpu_device = gpu_device
        self.model = None

    def fit(self, X, y):
        if self.target_dimension > 1:
            le = LabelEncoder()
            label = le.fit_transform(y)
            y = np_utils.to_categorical(label, self.target_dimension)

        self.model = self.create_model(X.shape)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=int(self.nb_epochs), verbose=1)
        return self

    def predict(self, X):
        if self.target_dimension > 1:
            predict_result = self.model.predict(X, batch_size=self.batch_size)
            max_index = np.argmax(predict_result, axis=1)
            return max_index
        else:
            return self.model.predict(X, batch_size=self.batch_size)

    def create_model(self, input_shape):
        model = Sequential()
        input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[4])

        model.add(keras.layers.Conv3D(input_shape=input_shape, filters=int(self.conv1Filters), kernel_size=int(self.conv1KernelSize), strides=int(self.conv1Strides), kernel_initializer='uniform', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.target_dimension, activation='softmax'))

        optimizer = self.define_optimizer(optimizer_type=self.optimizer,
                                          lr=self.learning_rate,
                                          momentum=self.momentum,
                                          decay=self.learningRateDecay)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        #model.summary()
        return model

    @staticmethod
    def define_optimizer(optimizer_type='sgd', lr=0.001, momentum=0.9, decay=0.01):
        # Todo: use kwargs to allow for additional optimizer tweaking
        try:
            optimizer_class = getattr(keras.optimizers, optimizer_type)
            optimizer = optimizer_class(lr=lr, momentum=momentum, decay=decay)
        except AttributeError as ae:
            raise ValueError('Could not find optimizer:',
                             optimizer_type, ' - check spelling!')

        return optimizer

    def getValidationResult(self, testData, testLabels):
        if self.target_dimension > 1:
            le = LabelEncoder()
            label = le.fit_transform(testLabels)
            testLabels = np_utils.to_categorical(label, self.target_dimension)

        result = self.model.evaluate(testData, testLabels, batch_size=self.batch_size, verbose=0)
        metric = self.model.metrics_names
        res = dict(zip(metric, result))

        return res