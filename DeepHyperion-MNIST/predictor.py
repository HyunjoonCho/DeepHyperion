# For Python 3.6 we use the base keras
import keras
#from tensorflow import keras

import numpy as np

from properties import MODEL1, MODEL2, EXPECTED_LABEL, num_classes


class Predictor:

    # Load the pre-trained model.
    model1 = keras.models.load_model(MODEL1)
    model2 = keras.models.load_model(MODEL2)
    print("Loaded model from disk")

    @staticmethod
    def predict(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)
        explabel = np.argmax(explabel.squeeze())

         #Predictions vector
        predictions1 = Predictor.model1.predict(img)
        predictions2 = Predictor.model2.predict(img)

        prediction1 = np.argsort(-predictions1[0])[0]
        prediction2 = np.argsort(-predictions2[0])[0]

        return prediction1, prediction2, sum([abs(x - y) for x, y in zip(predictions1[0], predictions2[0])])
