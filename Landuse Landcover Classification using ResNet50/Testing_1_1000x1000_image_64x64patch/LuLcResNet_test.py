
# Importing the modules
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import rasterio
import sys
print("Modules Loaded Successfully @ %s" % datetime.now())

# Loading the data and model
IMG = "13Bands_Cropped.tif"
MODEL = "LuLc_epoch30_97.h5"
BANDS = (2, 3, 4)
 

# Read the Data and Model
test_img = rasterio.open(IMG).read(BANDS)
print("Data loaded successfully @ %s" % datetime.now())


# Pad the image
test_img = np.pad(test_img, ((0, 0), (32, 32), (32, 32)), mode="empty")


# Test the Image
start = datetime.now()
ROWS = 1000
COLS = 1000

# Load the model one time
LuLcModel = tf.keras.models.load_model(MODEL)

prediction_file = open("predictions.txt", "w")
for i in range(383, ROWS):

    predictions = []
    print("Processing Row: %s @ %s" % (i, datetime.now()))
    for j in range(COLS):
               
        # Create tile
        tile = ((i, i+64), (j, j+64))
        tile_img = test_img[:, i:i+64, j:j+64]

        # Predict
        predicted = LuLcModel.predict(tile_img.reshape(1, 64, 64, 3).astype('float16'))
        predictions.append(str(predicted.argmax()))

        # Optimization        
        del tile_img, predicted
        K.clear_session()
    
    # Flush the predictions of row to file
    prediction_file.write(("%s," % i) + ",".join(predictions) + "\n")
    prediction_file.flush()
    sys.stdout.flush()

    # Optimization
    if i % 10 == 0:
        del LuLcModel
        K.clear_session()
        LuLcModel = tf.keras.models.load_model(MODEL)

end = datetime.now()
print("\nTime Taken for testing: %s" % (end-start))
prediction_file.close()
