
# Importing the modules
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import rasterio
import sys
import cv2
print("Modules Loaded Successfully @ %s" % datetime.now())

# Loading the data and model
IMG = "13Bands_Cropped.tif"
MODEL = "LuLc_epoch30_97.h5"
BANDS = (2, 3, 4)

start_x, start_y = (350, 700)
SIZE = 100
window_size = 8
model_img_size = 64
to_pad = int(window_size / 2)
 

# Read the Data and Model
test_img = rasterio.open(IMG).read(BANDS)
print("Data loaded successfully @ %s" % datetime.now())

# Get area of Interest
test_img_aoi = test_img[:, start_x:start_x+SIZE, start_y:start_y+SIZE]

# Enhancement
newImage = np.zeros(test_img_aoi.shape, dtype=test_img.dtype)
for bandNo in range(3):
    maxVal = np.max(test_img_aoi[bandNo])
    minVal = np.min(test_img_aoi[bandNo])
    newImage[bandNo] = ((test_img_aoi[bandNo] - minVal) / (maxVal - minVal)) * (2**12-1)

# Pad the image
test_img_aoi = np.pad(newImage, ((0, 0), (to_pad, to_pad), (to_pad, to_pad)), mode="empty")
print(test_img_aoi.shape)

# Test the Image
start = datetime.now()

# Load the model one time
LuLcModel = tf.keras.models.load_model(MODEL)

prediction_file = open("predictions_aoi_100x100_8x8_enhanced.txt", "w")

for i in range(SIZE):

    predictions = []
    print("Processing Row: %s @ %s" % (i, datetime.now()))

    for j in range(SIZE):
               
        # Create tile
        tile = ((i, i+window_size), (j, j+window_size))
        tile_img = test_img_aoi[:, i:i+window_size, j:j+window_size]
        
        res_tile = np.zeros((3, model_img_size, model_img_size))
        for k in range(3):
            # Bilinear Interpolation by default
            res_tile[k] = cv2.resize(tile_img[k], dsize=(model_img_size, model_img_size))

        # Predict
        predicted = LuLcModel.predict(res_tile.reshape(1, model_img_size, model_img_size, 3).astype('float16'))
        predictions.append(str(predicted.argmax()))

        # Optimization        
        del tile_img, # predicted
        K.clear_session()
    
    # Flush the predictions of row to file
    prediction_file.write(("%s," % i) + ",".join(predictions) + "\n")
    prediction_file.flush()
    sys.stdout.flush()

    # Optimization
    if i % 10 == 0 and i != 0:
        del LuLcModel
        K.clear_session()
        LuLcModel = tf.keras.models.load_model(MODEL)

end = datetime.now()
print("\nTime Taken for testing: %s" % (end-start))
prediction_file.close()
