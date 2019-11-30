import matplotlib.pyplot as plt
import numpy as np
import rasterio


PREDICTIONS = "predictions_aoi_100x100_16x16_enhanced.txt"
TEST_IMG = "13Bands_Cropped.tif"
classes = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 
           4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 
           7: 'Residential', 8: 'River', 9: 'SeaLake'}


start_x, start_y = (350, 700)
SIZE = 100


# Read the Predictions file
rows = {}
with open(PREDICTIONS) as data:
    for row in data:
        values = [int(i) for i in row.split(",")]
        rows[values[0]] = values[1:]
predicted_img = np.array([*rows.values()])


# Write as a raster
with rasterio.open(TEST_IMG) as src_dataset:
    
    test_img = src_dataset.read((2, 3, 4))
    kwds = src_dataset.profile

    kwds['count'] = 1          # Changing the no. of bands to 1
    kwds['dtype'] = 'int32'    # Changing the datatype
    kwds['width'] = kwds['height'] = SIZE
   
    with rasterio.open('predictions_aoi_100x100_16x16_enhanced.tif', 'w', **kwds) as dst_dataset:
        dst_dataset.write(predicted_img, 1)

# Get area of Interest
test_img_aoi = test_img[:, start_x:start_x+SIZE, start_y:start_y+SIZE]

# Plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Test image plot
test_img_aoi_toPlot = ((test_img_aoi / (2**12-1)) * 255).astype('uint8')
img1 = ax1.imshow(np.transpose(test_img_aoi_toPlot, (1, 2, 0)))
ax1.set_title("Test image (in 8-bit)")

# Predicted plot
img2 = ax2.imshow(predicted_img)
cbar = plt.colorbar(img2, fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(classes.values())
ax2.set_title("Predicted raster (ResNet50) 32 X 32")

# Display the plot
plt.show()
