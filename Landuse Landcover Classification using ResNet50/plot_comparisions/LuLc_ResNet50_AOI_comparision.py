from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import rasterio


TEST_IMG = "13Bands_Cropped.tif"
PREDICTED_FULL = "precitions_geotiff_result.tif"
PREDICTED_AOI = "precitions_aoi.tif"
classes = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 
           4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 
           7: 'Residential', 8: 'River', 9: 'SeaLake'}

start_x, start_y = (50, 600)
SIZE = 400
window_size = 16


# Test Image and AOI
test_img = rasterio.open(TEST_IMG).read((2, 3, 4))
test_img_aoi = test_img[:, start_x:start_x+SIZE, start_y:start_y+SIZE]


# Predicted Image (64X64) and AOI
predicted_img_64 = rasterio.open(PREDICTED_FULL).read(1)
predicted_img_64_aoi = predicted_img_64[start_x:start_x+SIZE, start_y:start_y+SIZE]


# Predicted Image (16X16) and AOI
predicted_img_16 = rasterio.open(PREDICTED_AOI).read(1)


# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))


# Test image plot
test_img_aoi_toPlot = ((test_img_aoi / (2**12-1)) * 255).astype('uint8')
img1 = ax1.imshow(np.transpose(test_img_aoi_toPlot, (1, 2, 0)))
ax1.set_title("Test image (in 8-bit)")

# Predicted plot - 64X64
img2 = ax2.imshow(predicted_img_64_aoi)
ax2.set_title("Predicted Image (64 X 64)")

# Predicted plot - 16X16
img3 = ax3.imshow(predicted_img_16)
ax3.set_title("Predicted Image (upsampled from 16 X 16)")

# Colorbar
cbar = plt.colorbar(img3, fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(classes.values())


# Display the plot
plt.show()
