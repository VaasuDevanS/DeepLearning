from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import rasterio


TEST_IMG = "13Bands_Cropped.tif"
PREDICTED_FULL = "preditions_geotiff_result.tif"
PREDICTED_AOI_16 = "preditions_aoi_16.tif"
PREDICTED_AOI_8 = "preditions_aoi_8.tif"
PREDICTED_AOI_32 = "preditions_aoi_32.tif"

classes = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 
           4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 
           7: 'Residential', 8: 'River', 9: 'SeaLake'}

start_x, start_y = (350, 700)
SIZE = 100

# Test Image and AOI
test_img = rasterio.open(TEST_IMG).read((2, 3, 4))
test_img_aoi = test_img[:, start_x:start_x+SIZE, start_y:start_y+SIZE]


# Predicted Image (64X64) and AOI
predicted_img_64 = rasterio.open(PREDICTED_FULL).read(1)
predicted_img_64_aoi = predicted_img_64[start_x:start_x+SIZE, start_y:start_y+SIZE]


start_x, start_y = (300, 100)

# Predicted Image (16X16) and AOI
predicted_img_16 = rasterio.open(PREDICTED_AOI_16).read(1)
predicted_img_16_aoi = predicted_img_16[start_x:start_x+SIZE, start_y:start_y+SIZE]


# Predicted Image (8X8)
predicted_img_8 = rasterio.open(PREDICTED_AOI_8).read(1)


# Predicted Image (32X32)
predicted_img_32 = rasterio.open(PREDICTED_AOI_32).read(1)


# Plotting
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 6))
ax3.axis("off")


# Test image plot
test_img_aoi_toPlot = ((test_img_aoi / (2**12-1)) * 255).astype('uint8')
img1 = ax1.imshow(np.transpose(test_img_aoi_toPlot, (1, 2, 0)))
ax1.set_title("Test image (in 8-bit)")

# Predicted plot - 64X64
img2 = ax2.imshow(predicted_img_64_aoi)
ax2.set_title("Predicted Image (64 X 64)")

# Predicted plot - 16X16
img3 = ax4.imshow(predicted_img_16_aoi)
ax4.set_title("Predicted Image (upsampled from 16 X 16)")

# Predicted plot - 8X8
img4 = ax5.imshow(predicted_img_8)
ax5.set_title("Predicted Image (upsampled from 8 X 8)")

# Predicted plot - 32X32
img5 = ax6.imshow(predicted_img_32)
ax6.set_title("Predicted Image (upsampled from 32 X 32)")


# Colorbar
cbar = plt.colorbar(img5, fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(classes.values())


# Display the plot
plt.show()
