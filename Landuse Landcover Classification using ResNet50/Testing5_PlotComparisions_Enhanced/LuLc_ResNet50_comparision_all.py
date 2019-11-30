from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import rasterio


TEST_IMG = "13Bands_Cropped.tif"
PREDICTED_FULL = "preditions_geotiff_result.tif"

PREDICTED_AOI_8 = "preditions_aoi_8.tif"
PREDICTED_AOI_16 = "preditions_aoi_16.tif"
PREDICTED_AOI_32 = "preditions_aoi_32.tif"
PREDICTED_AOI_8_ENHANCED = "predictions_aoi_8_enhanced.tif"
PREDICTED_AOI_16_ENHANCED = "predictions_aoi_16_enhanced.tif"
PREDICTED_AOI_32_ENHANCED = "predictions_aoi_32_enhanced.tif"

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


# Predicted Image (16X16)
predicted_img_32_enhanced = rasterio.open(PREDICTED_AOI_32_ENHANCED).read(1)


# Predicted Image (8X8)
predicted_img_8_enhanced = rasterio.open(PREDICTED_AOI_8_ENHANCED).read(1)


# Predicted Image (8X8)
predicted_img_16_enhanced = rasterio.open(PREDICTED_AOI_16_ENHANCED).read(1)



# Plotting
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 6))


# Test image plot
test_img_aoi_toPlot = ((test_img_aoi / (2**12-1)) * 255).astype('uint8')
ax1.imshow(np.transpose(test_img_aoi_toPlot, (1, 2, 0)))
ax1.set_title("100 X 100 Image")

# Enhancement
newImage = np.zeros(test_img_aoi.shape, dtype=test_img.dtype)
for bandNo in range(3):
    maxVal = np.max(test_img_aoi[bandNo])
    minVal = np.min(test_img_aoi[bandNo])
    newImage[bandNo] = ((test_img_aoi[bandNo] - minVal) / (maxVal - minVal)) * (2**12-1)


# Test image plot enhanced
test_img_aoi_toPlot = ((newImage / (2**12-1)) * 255).astype('uint8')
ax2.imshow(np.transpose(test_img_aoi_toPlot, (1, 2, 0)))
ax2.set_title("100 X 100 Image - Linearly Enhanced")

# Predicted plot - 64X64
ax3.imshow(predicted_img_64_aoi)
ax3.set_title("64 X 64 Enhancement before enhancement")

# Predicted plot - 8X8
ax4.imshow(predicted_img_8)
ax4.set_title("Prediction (before enh) [upsampled from 8 X 8]")

# Predicted plot - 16X16
ax5.imshow(predicted_img_16_aoi)
ax5.set_title("Prediction (before enh) [upsampled from 16 X 16]")

# Predicted plot - 32X32
ax6.imshow(predicted_img_32)
ax6.set_title("Prediction (before enh) [upsampled from 32 X 32]")



# Predicted plot - 8X8
ax7.imshow(predicted_img_8_enhanced)
ax7.set_title("Prediction (after enh) [upsampled from 8 X 8]")

# Predicted plot - 16X16
ax8.imshow(predicted_img_16_enhanced)
ax8.set_title("Prediction (after enh) [upsampled from 16 X 16]")

# Predicted plot - 32X32
img9 = ax9.imshow(predicted_img_32_enhanced)
ax9.set_title("Prediction (after enh) [upsampled from 32 X 32]")



# Colorbar
cbar = fig.colorbar(img9, ax=[ax9])
cbar.ax.set_yticklabels(classes.values())


# Display the plot
plt.show()
