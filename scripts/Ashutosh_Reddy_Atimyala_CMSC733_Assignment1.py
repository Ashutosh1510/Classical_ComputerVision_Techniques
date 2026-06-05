# Exported from Ashutosh_Reddy_Atimyala_CMSC733_Assignment1.ipynb
# Notebook markdown is preserved as comments so GitHub can render the source reliably.

# %% [markdown]
# Assignment 1: 
#
# Name: **Ashutosh Reddy Atimyala**
#
# UID: **118442129**
#
# Please submit to Gradescope
# - a PDF containing all outputs (by executing **Run all**)
# - your ipynb notebook containing all the code
#
# I understand the policy on academic integraty (collaboration and the use of online material).
# Please sign your name here: **Ashutosh Reddy Atimyala**

# %% [markdown]
# # Part A: Hybrid Image (25 Points)

# %% [markdown]
# ## Overview
#
# A hybrid image is the sum of a *low-pass filtered* version of the one image and a *high-pass filtered* version of a second image. There is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. This is called the “cutoff-frequency”. In the paper it is suggested to use two cutoff frequencies (one tuned for each image) and you are free to try that, as well. In the starter code, the cutoff frequency is controlled by changing the standard deviation of the Gausian filter used in constructing the hybrid images. [This](https://drive.google.com/uc?id=187FjBJLwnYXhylx08Vdh1SAA3AO-imYv) is the sample example.
#
# NOTE: 
#
# 1. Reading [this](https://stanford.edu/class/ee367/reading/OlivaTorralb_Hybrid_Siggraph06.pdf) will help in understanding Part A.
#
# 2. You can use any image processing libraries of your choice such as skimage or cv2; in python.
#
# We provided 7 pairs of aligned images. The alignment is important because it affects the perceptual grouping (read the paper for details). We encourage you to create additional examples (e.g. change of expression, morph between different objects, change over time, etc.).
#
# You are required to provide **THREE hybrid image results** and for ONE of your favorite result, please provide answers to the following **FOUR sub-parts** mentioned in the write-up.

# %% [markdown]
# ## Data
#
# **WARNING: Colab deletes all files everytime runtime is disconnected. Make sure to re-download the inputs when it happens.**

# %%
# Download Data -- run this cell only one time per runtime
!gdown 1KTDxPAkQam29YKtoX5dKPnLKpUOWCanC
!unzip "/content/hybrid_pyramid_input.zip" -d "/content/"

# %% [markdown]
# ## Code

# %%
# Helper Functions

def vis_hybrid_image(hybrid_image):
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = hybrid_image.shape[2] # counting how many color channels the input has
  output = hybrid_image
  cur_image = hybrid_image

  for i in range(2, scales):
      # add padding
      output = np.concatenate((output, np.ones((original_height, padding, num_colors), dtype=int)), axis=1)      
      # dowsample image;
      width = int(cur_image.shape[1] * scale_factor)
      height = int(cur_image.shape[0] * scale_factor)
      dim = (width, height)
      cur_image = cv2.resize(cur_image, dim, interpolation = cv2.INTER_LINEAR)
      # pad the top and append to the output
      tmp = np.concatenate((np.ones((original_height-cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
      output = np.concatenate((output, tmp), axis=1)
  
  output = (output * 255).astype(np.uint8)
  return output

def read_image(image_path):
  image = cv2.imread(image_path)
  return image

def gaussian_2D_filter(size, cutoff_frequency):
  gauss_1d = cv2.getGaussianKernel(size,cutoff_frequency)
  gauss_2d = gauss_1d * gauss_1d.T
  return gauss_2d

def imgfilter(image, filter):
  img_filter = cv2.filter2D(image, -1 , filter)
  return img_filter

def log_mag_FFT(image):
  magnitude_spectrum = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))
  return magnitude_spectrum

# %%
# Import necessary packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hybrid_image(image_1,image_2,cutoff_frequency):
  filter_size = cutoff_frequency*4+1
  """cutoff_frequency is the standard deviation, in pixels, of the 
  Gaussian blur that will remove the high frequencies from one image (image_1) and 
  remove the low frequencies from another image (image_2) (to do so, subtract a blurred
  version from the original version). You will want to tune this for every image pair to get the best results."""

  filter = gaussian_2D_filter(filter_size, cutoff_frequency)

  """Use imgfilter() to create 'low_frequencies' and 'high_frequencies' and then combine them to create 'hybrid_image'.
  Remove the high frequencies from image_1 by blurring it. The amount of blur that works best will vary with different image pairs."""

  blurred_image1 = imgfilter(image_1, filter)
  low_frequencies = blurred_image1

  """Remove the low frequencies from image_2. The easiest way to do this is to
  subtract a blurred version of image_2 from the original version of image_2.
  This will give you an image centered at zero with negative values."""

  blurred_image2 = imgfilter(image_2, filter)
  high_frequencies = image_2-blurred_image2

  """Combine the high frequencies and low frequencies to obtain hybrid_image."""
  hybrid_image= low_frequencies + high_frequencies

  """Firstly, visualize low_frequencies, high_frequencies, and the hybrid image."""

  """Secondly, also visualize log magnitude of Fourier Transform of the above.
  HINT: You may use np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))) to achieve it."""
  FFT_image_low = log_mag_FFT(low_frequencies)
  FFT_image_high = log_mag_FFT(high_frequencies)
  FFT_image_hybrid = log_mag_FFT(hybrid_image)
  FFT_image_image1 = log_mag_FFT(image_1)
  FFT_image_image2 = log_mag_FFT(image_2)


  """Thirdly, visualize hybrid_image_scale using helper function vis_hybrid_image.
  Lastly, save all your outputs."""
  hybrid_image_scale = vis_hybrid_image(hybrid_image)

  fig, axarr = plt.subplots(3, 4,figsize=(25, 10))
  fig.suptitle('VISUALIZATION OF Original and Filtered Images, Hybrid Image, with their FFT Magnitude and Hybrid Scale Image ', fontsize=16)
  plt.gcf().set_facecolor('white')

  axarr[0,0].imshow(cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
  axarr[0,0].set_title("Image 1")
  axarr[0,0].axis('off')

  axarr[0,1].imshow(cv2.cvtColor(FFT_image_image1.astype('uint8'), cv2.COLOR_RGB2GRAY),cmap='jet')
  axarr[0,1].set_title("Fourier Transform of Image 1")
  axarr[0,1].axis('off')

  axarr[0,2].imshow(cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR))
  axarr[0,2].set_title("Image 2")
  axarr[0,2].axis('off')

  axarr[0,3].imshow(cv2.cvtColor(FFT_image_image2.astype('uint8'), cv2.COLOR_RGB2GRAY),cmap='jet')
  axarr[0,3].set_title("Fourier Transform of Image 2")
  axarr[0,3].axis('off')

  axarr[1,0].imshow(cv2.cvtColor(low_frequencies, cv2.COLOR_RGB2BGR))
  axarr[1,0].set_title("Low Frequency of image1")
  axarr[1,0].axis('off')

  axarr[1,1].imshow(cv2.cvtColor(FFT_image_low.astype('uint8'), cv2.COLOR_RGB2GRAY),cmap='jet')
  axarr[1,1].set_title("Fourier Transform of Low frequency image")
  axarr[1,1].axis('off')

  axarr[1,2].imshow(cv2.cvtColor(high_frequencies, cv2.COLOR_RGB2BGR))
  axarr[1,2].set_title("High Frequency of image2")
  axarr[1,2].axis('off')

  axarr[1,3].imshow(cv2.cvtColor(FFT_image_high.astype('uint8'), cv2.COLOR_RGB2GRAY),cmap='jet')
  axarr[1,3].set_title("Fourier Transform of High frequency image")
  axarr[1,3].axis('off')

  axarr[2,0].imshow(cv2.cvtColor(hybrid_image, cv2.COLOR_RGB2BGR))
  axarr[2,0].set_title("Hybrid Image")
  axarr[2,0].axis('off')

  axarr[2,1].imshow(cv2.cvtColor(FFT_image_hybrid.astype('uint8'), cv2.COLOR_RGB2GRAY),cmap='jet')
  axarr[2,1].set_title("Fourier Transform of Hybrid image")
  axarr[2,1].axis('off')

  axarr[2,2].imshow(cv2.cvtColor(hybrid_image_scale, cv2.COLOR_RGB2BGR))
  axarr[2,2].set_title("Hybrid Image Scale")
  axarr[2,2].axis('off')

  axarr[2,3].set_visible(False)

image_1a = read_image("/content/data/bicycle.bmp")
image_2a = read_image('/content/data/motorcycle.bmp')
hybrid_image(image_1a,image_2a,2)

image_1b = read_image("/content/data/plane.bmp")
image_2b = read_image('/content/data/bird.bmp')
hybrid_image(image_1b,image_2b,8)


image_1c = read_image("/content/data/einstein.bmp")
image_2c = read_image('/content/data/marilyn.bmp')
hybrid_image(image_1c,image_2c,3)

# %% [markdown]
# ## **Write-up**
#
#
# 1.   Provide the original and filtered images.
# 2.   Provide the the hybrid image and hybrid_image_scale using given helper function *vis_hybrid_image*.
# 3.   Log magnitude of the Fourier transform of the two original images, the filtered images, and the hybrid image.
# 4.   Briefly explain how this works, using your favorite results as illustrations.

# %% [markdown]
# **Hybrid Image:**
#
# In general, hybrid images describe how frequencies play a role in visualizing the image from certain distances where high frequency of the image dominates when the distance is lower while the low frequency of the image dominates then the distance is larger.
#
# With respect to example-2 of the plane and the bird, the low frequency image, i.e, image of the plane(image_1) is visible from a larger distance while the distance between the image and the viewer minimizes the high frequency image, i.e, image of the bird(image_2) would be visible. This addition of a low frequency image and a high frequency image is called as 'Hybrid Image'.      

# %% [markdown]
# # Part B: Pyramid Image (25 Points)

# %% [markdown]
# ## Overview
# Choose an image that has interesting variety of textures (from Flickr or your own images). The images should be atleast 640X480 pixels and converted to grayscale. Write code for a Gaussian and Laplacian pyramid of level N (use for loops). In each level, the resolution should be reduced by a factor of 2. Show the pyramids for your chosen image in your write-up. Here is an [example](https://drive.google.com/uc?id=17Y287EA-GJ2z0wtm_M7StIWsXyFeHvrz).

# %% [markdown]
# ## Data
#
# **WARNING: Colab deletes all files everytime runtime is disconnected. Make sure to re-download the inputs when it happens.**

# %%
# Download Data -- run this cell only one time per runtime
# !gdown 1oOxF1fChnRtrJ7sa_L3G8UVkKlh6IOlT
!gdown 1R2hxcMNnlB6zPKma659XkW33WzsSxDY2
# !unzip "/content/hybrid_pyramid_input.zip" -d "/content/"

# %% [markdown]
# ## Code

# %%
# Populate Helper Functions:

def pyramidsGL(image, num_levels):
  ''' Creates Gaussian (G) and Laplacian (L) pyramids of level "num_levels" from image im. 
  G and L are list where G[i], L[i] stores the i-th level of Gaussian and Laplacian pyramid, respectively. '''
  layer = image.copy()
  gp = [layer]    #Gaussian Pyramid
  lp = []         # Laplacian Pyramid
  for i in range(num_levels):
    blur = cv2.GaussianBlur(gp[i], (5,5),5)
    laplacian = gp[i]-blur
    width = int(blur.shape[1] / 2)
    height = int(blur.shape[0] / 2)
    layer = cv2.resize(blur,(width,height))
    gp.append(layer)
    lp.append(laplacian)
  gp.pop(-1)
  return gp, lp

def display_resizedpyramids(gp, lp, num_levels):
  gp_resized, lp_resized=[],[]
  for i in range(num_levels):
    img_gp= cv2.resize(gp[i],(img.shape[0],img.shape[1]))
    img_lp = cv2.resize(lp[i],(img.shape[0],img.shape[1]))
    gp_resized.append(img_gp)
    lp_resized.append(img_lp)
  fig, axarr = plt.subplots(2, 5,figsize=(25, 10))
  plt.gcf().set_facecolor('white')
  fig.suptitle('Gaussian and Laplacian Pyramids of level 5', fontsize=16)

  axarr[0,0].imshow(gp_resized[0],cmap='gray')
  axarr[0,0].set_title("G1")
  axarr[0,0].axis('off')

  axarr[0,1].imshow(gp_resized[1],cmap='gray')
  axarr[0,1].set_title("G2")
  axarr[0,1].axis('off')

  axarr[0,2].imshow(gp_resized[2],cmap='gray')
  axarr[0,2].set_title("G3")
  axarr[0,2].axis('off')

  axarr[0,3].imshow(gp_resized[3],cmap='gray')
  axarr[0,3].set_title("G4")
  axarr[0,3].axis('off')

  axarr[0,4].imshow(gp_resized[4],cmap='gray')
  axarr[0,4].set_title("G5")
  axarr[0,4].axis('off')

  axarr[1,0].imshow(lp_resized[0],cmap='gray')
  axarr[1,0].set_title("L1")
  axarr[1,0].axis('off')

  axarr[1,1].imshow(lp_resized[1],cmap='gray')
  axarr[1,1].set_title("L2")
  axarr[1,1].axis('off')

  axarr[1,2].imshow(lp_resized[2],cmap='gray')
  axarr[1,2].set_title("L3")
  axarr[1,2].axis('off')

  axarr[1,3].imshow(lp_resized[3],cmap='gray')
  axarr[1,3].set_title("L4")
  axarr[1,3].axis('off')

  axarr[1,4].imshow(lp_resized[4],cmap='gray')
  axarr[1,4].set_title("L5")
  axarr[1,4].axis('off')
  plt.show()
  return gp_resized, lp_resized


def log_mag_FFT(image):
  magnitude_spectrum = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))
  return magnitude_spectrum

def displayPyramids(G, L):

  '''Role of this function is to display intensity and Fast Fourier Transform (FFT) images of pyramids.
  NOTE: You may re-use your helper function  "log_mag_FFT" to compute this.'''
  fig, axarr_fft = plt.subplots(2, 5,figsize=(25, 10))
  plt.gcf().set_facecolor('white')
  fig.suptitle(' Fast Fourier Transform (FFT) of Gaussian and Laplacian Pyramids of level 5', fontsize=16)

  axarr_fft[0,0].imshow(log_mag_FFT(G[0]),cmap='jet')
  axarr_fft[0,0].set_title("G1_FFT")
  axarr_fft[0,0].axis('off')

  axarr_fft[0,1].imshow(log_mag_FFT(G[1]),cmap='jet')
  axarr_fft[0,1].set_title("G2_FFT")
  axarr_fft[0,1].axis('off')

  axarr_fft[0,2].imshow(log_mag_FFT(G[2]),cmap='jet')
  axarr_fft[0,2].set_title("G3_FFT")
  axarr_fft[0,2].axis('off')

  axarr_fft[0,3].imshow(log_mag_FFT(G[3]),cmap='jet')
  axarr_fft[0,3].set_title("G4_FFT")
  axarr_fft[0,3].axis('off')

  axarr_fft[0,4].imshow(log_mag_FFT(G[4]),cmap='jet')
  axarr_fft[0,4].set_title("G5_FFT")
  axarr_fft[0,4].axis('off')

  axarr_fft[1,0].imshow(log_mag_FFT(L[0]),cmap='jet')
  axarr_fft[1,0].set_title("L1_FFT")
  axarr_fft[1,0].axis('off')

  axarr_fft[1,1].imshow(log_mag_FFT(L[1]),cmap='jet')
  axarr_fft[1,1].set_title("L2_FFT")
  axarr_fft[1,1].axis('off')

  axarr_fft[1,2].imshow(log_mag_FFT(L[2]),cmap='jet')
  axarr_fft[1,2].set_title("L3_FFT")
  axarr_fft[1,2].axis('off')

  axarr_fft[1,3].imshow(log_mag_FFT(L[3]),cmap='jet')
  axarr_fft[1,3].set_title("L4_FFT")
  axarr_fft[1,3].axis('off')

  axarr_fft[1,4].imshow(log_mag_FFT(L[4]),cmap='jet')
  axarr_fft[1,4].set_title("L5_FFT")
  axarr_fft[1,4].axis('off')
  plt.show()

def reconstructLaplacianPyramid(L):
  '''Given a Laplacian Pyramid L, reconstruct an image img.'''
  for i in range((len(L)-2), -1,-1):
    _h = L[i+1].shape[1]
    _w = L[i+1].shape[0]
    if i == len(L)-2:
      G_rec = L[i] + cv2.GaussianBlur(cv2.resize(L[i+1], (int(_h*2), int(_w*2))),(5,5),1)
    else:
      G_rec = L[i] + cv2.GaussianBlur(cv2.resize(G_rec, (int(G_rec.shape[1]*2),int(G_rec.shape[0]*2))),(5,5),1)
  return G_rec
  

# %%
""" 
Steps:
1. Load an image im.
2. Call function pyramidsGL with image and num_levels = 5
3. Call function displayPyramids with G, L
4. Call function reconstructLaplacianPyramid with the generated L
5. Compute reconstruction error with L2 norm and print the error value.
"""
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from PIL import Image

img = cv2.imread('/content/wolf.jpg',0)
# cv2_imshow(img)
num_levels = 5
gp,lp= pyramidsGL(img,num_levels)
gp_resized,lp_resized = display_resizedpyramids(gp,lp,num_levels)
displayPyramids(gp_resized,lp_resized)
lp[-1] = gp[-1]
img_reconstructed = reconstructLaplacianPyramid(lp)

##### Displaying 'ORIGINAL AND RECONSTRUCTED' Images.
fig1, axarr1 = plt.subplots(1, 2,figsize=(15, 5))
fig1.suptitle(' Original and Reconstructed Images', fontsize=16)

plt.gcf().set_facecolor('white')
axarr1[0].imshow(img, cmap='gray')
axarr1[0].set_title("Original Image")
axarr1[0].axis('off')

axarr1[1].imshow(img_reconstructed, cmap='gray')
axarr1[1].set_title("Reconstructed Image")
axarr1[1].axis('off')

# %% [markdown]
# ## **Write-up**
#
# 1. (10 Points) Display a Gaussian and Laplacian pyramid of level 5 (using your code). It should be formatted similar to [this](https://drive.google.com/file/d/1mAommQeJsp7WS8QCrZRcr8cQiltPPOh2/view?usp=sharing) figure.
#
# 2. (10 Points) Display the FFT amplitudes of your Gaussian/Laplacian pyramids Appropriate display ranges (from 0 to 1) should be chosen so that the changes in frequency in different levels of the pyramid are clearly visible. Explain  what the Laplacian and Gaussian pyramids are doing in terms of frequency. [This](https://drive.google.com/file/d/1BqTPKq6Mqqxl5jNNPkvx4JOA5MRgVq08/view?usp=sharing) looks like the expected output.
#
# 3. (5 Points) Image Reconstruction

# %% [markdown]
# In Gaussian pyramid, smoothing removes high-frequency components resulting in aliasing and displays low resolution images as the number of levels increase.The images lose image-structure information corresponding to high frequency components, decreasing the image quality. 
#
# In Laplacian pyramid, images are obtained from lost high- frequency components and these lost frequencies of the images can be regained.
#
# The Gaussian and Laplacian pyramids, retain the superior frequency orientations and their corresponding low and high frequency components respectively.

# %% [markdown]
# # Part C: Edge detection (25 points)

# %% [markdown]
# ## Overview
# The main steps of edge detection are: (1) assign a score to each pixel; (2) find local maxima along the direction perpendicular to the edge. Sometimes a third step is performed where local evidence is propagated so that long contours are more confident or strong edges boost the confidence of nearby weak edges. Optionally, a thresholding step can then convert from soft boundaries to hard binary boundaries. Here are sample outputs.
#
# <table><tr>
# <td> <img src="https://drive.google.com/uc?id=1orUji5-1CzjWmHk0g5y5kOVFhshNfhN8" alt="Drawing" style="width: 250px;"/> </td>
# <td> <img src="https://drive.google.com/uc?id=1npyMjhlRAeP1GaukV38SOlCe-O0whX37" alt="Drawing" style="width: 250px;"/> </td>
# </tr></table>
# <table><tr>
# <td> <img src="https://drive.google.com/uc?id=1TX54zNTG6q5ajitwV024FS-nOJiVP2VN" alt="Drawing" style="width: 250px;"/> </td>
# <td> <img src="https://drive.google.com/uc?id=1dPPSFDmakh8DQwlpTYmNXJPNHapdZF8S" alt="Drawing" style="width: 250px;"/> </td>
# </tr></table>
#
# **Hint:** Reading these papers will help understanding and may help with the programming assignment.
#
# [The design and use of steerable filters](http://people.csail.mit.edu/billf/papers/steerpaper91FreemanAdelson.pdf)
#
# [Berkeley Pb Detector](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/mfm-pami-boundary.pdf)
#
# [Multi-scale edge detection](https://home.ttic.edu/~xren/publication/xren_eccv08_multipb.pdf)

# %% [markdown]
# ## Data
# We have provided 50 test images and the codes associated to download the unzip the data. Your job is to build a simple gradient-based edge detector and to extend it using multiple oriented filters.
#
# **WARNING: Colab deletes all files everytime runtime is disconnected. Make sure to re-download the inputs when it happens.**

# %%
# Download Data -- run this cell only one time per runtime
!gdown 1zgblBWTQ847yZKnRmM1QrRiEWu1WvEo7
!unzip "/content/edge_detection_inputs.zip" -d "/content/"

# %%
# Import necessary packages
import numpy as np
import cv2
from google.colab.patches import cv2_imshow ## Use this to show image in colab

# %% [markdown]
# ## Subpart 1: Simple edge detection (8 points)
# Build a simple gradient-based edge detector that includes the following functions
# ```
# def gradientMagnitude(im, signma)
# ```
# This function should take an RGB image as input, smooth the image with Gaussian std=sigma, compute the x and y gradient values of the smoothed image, and output image maps of the gradient magnitude and orientation at each pixel. You can compute the gradient magnitude of an RGB image by taking the L2-norm of the R, G, and B gradients. The orientation can be computed from the channel corresponding to the largest gradient magnitude. The overall gradient magnitude is the L2-norm of the x and y gradients. mag and theta should be the same size as im.
#
# ```
# def edgeGradient(im):
# ```
# This function should use gradientMagnitude to compute a soft boundary map and then perform non-maxima suppression. For this assignment, it is acceptable to perform non-maxima suppression by retaining only the magnitudes along the binary edges produce by the Canny edge detector: `cv2.Canny(im)`. 
#
# If desired, the boundary scores can be rescaled, e.g., by raising to an exponent: `mag2 = mag.^0.7` , which is primarily useful for visualization. 

# %%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def gradientMagnitude(im, sigma):
  '''
  im: input image
  sigma: standard deviation value to smooth the image

  outputs: gradient magnitude and gradient direction of the image
  '''
  im = cv2.GaussianBlur(im,(5,5),sigma)

  lap = cv2.Laplacian(im,cv2.CV_64F,ksize=3) 
  lap = np.uint8(np.absolute(lap))
  
  sobelx= cv2.Sobel(im,0, dx=1,dy=0)
  sobelx= np.uint8(np.absolute(sobelx))
  
  sobely= cv2.Sobel(im,0, dx=0,dy=1)
  sobely = np.uint8(np.absolute(sobely))
  combined = sobelx + sobely
  combined1 = combined/combined.max()*255
  g_orientation = np.zeros_like(im)
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      max_idx = np.argmax(combined[i,j,:])
      g_orientation[i,j,max_idx] = np.arctan2(sobely[i,j,max_idx], sobelx[i,j,max_idx])
  g_orientation2 = g_orientation / g_orientation.max() * 255

  return combined,g_orientation, g_orientation2


def edgeGradient(im):
  '''
  im: input image

  output: a soft boundary map of the image
  '''
  
  img = cv2.Canny(im,50,150, L2gradient=True)
  return img
  
def simple_edge(img):
  sigma = 1
  g_mag, g_orient, g_orient_norm  = gradientMagnitude(img,sigma)
  canny_img = edgeGradient(g_mag)

  fig, axarr = plt.subplots(1, 4,figsize=(25, 10))
  fig.suptitle('Simple Edge Detection', fontsize=16)
  plt.gcf().set_facecolor('white')

  axarr[0].imshow(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[0].set_title("Image")
  axarr[0].axis('off')

  axarr[1].imshow(cv2.cvtColor(g_mag.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[1].set_title("Gradient Magnitude")
  axarr[1].axis('off')

  axarr[2].imshow(cv2.cvtColor(g_orient_norm.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[2].set_title("Gradient Orientation")
  axarr[2].axis('off')

  axarr[3].imshow(canny_img, cmap='gray')
  axarr[3].set_title("Using Canny Edge")
  axarr[3].axis('off')


img1 = cv2.imread("/content/edge_detection_inputs/86000.jpg")
simple_edge(img1)

img2 = cv2.imread("/content/edge_detection_inputs/102061.jpg")
simple_edge(img2)

img3 = cv2.imread("/content/edge_detection_inputs/69015.jpg")
simple_edge(img3)

img4 = cv2.imread("/content/edge_detection_inputs/78004.jpg")
simple_edge(img4)

img5 = cv2.imread("/content/edge_detection_inputs/101085.jpg")
simple_edge(img5)

# %% [markdown]
# ## Subpart 2: Improved Edge Detection (8 points)
# Try to improve your results using a set of oriented filters, rather than the simple derivative of Gaussian approach above, including the following functions:
# ```
# def orientedFilterMagnitude(im):
# ```
# Computes the boundary magnitude and orientation using a set of oriented filters, such as elongated Gaussian derivative filters. Explain your choice of filters in the write-up. Use at least four orientations. One way to combine filter responses is to compute a boundary score for each filter (simply by filtering with it) and then use the max and argmax over filter responses to compute the magnitude and orientation for each pixel.
# ```
# def edgeOrientedFilters(im):
# ```
# Similar to Subpart 1, this should call orientedFilterMagnitude, perform the non-maxima suppression, and output the final soft edge map.

# %%
from IPython.core.interactiveshell import magic
# Import necessary packages
import numpy as np
import cv2
from google.colab.patches import cv2_imshow ## Use this to show image in colab


def orientedFilterMagnitude(im):
  '''
  im: input image

  outputs: gradient magnitude and gradient direction of the image
  '''
  gauss_1d = cv2.getGaussianKernel(5,1)
  gauss_2d = gauss_1d * gauss_1d.T
  sobel_x = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
  sobel_y = sobel_x.T

  dog_x =  cv2.filter2D(gauss_2d,-1,sobel_x) # derivative of gaussian along x direction
  dog_y =  cv2.filter2D(gauss_2d,-1,sobel_y) # derivative of gaussian along y direction

  #Filters
  filter1 = dog_x * np.cos(np.deg2rad(0)) + dog_y*np.sin(np.deg2rad(0))
  filter2 = dog_x * np.cos(np.deg2rad(30)) + dog_y*np.sin(np.deg2rad(30))
  filter3 = dog_x * np.cos(np.deg2rad(60)) + dog_y*np.sin(np.deg2rad(60))
  filter4 = dog_x * np.cos(np.deg2rad(90)) + dog_y*np.sin(np.deg2rad(90))
  filter5 = dog_x * np.cos(np.deg2rad(120)) + dog_y*np.sin(np.deg2rad(120))
  filter6 = dog_x * np.cos(np.deg2rad(150)) + dog_y*np.sin(np.deg2rad(150))

  blur = cv2.GaussianBlur(im,(5,5),1)

  #Gradient Magnitude 
  Gm_1 = cv2.filter2D(blur,-1,filter1)
  Gm_2 = cv2.filter2D(blur,-1,filter2)
  Gm_3 = cv2.filter2D(blur,-1,filter3)
  Gm_4 = cv2.filter2D(blur,-1,filter4)
  Gm_5 = cv2.filter2D(blur,-1,filter5)
  Gm_6 = cv2.filter2D(blur,-1,filter6)

  mag = np.zeros_like(im)
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      mag[i,j,0]= max(Gm_1[i,j,0],Gm_2[i,j,0],Gm_3[i,j,0],Gm_4[i,j,0],Gm_5[i,j,0],Gm_6[i,j,0])
      mag[i,j,1]= max(Gm_1[i,j,1],Gm_2[i,j,1],Gm_3[i,j,1],Gm_4[i,j,1],Gm_5[i,j,1],Gm_6[i,j,1])
      mag[i,j,2]= max(Gm_1[i,j,2],Gm_2[i,j,2],Gm_3[i,j,2],Gm_4[i,j,2],Gm_5[i,j,2],Gm_6[i,j,2])
  mag2 = mag/mag.max()*255
  g_orientation = np.zeros_like(im)
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      g_orientation[i,j,:] = np.arctan2(Gm_4[i,j,:], Gm_1[i,j,:])
  g_orientation2 = g_orientation / g_orientation.max() * 255
  
  return mag,g_orientation,g_orientation2,mag2

def edgeOrientedFilters(im):
  '''
  im: input image

  output: a soft boundary map of the image
  '''
  img_canny = cv2.Canny(im,5,150, L2gradient=True)
  return img_canny
  
def improved_edge(img):
  mag,g_orientation,g_orientation_norm,mag_norm = orientedFilterMagnitude(img)
  canny_img = edgeOrientedFilters(mag)
  fig, axarr = plt.subplots(1, 4,figsize=(25, 10))
  fig.suptitle('Improved Edge Detection', fontsize=16)
  plt.gcf().set_facecolor('white')

  axarr[0].imshow(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[0].set_title("Image")
  axarr[0].axis('off')

  axarr[1].imshow(cv2.cvtColor(mag_norm.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[1].set_title("Gradient Magnitude")
  axarr[1].axis('off')

  axarr[2].imshow(cv2.cvtColor(g_orientation_norm.astype('uint8'), cv2.COLOR_RGB2BGR))
  axarr[2].set_title("Gradient Orientation")
  axarr[2].axis('off')

  axarr[3].imshow(canny_img, cmap='gray')
  axarr[3].set_title("Using Canny Edge")
  axarr[3].axis('off')


img1 = cv2.imread("/content/edge_detection_inputs/86000.jpg")
improved_edge(img1)

img2 = cv2.imread("/content/edge_detection_inputs/102061.jpg")
improved_edge(img2)

img3 = cv2.imread("/content/edge_detection_inputs/69015.jpg")
improved_edge(img3)

img4 = cv2.imread("/content/edge_detection_inputs/78004.jpg")
improved_edge(img4)

img5 = cv2.imread("/content/edge_detection_inputs/101085.jpg")
improved_edge(img5)

# %%
# Import necessary packages
import numpy as np
import cv2
from google.colab.patches import cv2_imshow ## Use this to show image in colab
import matplotlib.pyplot as plt

gauss_1d = cv2.getGaussianKernel(5,1)
gauss_2d = gauss_1d * gauss_1d.T
sobel_x = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
sobel_y = sobel_x.T


dog_x =  cv2.filter2D(gauss_2d,-1,sobel_x) # derivative of gaussian along x direction
dog_y =  cv2.filter2D(gauss_2d,-1,sobel_y)
filter1 = dog_x * np.cos(np.deg2rad(0)) + dog_y*np.sin(np.deg2rad(0))
filter2 = dog_x * np.cos(np.deg2rad(30)) + dog_y*np.sin(np.deg2rad(30))
filter3 = dog_x * np.cos(np.deg2rad(60)) + dog_y*np.sin(np.deg2rad(60))
filter4 = dog_x * np.cos(np.deg2rad(90)) + dog_y*np.sin(np.deg2rad(90))
filter5 = dog_x * np.cos(np.deg2rad(120)) + dog_y*np.sin(np.deg2rad(120))
filter6 = dog_x * np.cos(np.deg2rad(150)) + dog_y*np.sin(np.deg2rad(150))

fig, axarr = plt.subplots(1, 6,figsize=(25, 5))
fig.suptitle('Filters', fontsize=16)
plt.gcf().set_facecolor('white')

axarr[0].imshow(filter1)
axarr[0].set_title("Filter1")
axarr[0].axis('off')

axarr[1].imshow(filter2)
axarr[1].set_title("Filter2")
axarr[1].axis('off')

axarr[2].imshow(filter3)
axarr[2].set_title("filter3")
axarr[2].axis('off')

axarr[3].imshow(filter4)
axarr[3].set_title("filter4")
axarr[3].axis('off')


axarr[4].imshow(filter5)
axarr[4].set_title("filter5")
axarr[4].axis('off')


axarr[5].imshow(filter6)
axarr[5].set_title("filter6")
axarr[5].axis('off')

# %% [markdown]
# ## Write-up (9 points)
#
# 1.   **(2 points)** Description of any design choices and parameters
# 2.   **(1 points)** The bank of filters used for Subpart 2 ([plt.imshow with extent](https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect/13390798#13390798) or [cv2.normalize to convert output into grayscale](https://stackoverflow.com/questions/39808545/implement-mat2gray-in-opencv-with-python) may help with visualization)
# 3.   **(5 points)** Qualitative results: choose five example images; show input images and outputs of each edge detector
# 4.   **(1 points)** Discuss the quality of your outputs and state one possible way for improvement. Improvements could provide, for example, a better boundary pixel score or a better suppression technique. Your idea could come from a paper you read, but cite any sources of ideas.

# %% [markdown]
#
# 1. The design choices and parameters considered during the implementation were the cut-off frequencies for each data of the images, kernel size of the gaussian 2Dfilter. For the improved edged detection, the DoG filter with 6 different orientations were considered to detect edges. The maximum values among the 6 filtered images along the RGB channels were computed for determining gradient magnitude. For computing the gradient orientation, two random filtered images were selected out of the six filtered images as Gy and Gx.
#
# 4. From the results, it can be noticed that an improved criteria can be used for calculating gradient magnitude and gradient orientation instead of using the maximum valued filter for the gradient magnitude and selecting random filtered images for the gradient orientation respectively.

# %% [markdown]
# # Part D: Template Matching (25 points) 

# %% [markdown]
# ## Overview
# The goal of this part is to build a template maching algorithm for where's waldo puzzle. 
# The end product should be finding waldo in puzzle images. 

# %% [markdown]
# ## Data
#
# **WARNING: Colab deletes all files everytime runtime is disconnected. Make sure to re-download the inputs when it happens.**

# %%
# Download Data -- run this cell only one time per runtime
!gdown 1_PHimFhPSajbTWzAL6-PwM803uzA7Ymb
!unzip "/content/Part4_data.zip" -d "/content/"

# %% [markdown]
# ## Code
#
# We provide the following functions for plotting your results 

# %%
%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(im,title,xticks=[],yticks= [],cv2 = True):
    """
    im :Image to plot
    title : Title of image 
    xticks : List of tick values. Defaults to nothing
    yticks :List of tick values. Defaults to nothing 
    cv2 :Is the image cv2 image? cv2 images are BGR instead of RGB. Default True
    """
    plt.figure()
    plt.imshow(im[:,:,::-1])
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)

# %% [markdown]
# It is always advised for you to visualize and get familar with waldo and puzzle image. 
#
# Visualize both the waldo and puzzle images.

# %%
# map = cv2.imread("/content/Part4_data/puzzle1/pic1.jpeg")
# waldo = cv2.imread("/content/Part4_data/puzzle1/waldo.jpg")

map = cv2.imread("/content/Part4_data/puzzle2/map.jpg")
waldo = cv2.imread("/content/Part4_data/puzzle2/waldo.png")

plot_image(map,'Puzzle')
plot_image(waldo,'Waldo')

# %% [markdown]
# ### Subpart 1: Template Matching with OpenCV
# OpenCV provide Template Matching functions below link. 
#
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
#
# You can use this function for implementing simple where's waldo algorithm.
#
# This part helps you to understand the concepts of Template Matching and OpenCV Library. This is just for reference.

# %%
(waldoHeight, waldoWidth) = waldo.shape[:2]

result = cv2.matchTemplate(map, waldo, cv2.TM_CCOEFF)

(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# grab the bounding box of waldo and extract him from the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = map[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the map except for Waldo
mask = np.zeros(map.shape, dtype = "uint8")
map = cv2.addWeighted(map, 0.25, mask, 0.75, 0)

map[topLeft[1] : botRight[1], topLeft[0] : botRight[0]] = roi

# display the images
result_rgb = cv2.cvtColor(map, cv2.COLOR_RGB2BGR)
plt.figure(figsize = (15, 15))
plt.imshow(result_rgb)

# %% [markdown]
# ### Subpart 2: Template Matching from Scratch
#
# Implement the Sum of Squared Distance (SSD) template matching algorithm from scratch (Don't use cv2.matchTemplate).
#
# Then, show the results of where's waldo for two puzzle images.
#
# Hints: You can borrow the codes from Part1 and Part2. Please read methods for matching with filters in Lecture Slide.  

# %%
import numpy as np
import cv2
from google.colab.patches import cv2_imshow ## Use this to show image in colab
import matplotlib.pyplot as plt

def SSD(image, template):
    I, T = image.astype('float'),template.astype('float')
    H, W = I.shape[0], I.shape[1]
    h, w = T.shape[0], T.shape[1]
    R = np.ones_like(image)*np.inf
    
    for i in range(W-w):
        for j in range(H-h):
            t = (I[j: j+h, i: i+w] - T).ravel()
            R[j, i] = t.dot(t)

    return R

def match_template(image, template):

    R = SSD(image, template)
    _, _, min_loc, max_loc = cv2.minMaxLoc(-R)
    top_left = max_loc
    h,w = template.shape
    botRight = (top_left[0] + w, top_left[1] + h)
    return R, top_left,botRight

def plot(map_puzzle,topLeft,botRight):
  roi = map_puzzle[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]
  mask = np.zeros(map_puzzle.shape, dtype = "uint8")
  image_map = cv2.addWeighted(map_puzzle, 0.25, mask, 0.75, 0)
  image_map[topLeft[1] : botRight[1], topLeft[0] : botRight[0]] = roi
  result_rgb = cv2.cvtColor(image_map, cv2.COLOR_RGB2BGR)
  cv2.rectangle(result_rgb, top_left, botRight, (0,255,0),4)
  plt.figure(figsize = (15, 15))
  plt.axis('off')
  plt.imshow(result_rgb)


####### MAIN FUNCTION #########

##Puzzle-1:
map_puzzle1 = cv2.imread('/content/Part4_data/puzzle1/pic1.jpeg')
gray_map1 = cv2.cvtColor(map_puzzle1.copy(), cv2.COLOR_BGR2GRAY)

waldo_template1 = cv2.imread('/content/Part4_data/puzzle1/waldo.jpg')
gray_waldo1 = cv2.cvtColor(waldo_template1.copy(), cv2.COLOR_BGR2GRAY)

R,  top_left,botRight = match_template(gray_map1,gray_waldo1)

plot(map_puzzle1,top_left,botRight)

##Puzzle-2:
map_puzzle2 = cv2.imread('/content/Part4_data/puzzle2/map.jpg')
gray_map2 = cv2.cvtColor(map_puzzle2.copy(), cv2.COLOR_BGR2GRAY)

waldo_template2 = cv2.imread('/content/Part4_data/puzzle2/waldo.png')
gray_waldo2 = cv2.cvtColor(waldo_template2.copy(), cv2.COLOR_BGR2GRAY)

R,  top_left,botRight = match_template(gray_map2,gray_waldo2)

plot(map_puzzle2,top_left,botRight)
