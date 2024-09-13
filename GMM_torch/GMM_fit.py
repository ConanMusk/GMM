import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Step 1: Generate a 256x256 Gamma-distributed speckle noise image
M=2
shape_param = M  # Shape parameter for the Gamma distribution
scale_param = 1/M  # Scale parameter for the Gamma distribution
gamma_noise_image = np.random.gamma(shape=shape_param, scale=scale_param, size=(256, 256))

# Step 2: Fit the Gamma noise image using a Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3)  # Using 3 components for GMM
gamma_noise_flat = gamma_noise_image.flatten().reshape(-1, 1)  # Flatten the image for fitting
gmm.fit(gamma_noise_flat)

# Step 3: Sample new image data from the fitted GMM
gmm_sample_flat = gmm.sample(256 * 256)[0]
gmm_sample_image = gmm_sample_flat.reshape(256, 256)

# Step 4: Plot the original image, GMM-sampled image, and histograms with GMM fits
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Original Gamma noise image
axs[0, 0].imshow(gamma_noise_image, cmap='gray')
axs[0, 0].set_title('Original Gamma Noise Image')

# GMM sampled image
axs[0, 1].imshow(gmm_sample_image, cmap='gray')
axs[0, 1].set_title('GMM Sampled Image')

# Histogram of original image with GMM fit
axs[1, 0].hist(gamma_noise_flat, bins=50, density=True, alpha=0.6, color='gray')
x = np.linspace(np.min(gamma_noise_flat), np.max(gamma_noise_flat), 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)
axs[1, 0].plot(x, pdf, '-r')
axs[1, 0].set_title('Original Image Histogram with GMM Fit')

# Histogram of GMM sampled image
gmm_sample_flat = gmm_sample_image.flatten().reshape(-1, 1)
axs[1, 1].hist(gmm_sample_flat, bins=50, density=True, alpha=0.6, color='gray')
axs[1, 1].set_title('GMM Sampled Image Histogram')

plt.tight_layout()
plt.show()
