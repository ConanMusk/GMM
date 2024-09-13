from gmm.GaussianSoftClustering import GaussianSoftClustering
from data_generator import DataGenerator
from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d
import numpy as np
import matplotlib.pyplot as plt

data = DataGenerator()
# data.show()

gmm_model = GaussianSoftClustering()


# 生成 256x256 Gamma 分布的噪声图像
shape, scale = 30., 1/30  # Gamma 分布的形状参数和尺度参数
gamma_noise = np.random.gamma(shape, scale, (10, 10))
flat =gamma_noise.flatten()
plt.grid(True)
plt.ylabel("Density")
plt.xlabel("value of feature (x)")
plt.title("Gaussian mixtures")
plt.hist(flat, bins=50)
plt.show()
flat = flat.reshape((flat.shape[0], 1))



observations = data.x.copy()
observations = observations.reshape((observations.shape[0], 1))
best_loss, best_parameters = gmm_model.train_EM(observations,
                                                     number_of_clusters=3,
                                                     restarts=200,
                                                     max_iter=100)

mu = best_parameters.mu.ravel()
sigma = best_parameters.sigma.ravel()

mixture = GaussianMixtureFunctor_1d(best_parameters.hidden_states_prior, mu, sigma)

x = np.linspace(-1, 11, 100)
density = np.array([mixture(value) for value in x])

plt.grid(True)
plt.plot(x, density,  linewidth=3, label="density calculated")
plt.hist(data.x, density=True, color="blue", bins=20, alpha = 0.5, label="data simulated")
plt.ylabel("density")
plt.xlabel("feature value (x)")
plt.title("Gaussian mixture obtained using Expectation-Maximization")
plt.legend()
plt.show()


