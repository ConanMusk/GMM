from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt

#------------------------------------------------------------------ 

def check_numerics(tensor, message='tensor contains NaN or Inf'):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(message + str(tensor))
    
def check_zeros(tensor, message='tensor contains zeros'):
    if torch.any((tensor == 0)):
        raise ValueError(message + str(tensor))

def norm_pdf(x, mu, sigma):
  retval = (1. / torch.sqrt(2 * 3.1415927410125732 * sigma**2)) * torch.exp(-((x - mu)**2) / (2 * sigma**2))
  check_numerics(retval)
  return retval

def log_norm_pdf(x, mu, sigma):
  check_zeros(sigma)

  log_pdf = -0.5 * (torch.log(2 * torch.tensor(3.1415927410125732)) + 2 * torch.log(sigma) + ((x - mu) / sigma)**2)

  check_numerics(log_pdf)
  return log_pdf

def log_sum_exp(values, dim=None):
    """Numerically stable implementation of the operation"""
    m = torch.max(values, dim=dim, keepdim=True)[0]
    values0 = values - m
    return m + torch.log(torch.sum(torch.exp(values0), dim=dim, keepdim=True))

def loss(data, weights, means, stds):
  """Compute the loss for a GMM as the negative of the log likelihood.

  Params:
    data: The actual data. Does not need to sum to one.
    component: A three dimensional numpy array, organized as (k, 3). Each row is a component, and each 
      column is a parameter. The first column is the weight, the second is the mean, and the third is the
      standard deviation."""
  
  if isinstance(weights, (int, float, complex)):
    weights = torch.tensor([weights])
  if isinstance(means, (int, float, complex)):
    means = torch.tensor([means])
  if isinstance(stds, (int, float, complex)):
    stds = torch.tensor([stds])
  
  N, K = data.size(0), len(weights)
  log_weighted_pdfs = torch.zeros(N, K)  # Initialize a 2D tensor to store the log-weighted PDF values
  
  for k in range(K):
      log_component_pdf = log_norm_pdf(data, means[k], stds[k])  # Log-probability of data under k-th Gaussian component
      log_weighted_pdfs[:, k] = log_component_pdf + torch.log(weights[k])  # Store log-weighted probability for each data point
  
  log_sum_per_data_point = log_sum_exp(log_weighted_pdfs, dim=1)  # Sum log-probabilities across components for each data point
  neg_log_likelihood = -torch.mean(log_sum_per_data_point)  # mean log-sums across all data points
  
  return neg_log_likelihood

def plot_gmm_histogram(data, weights, means, stds):
    # Create a grid of x values to evaluate the PDF
    x = torch.linspace(min(data), max(data), 1000)

    gmm_pdfs = torch.stack([w * norm_pdf(x, mu=m, sigma=s) for w, m, s in zip(weights, means, stds)], dim=0)

    # Plot histogram
    plt.hist(data, bins=50, density=True, alpha=0.5, color='g', label='Data Histogram')

    # Plot each GMM component
    for i, gmm_pdf in enumerate(gmm_pdfs, 1):
        plt.plot(x, gmm_pdf.detach().numpy(), label=f'GMM Component {i}')

    # Plot the sum of the GMM components
    plt.plot(x, gmm_pdfs.sum(0).detach().numpy(), label='GMM Sum', linestyle='dashed')

    # Print data histogram values
    counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0)
    # print(f'Histogram Counts: {counts}')
    # print(f'Bin Edges: {bins}')

    # Add legend and labels
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Histogram and GMM Components')

    # Show the plot
    plt.show()

class GMM(torch.nn.Module):
  def __init__(self, n_components):
    super(GMM, self).__init__()
    self.n_components=n_components
    self.weights_logit=torch.nn.Parameter(torch.zeros(n_components), requires_grad=True)
    self.means=torch.nn.Parameter(torch.zeros(n_components), requires_grad=True)
    self.stds_logit=torch.nn.Parameter(torch.zeros(n_components), requires_grad=True)

  def initialize(self, data):
    def linspace_median(start, stop, num, requires_grad=True):
      if num == 1:
          return torch.tensor([(start + stop) / 2], requires_grad=requires_grad)
      else:
          return torch.linspace(start, stop, num, requires_grad=requires_grad)
    
    assert data.dim() == 1, data.dim()
    mean = torch.mean(data)
    std = torch.std(data)

    weights_logit = torch.tensor([0.0] * self.n_components, requires_grad=True)
    centroids = linspace_median(mean -  std, mean + std, self.n_components, requires_grad=True)
    stds_logit = torch.tensor([torch.log(std / self.n_components)] * self.n_components, requires_grad=True)

    with torch.no_grad():  # Disable gradient tracking while updating the parameter
        self.weights_logit.data.copy_(weights_logit)
        self.means.data.copy_(centroids)
        self.stds_logit.data.copy_(stds_logit)

  def forward(self):
    weights = torch.nn.functional.softmax(self.weights_logit)
    means = self.means
    stds = torch.exp(self.stds_logit)

    return weights, means, stds


def fit_backprop(model, data, epochs=1000):
  optimizer = torch.optim.AdamW([model.weights_logit, model.means, model.stds_logit], lr=.1)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=False)

  print('data mean/std', torch.mean(data), torch.std(data))
  print('loss on uninitialized model', loss(data, *model()))

  with tqdm(range(epochs), 'fitting gmm') as pbar:
    previous_loss_val = torch.tensor(float('inf'))
    for epoch in pbar:
      optimizer.zero_grad()
      weights, means, stds = model()
      loss_val = loss(data, weights, means, stds)
      loss_val.backward()
      torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) 
      optimizer.step()
      scheduler.step(loss_val)     

      pbar.set_postfix({'Epoch': epoch, 'loss': loss_val})

      if previous_loss_val <= loss_val:
        print('loss not decreasing, stopping')
        break
      else:
        previous_loss_val = loss_val


def fit_em(model, data, epochs=1000):
  with tqdm(range(epochs), 'fitting gmm') as pbar:
    previous_loss_val = torch.tensor(float('inf'))
    for epoch in range(epochs):
      # E-step: expectation.
      log_gamma = torch.zeros((data.size(0), model.n_components))  
      new_weights_log = torch.zeros_like(model.weights_logit)
      new_means = torch.zeros_like(model.means)
      new_stds_log = torch.zeros_like(model.stds_logit)

      for i, (w, m, s) in enumerate(zip(*model())):
          log_gamma[:, i] = torch.log(w) + log_norm_pdf(data, mu=m, sigma=s)  

      # Normalize to add up to one: e.g. 0.2 and 0.8.
      log_gamma -= log_sum_exp(log_gamma, dim=1)

      # M-step: Maximization. update the estimates for mean, stds, and weights based on the new gammas
      for i in range(model.n_components):
          component_effective_points = log_gamma[:, i].exp().sum()  # Effective number of points assigned to this component.

          new_weights_log[i] = torch.log(component_effective_points) - torch.log(torch.tensor(data.size(0)))  # update weight. 
          new_means[i] = (log_gamma[:, i].exp() @ data) / component_effective_points  # new mean = weighted average of points.
          var = (log_gamma[:, i].exp() @ (data - new_means[i])**2) / component_effective_points  # update variance
          new_stds_log[i] = torch.log(torch.sqrt(var))  # update std in log space

      # Update the model parameters
      model.weights_logit.data.copy_(new_weights_log)
      model.means.data.copy_(new_means)
      model.stds_logit.data.copy_(new_stds_log)
      
      # Compute log-likelihood to check for convergence (optional)
      loss_val = loss(data, *model())

      pbar.set_postfix({'Epoch': epoch, 'loss': loss_val})

      if previous_loss_val <= loss_val:
        print('loss increased, stopping')
        break
      else:
        previous_loss_val = loss_val


def test_backprop(n_components, data):
  gmm = GMM(n_components)
  gmm.initialize(data)

  weights, means, stds = gmm()
  print('components after init, before fitting', weights.detach().numpy(), means.detach().numpy(), stds.detach().numpy())

  fit_backprop(gmm, torch.tensor(data))

  weights, means, stds = gmm()
  print('components after fitting', weights.detach().numpy(), means.detach().numpy(), stds.detach().numpy())

  plot_gmm_histogram(data, weights, means, stds)

def test_em(n_components, data):
  gmm = GMM(n_components)
  gmm.initialize(data)

  weights, means, stds = gmm()
  print('components after init, before fitting', weights.detach().numpy(), means.detach().numpy(), stds.detach().numpy())

  fit_em(gmm, torch.tensor(data))

  weights, means, stds = gmm()
  print('components after fitting', weights.detach().numpy(), means.detach().numpy(), stds.detach().numpy())

  plot_gmm_histogram(data, weights, means, stds)


data = torch.randn(65536)

# 设置图像尺寸
image_size = (256, 256)

# 定义Gamma分布的shape和scale参数
shape_param = torch.tensor([10.0])  # 形状参数
scale_param = torch.tensor([10.0])  # 比例参数

# 使用Gamma分布生成噪声
gamma_dist = torch.distributions.Gamma(shape_param, scale_param)
gamma_noise = gamma_dist.sample(image_size)
data = gamma_noise.flatten()
test_em(4, data)

