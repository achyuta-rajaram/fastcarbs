import torch, pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np
import fastcarbs_core as core

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Synthetic data
N, D = 40, 3
X = torch.randn(N, D)
true_kernel = gp.kernels.Matern32(input_dim=D, variance=torch.tensor(1.5), lengthscale=torch.tensor([0.7,1.2,0.9]))
K = true_kernel(X) + 0.05*torch.eye(N)
y = torch.distributions.MultivariateNormal(torch.zeros(N), covariance_matrix=K).sample()

# Pyro model (your snippet-style)
kernel = gp.kernels.Matern32(input_dim=D)
model = gp.models.GPRegression(X, y, kernel=kernel, jitter=1e-4).to(X)
model.noise = pyro.nn.PyroSample(dist.LogNormal(np.log(1e-2), 0.5))
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
gp.util.train(model, opt, num_steps=1000)  # same defaults
model.eval()
with torch.no_grad():
    m_py, v_py = model.forward(X, full_cov=False, noiseless=True)

# C++ model
k_cpp = core.Matern32ARD(D, 1.0, 1.0)
gpr = core.GPRegression(k_cpp, noise_var=1e-2, jitter=1e-4)
gpr.set_data(X.numpy(), y.numpy())
gpr.fit(num_steps=1000, lr=1e-4)
m_c, v_c = gpr.predict(X.numpy(), noiseless=True)
m_c = torch.from_numpy(np.array(m_c)); v_c = torch.from_numpy(np.array(v_c))

print("max |mean diff|:", (m_c - m_py).abs().max().item())
print("max |var  diff|:", (v_c - v_py).abs().max().item())
