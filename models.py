import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import gpytorch
from gpytorch.kernels import RBFKernel, LinearKernel


class Policy(nn.Module):
    # Deep neural policy network
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class FeatureExtractor(nn.Module):
    # Common feature extractor shared between the state-value V(s) and action-value Q(s,a) function approximations
    def __init__(self, num_inputs, num_outputs):
        super(FeatureExtractor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 48)
        self.affine3 = nn.Linear(48, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        return x

class Value(gpytorch.models.ExactGP, nn.Module):
    # Monte-Carlo PG estimator :- Only State-value V(s) function approximation
    # Bayesian Quadrature PG estimator :- Both state-value V(s) and action-value Q(s,a) function approximation
    def __init__(self, NN_num_inputs, pg_estimator, fisher_num_inputs = None, gp_likelihood = None):
        # fisher_num_inputs is same as svd_low_rank, because of the linear approximation of the Fisher kernel through FastSVD.
        if pg_estimator == 'MC':
            nn.Module.__init__(self)
        else:
            gpytorch.models.ExactGP.__init__(self, None, None, gp_likelihood)
        self.NN_num_inputs = NN_num_inputs
        NN_num_outputs = 10

        self.feature_extractor = FeatureExtractor(NN_num_inputs, NN_num_outputs)
        # value_head is used for computing the state-value function approximation V(s) and subsequently GAE estimates
        self.value_head = nn.Linear(NN_num_outputs, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if pg_estimator == 'BQ':
            grid_size = 128
            # Like value_head, the following code constructs the GP head for action-value function approximation Q(s,a)
            # Note that both V(s) and Q(s,a) share the same feature extractor for the state-values "s".
            self.mean_module = gpytorch.means.ConstantMean()
            
            # First NN_num_outputs indices of GP's input correspond to the state_kernel
            state_kernel_active_dims = torch.tensor(list(range(NN_num_outputs)))
            # [NN_num_outputs, GP_input.shape[1]-1] indices of GP's input correspond to the fisher_kernel
            fisher_kernel_active_dims = torch.tensor(list(range(fisher_num_inputs+NN_num_outputs))[NN_num_outputs:])
            self.covar_module_2 = LinearKernel(active_dims=torch.tensor(list(range(fisher_num_inputs+NN_num_outputs))[NN_num_outputs:]))
            self.covar_module_1 = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.GridInterpolationKernel(
                RBFKernel(ard_num_dims=1), grid_size=grid_size, num_dims=1)),num_dims=NN_num_outputs, active_dims=torch.tensor(list(range(NN_num_outputs))))

    def nn_forward(self, x):
        # Invokes the value_head for computing the state value function V(s), subsequently used for computing GAE estimates
        extracted_features = self.feature_extractor(x[:,:self.NN_num_inputs])
        state_value_estimate = self.value_head(extracted_features)
        return state_value_estimate

    def forward(self, x, state_multiplier, fisher_multiplier, only_state_kernel=False):
        # Refer GPytorch documentation (https://docs.gpytorch.ai/) for better understanding
        extracted_features = self.feature_extractor(x[:,:self.NN_num_inputs])
        extracted_features = gpytorch.utils.grid.scale_to_bounds(extracted_features, -1, 1)

        if only_state_kernel:
            mean_x = self.mean_module(extracted_features)
            # Implicitly computes (c_1 K_s + sigma^2 I) which can be used for efficiently computing the MVM (c_1 K_s + sigma^2 I)^{-1}*v 
            covar_x = gpytorch.lazy.ConstantMulLazyTensor(self.covar_module_1(extracted_features), state_multiplier)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        GP_input = torch.cat([extracted_features, x[:,self.NN_num_inputs:]],1)
        mean_x = self.mean_module(GP_input)
        # Implicitly computes (c_1 K_s + c_2 K_f + sigma^2 I) which can be used for efficiently computing the MVM (c_1 K_s + c_2 K_f + sigma^2 I)^{-1}*v 
        covar_x = gpytorch.lazy.ConstantMulLazyTensor(self.covar_module_1(GP_input), state_multiplier) + \
                                gpytorch.lazy.ConstantMulLazyTensor(self.covar_module_2(GP_input), fisher_multiplier)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)