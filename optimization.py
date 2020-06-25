import numpy as np

import torch
import fast_svd
import gpytorch
from torch.autograd import Variable
from utils import *


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, device = "cpu"):
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(policy_net,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(policy_net, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew
    return False, x

def update_policy(args, get_loss, get_kl, policy_net, policy_optimizer = None, value_net = None, likelihood = None):
    grads = torch.autograd.grad(get_loss(), policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()
        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        return flat_grad_grad_kl + v * args.damping

    if args.pg_algorithm == "VanillaPG":
        if args.pg_estimator == 'BQ' and args.UAPG_flag:
            u_cov, s_cov, v_cov = fast_svd.pca_Cov(args, conjugate_gradients, Fvp, policy_net, value_net, likelihood,
                                        args.svd_low_rank, args.state_coefficient, raw=True, n_iter = 0)
            mod_s_cov = 1 - torch.sqrt(s_cov.min())/torch.sqrt(s_cov)
            loss_grad = loss_grad - torch.matmul(u_cov, torch.matmul(torch.diag(mod_s_cov), torch.matmul(v_cov, loss_grad)))
        policy_optimizer.zero_grad()
        set_flat_grad_to(policy_net, loss_grad)
        policy_optimizer.step()

    else:
        neg_stepdir = conjugate_gradients(Fvp, loss_grad, 50, device = args.device)
        if args.pg_estimator == 'BQ' and args.UAPG_flag:
            u_cov, s_cov, v_cov = fast_svd.pca_NPG_InvCov(args, conjugate_gradients, Fvp, policy_net, value_net, likelihood,
                                        args.svd_low_rank, args.state_coefficient, raw=True, n_iter = 0)
            mod_s_cov = torch.clamp(torch.sqrt(s_cov/s_cov.min()), 1, args.UAPG_epsilon) - 1
            neg_stepdir = neg_stepdir + torch.matmul(u_cov, torch.matmul(torch.diag(mod_s_cov), torch.matmul(v_cov, neg_stepdir)))
        
        if args.pg_algorithm == "NPG":
            policy_optimizer.zero_grad()
            set_flat_grad_to(policy_net, neg_stepdir)
            policy_optimizer.step()
        else:
            # pg_algorithm = "TRPO"
            stepdir = -neg_stepdir
            shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
            lm = torch.sqrt(shs / args.max_kl)
            fullstep = stepdir / lm[0]
            neggdotstepdir = (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
            prev_params = get_flat_params_from(policy_net)
            success, new_params = linesearch(policy_net, get_loss, prev_params, fullstep, neggdotstepdir/lm[0])
            set_flat_params_to(policy_net, new_params)

def update_params(args, batch, policy_net, value_net, policy_optimizer, likelihood, gp_mll, gp_value_optimizer, nn_value_optimizer):
    states = Variable(torch.Tensor(batch.state)).to(args.device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).to(args.device)
    action_means, action_log_stds, action_stds = policy_net(states)
    rewards = torch.Tensor(batch.reward).to(args.device)
    masks = torch.Tensor(batch.mask).to(args.device)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Estimating the Monte-Carlo/TD(1) returns and generalized advantage estimates (GAE)
    returns = torch.Tensor(actions.size(0),1).to(args.device)
    prev_return = 0
    if args.advantage_flag:
        state_values_estimates = value_net.nn_forward(Variable(states))
        deltas = torch.Tensor(actions.size(0),1).to(args.device)
        advantages = torch.Tensor(actions.size(0),1).to(args.device)
        prev_value = 0
        prev_advantage = 0
    
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        prev_return = returns[i, 0]
        if args.advantage_flag:
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - state_values_estimates.data[i]
            # GAE uses 'gamma' to trade estimation bias for lower variance.
            advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
            # For unbiased advantage estimates, replace GAE with this
            # advantages[i] = returns[i] - state_values_estimates[i]
            prev_value = state_values_estimates.data[i, 0]
            prev_advantage = advantages[i, 0]
    returns = Variable(returns).to(args.device)    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Optimizing the parameters of state value function V(s), i.e., neural net feature_extractor + value_head
    if args.advantage_flag:
        nn_value_optimizer.zero_grad()
        state_value_loss = (state_values_estimates - returns).pow(2).mean()
        state_value_loss.backward()
        nn_value_optimizer.step()
        advantages = Variable((advantages - advantages.mean()) / advantages.std()).to(args.device)
    targets = advantages if args.advantage_flag else returns
    if args.pg_estimator == 'BQ':
        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Optimizing the parameters of action value function Q(s,a), i.e., neural net feature_extractor + GP parameters
        U_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        u_fb_t,s_fb_t,v_fb_t = fast_svd.pca_U(U_prob.squeeze(-1), policy_net, args.svd_low_rank, raw=True, n_iter = 0)
        args.u_tens = Variable(u_fb_t).detach()
        args.s_tens = Variable(s_fb_t).detach()
        args.v_tens = Variable(v_fb_t).detach()

        args.v_tens = args.v_tens.transpose(1,0)     # we only need the v_tensor as K_f = U^T*(U*U^T)^{-1}*U = v*v^T, where U = u*s*v^T
        critic_GP_inputs = torch.cat([states,args.v_tens],1)
        args.critic_GP_inputs = critic_GP_inputs
        
        value_net.set_train_data(critic_GP_inputs, targets.squeeze(-1), strict = False)
        with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(50):
            gp_value_optimizer.zero_grad()
            fisher_multiplier = args.fisher_coefficient*args.v_tens.shape[0]
            values_ = value_net(critic_GP_inputs, state_multiplier = args.state_coefficient, fisher_multiplier = fisher_multiplier)
            value_loss = -gp_mll(values_, targets.squeeze(-1)).mean()
            value_loss.backward()
            gp_value_optimizer.step()
        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Optimizing the parameters of action value function Q(s,a), i.e., neural net feature_extractor + GP parameters
        with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(50), gpytorch.settings.fast_pred_var():
            values_mvn = likelihood(value_net(critic_GP_inputs, state_multiplier = args.state_coefficient, fisher_multiplier = fisher_multiplier))
            values = values_mvn.mean.unsqueeze(-1)
            alpha_t = values_mvn.lazy_covariance_matrix.inv_matmul(targets.squeeze(-1)).unsqueeze(-1)

    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(states)
        else:
            action_means, action_log_stds, action_stds = policy_net(states)
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_value_proxy = Variable(alpha_t.detach()) if args.pg_estimator == 'BQ' else targets
        if args.pg_algorithm == 'TRPO':
            action_loss = (-action_value_proxy * torch.exp(log_prob - Variable(fixed_log_prob))).mean()
        else:
            # The gradient of this expression is same as the previous, but the absence of exp makes it marginally faster to compute,
            # besides it is to distinguish VanillaPG and NPG from TRPO
            action_loss = (-action_value_proxy * log_prob).mean()
        return action_loss

    def get_kl():
        mean1, log_std1, std1 = policy_net(states)

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    update_policy(args, get_loss, get_kl, policy_net, policy_optimizer, value_net, likelihood)