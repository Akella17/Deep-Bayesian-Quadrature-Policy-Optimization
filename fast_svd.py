from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import unittest
import math
import numpy as np
import torch
import gpytorch
from torch.autograd import Variable
from utils import *


def pca_U(args, A, model):
    # Computes the low-rank SVD for U matrix
    k = args.svd_low_rank
    l = k + 2
    n = A.shape[0]
    m = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert k > 0
    assert k <= min(m, n)
    assert l >= k

    R = 2 * (torch.rand(l, m).cuda() - 0.5)

    Q = []
    tem_2 = torch.rand_like(A, requires_grad=True).cuda()
    grads = torch.autograd.grad((A * tem_2).sum(0),
                                model.parameters(),
                                create_graph=True)
    right_grad_torch = torch.cat([grad.view(-1)
                                  for grad in grads]).unsqueeze(-1)
    for row_vectors in R:
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=row_vectors.unsqueeze(-1),
                                    create_graph=True)
        Q.append(torch.cat([grad.view(-1) for grad in grads]).unsqueeze(0))
    Q = torch.cat(Q).transpose(1, 0)
    (Q, _) = torch.qr(Q)

    final_prod = []
    for col_vectors in Q.transpose(1, 0):
        grads = torch.autograd.grad(
            (A * Variable(col_vectors).detach()).sum(0),
            model.parameters(),
            retain_graph=True)
        final_prod.append(
            torch.cat([grad.view(-1) for grad in grads]).unsqueeze(0))
    final_prod = torch.cat(final_prod).transpose(1, 0)
    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))
    return U[:, :k], s[:k], Va[:k, :]


def pca_Cov(args, conjugate_gradients, Fvp, policy_net, value_net, likelihood):
    # Computes the low-rank SVD of the vanilla policy gradient's covariance matrix, using approximate vector products with the low-rank approximation of U matrix.
    k = args.svd_low_rank
    l = k + 2
    m = args.u_tens.shape[0]
    n = m
    assert k > 0
    assert k <= min(m, n)
    assert l >= k

    R = 2 * (torch.rand(l, m).cuda() - 0.5)
    first_term = []
    for row_vectors in R:
        first_term.append(Fvp(row_vectors))
    first_term = torch.stack(first_term).transpose(1, 0)
    second_term_0 = torch.matmul(
        args.v_tens,
        torch.matmul(
            torch.diag(args.s_tens),
            torch.matmul(args.u_tens.transpose(1, 0), R.transpose(1, 0))))
    with gpytorch.settings.max_cg_iterations(
            1000), gpytorch.settings.max_preconditioner_size(
                50), gpytorch.settings.fast_pred_var():
        fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
        values_mvn = value_net(args.GP_inputs,
                               state_multiplier=args.state_coefficient,
                               fisher_multiplier=fisher_multiplier)
        second_term_1 = likelihood(
            values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
    second_term = torch.matmul(
        args.u_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.v_tens.transpose(1, 0), second_term_1)))
    Q = first_term - args.fisher_coefficient * second_term
    del second_term, second_term_0, second_term_1, first_term, values_mvn
    with torch.no_grad():
        (Q, _) = torch.qr(Q)

    first_term = []
    for row_vectors in Q.transpose(1, 0):
        first_term.append(Fvp(row_vectors))
    first_term = torch.stack(first_term).transpose(1, 0)
    second_term_0 = torch.matmul(
        args.v_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.u_tens.transpose(1, 0), Q))).squeeze(-1)
    with gpytorch.settings.max_cg_iterations(
            1000), gpytorch.settings.max_preconditioner_size(
                50), gpytorch.settings.fast_pred_var():
        fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
        values_mvn = value_net(args.GP_inputs,
                               state_multiplier=args.state_coefficient,
                               fisher_multiplier=fisher_multiplier)
        second_term_1 = likelihood(
            values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
    second_term = torch.matmul(
        args.u_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.v_tens.transpose(1, 0), second_term_1)))
    final_prod = first_term - args.fisher_coefficient * second_term

    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))

    del Ra, final_prod, second_term, second_term_0, second_term_1, first_term, values_mvn
    return U[:, :k], s[:k], Va[:k, :]


def pca_Cov_Exact(args, conjugate_gradients, Fvp, policy_net, value_net,
                  likelihood):
    # Computes the low-rank SVD of the vanilla policy gradient's covariance matrix, using exact vector products by relying on autograd.
    # SLOWER than pca_Cov which computes low-rank SVD of Cov
    k = args.svd_low_rank
    l = k + 2
    m = args.u_tens.shape[0]
    n = m
    assert k > 0
    assert k <= min(m, n)
    assert l >= k

    R = 2 * (torch.rand(l, m).cuda() - 0.5)
    action_means, action_log_stds, action_stds = policy_net(
        Variable(torch.Tensor(args.batch.state)).cuda())
    actions = torch.Tensor(np.concatenate(args.batch.action, 0)).cuda()
    U_prob = normal_log_density(Variable(actions), action_means,
                                action_log_stds, action_stds).squeeze(-1)
    Q = []
    tem_2 = torch.rand_like(U_prob, requires_grad=True).cuda()
    grads = torch.autograd.grad((U_prob * tem_2).sum(0),
                                policy_net.parameters(),
                                create_graph=True)
    right_grad_torch = torch.cat([grad.view(-1) for grad in grads])
    for row_vectors in R:
        first_term = Fvp(row_vectors)
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=row_vectors,
                                    retain_graph=True)
        second_term_0 = torch.cat([grad.view(-1) for grad in grads])
        with gpytorch.settings.max_cg_iterations(
                1000), gpytorch.settings.max_preconditioner_size(
                    50), gpytorch.settings.fast_pred_var():
            fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
            values_mvn = value_net(args.GP_inputs,
                                   state_multiplier=args.state_coefficient,
                                   fisher_multiplier=fisher_multiplier)
            second_term_1 = likelihood(
                values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
        grads = torch.autograd.grad(
            (U_prob * Variable(second_term_1.detach())).sum(0),
            policy_net.parameters(),
            retain_graph=True)
        second_term = torch.cat([grad.view(-1) for grad in grads])
        Q.append(first_term - args.fisher_coefficient * second_term)

    Q = torch.stack(Q).transpose(1, 0)
    with torch.no_grad():
        (Q, _) = torch.qr(Q)

    final_prod = []
    for row_vectors in Q.transpose(1, 0):
        first_term = Fvp(row_vectors)
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=row_vectors,
                                    create_graph=True)
        second_term_0 = torch.cat([grad.view(-1) for grad in grads])
        with gpytorch.settings.max_cg_iterations(
                1000), gpytorch.settings.max_preconditioner_size(
                    50), gpytorch.settings.fast_pred_var():
            fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
            values_mvn = value_net(args.GP_inputs,
                                   state_multiplier=args.state_coefficient,
                                   fisher_multiplier=fisher_multiplier)
            second_term_1 = likelihood(
                values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
        grads = torch.autograd.grad(
            (U_prob * Variable(second_term_1.detach())).sum(0),
            policy_net.parameters(),
            retain_graph=True)
        second_term = torch.cat([grad.view(-1) for grad in grads])
        final_prod.append(first_term - args.fisher_coefficient * second_term)

    final_prod = torch.stack(final_prod).transpose(1, 0)
    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))
    return U[:, :k], s[:k], Va[:k, :]


def pca_InvCov(args, conjugate_gradients, Fvp, policy_net, value_net,
               likelihood):
    # Computes the low-rank SVD of the inverse of vanilla policy gradient's covariance matrix
    k = args.svd_low_rank
    l = k + 2
    m = args.u_tens.shape[0]
    n = m
    assert k > 0
    assert k <= min(m, n)
    assert l >= k

    R = 2 * (torch.rand(l, m).cuda() - 0.5)
    action_means, action_log_stds, action_stds = policy_net(
        Variable(torch.Tensor(args.batch.state)).cuda())
    actions = torch.Tensor(np.concatenate(args.batch.action, 0)).cuda()
    U_prob = normal_log_density(Variable(actions), action_means,
                                action_log_stds, action_stds).squeeze(-1)

    Q = []
    tem_2 = torch.rand_like(U_prob, requires_grad=True).cuda()
    grads = torch.autograd.grad((U_prob * tem_2).sum(0),
                                policy_net.parameters(),
                                create_graph=True)
    right_grad_torch = torch.cat([grad.view(-1) for grad in grads])
    for row_vectors in R:
        first_term = conjugate_gradients(Fvp,
                                         row_vectors,
                                         50,
                                         device=args.device)
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=first_term,
                                    retain_graph=True)
        second_term_0 = torch.cat([grad.view(-1) for grad in grads])
        with gpytorch.settings.max_cg_iterations(
                1000), gpytorch.settings.max_preconditioner_size(
                    50), gpytorch.settings.fast_pred_var():
            fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
            values_mvn = value_net(args.GP_inputs,
                                   state_multiplier=args.state_coefficient,
                                   fisher_multiplier=None,
                                   only_state_kernel=True)
            second_term_1 = likelihood(
                values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
        grads = torch.autograd.grad(
            (U_prob * Variable(second_term_1.detach())).sum(0),
            policy_net.parameters(),
            retain_graph=True)
        second_term = torch.cat([grad.view(-1) for grad in grads])
        Q.append(first_term + args.fisher_coefficient *
                 conjugate_gradients(Fvp, second_term, 50, device=args.device))
    Q = torch.stack(Q).transpose(1, 0)
    (Q, _) = torch.qr(Q)

    final_prod = []
    for row_vectors in Q.transpose(1, 0):
        first_term = conjugate_gradients(Fvp,
                                         row_vectors,
                                         50,
                                         device=args.device)
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=first_term,
                                    retain_graph=True)
        second_term_0 = torch.cat([grad.view(-1) for grad in grads])
        with gpytorch.settings.max_cg_iterations(
                1000), gpytorch.settings.max_preconditioner_size(
                    50), gpytorch.settings.fast_pred_var():
            fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
            values_mvn = value_net(args.GP_inputs,
                                   state_multiplier=args.state_coefficient,
                                   fisher_multiplier=None,
                                   only_state_kernel=True)
            second_term_1 = likelihood(
                values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
        grads = torch.autograd.grad(
            (U_prob * Variable(second_term_1.detach())).sum(0),
            policy_net.parameters(),
            retain_graph=True)
        second_term = torch.cat([grad.view(-1) for grad in grads])
        final_prod.append(
            first_term + args.fisher_coefficient *
            conjugate_gradients(Fvp, second_term, 50, device=args.device))

    final_prod = torch.stack(final_prod).transpose(1, 0)
    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))
    return U[:, :k], s[:k], Va[:k, :]


def pca_NPG_InvCov(args, conjugate_gradients, Fvp, policy_net, value_net,
                   likelihood):
    # Computes the low-rank SVD of the natural policy gradient's covariance matrix
    k = args.svd_low_rank
    l = k + 2
    m = args.u_tens.shape[0]
    n = m
    assert k > 0
    assert k <= min(m, n)
    assert l >= k

    R = 2 * (torch.rand(l, m).cuda() - 0.5)
    first_term = []
    for row_vectors in R:
        first_term.append(Fvp(row_vectors))
    first_term = torch.stack(first_term).transpose(1, 0)
    second_term_0 = torch.matmul(
        args.v_tens,
        torch.matmul(
            torch.diag(args.s_tens),
            torch.matmul(args.u_tens.transpose(1, 0), R.transpose(1, 0))))
    with gpytorch.settings.max_cg_iterations(
            1000), gpytorch.settings.max_preconditioner_size(
                50), gpytorch.settings.fast_pred_var():
        fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
        values_mvn = value_net(args.GP_inputs,
                               state_multiplier=args.state_coefficient,
                               fisher_multiplier=None,
                               only_state_kernel=True)
        second_term_1 = likelihood(
            values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
    second_term = torch.matmul(
        args.u_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.v_tens.transpose(1, 0), second_term_1)))
    Q = first_term + args.fisher_coefficient * second_term
    del second_term, second_term_0, second_term_1, first_term, values_mvn
    with torch.no_grad():
        (Q, _) = torch.qr(Q)

    first_term = []
    for row_vectors in Q.transpose(1, 0):
        first_term.append(Fvp(row_vectors))
    first_term = torch.stack(first_term).transpose(1, 0)
    second_term_0 = torch.matmul(
        args.v_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.u_tens.transpose(1, 0), Q))).squeeze(-1)
    with gpytorch.settings.max_cg_iterations(
            1000), gpytorch.settings.max_preconditioner_size(
                50), gpytorch.settings.fast_pred_var():
        fisher_multiplier = args.fisher_coefficient * args.v_tens.shape[0]
        values_mvn = value_net(args.GP_inputs,
                               state_multiplier=args.state_coefficient,
                               fisher_multiplier=None,
                               only_state_kernel=True)
        second_term_1 = likelihood(
            values_mvn).lazy_covariance_matrix.inv_matmul(second_term_0)
    second_term = torch.matmul(
        args.u_tens,
        torch.matmul(torch.diag(args.s_tens),
                     torch.matmul(args.v_tens.transpose(1, 0), second_term_1)))
    final_prod = first_term + args.fisher_coefficient * second_term

    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))

    del Ra, final_prod, second_term, second_term_0, second_term_1, first_term, values_mvn
    return U[:, :k], s[:k], Va[:k, :]


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
