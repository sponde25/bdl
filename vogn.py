import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F

#TODO Refactor gradient scaling

################################
## PyTorch Optimizer for VOGN ##
################################
required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


class VOGN(Optimizer):
    """Implements the VOGN algorithm. It uses the Generalized Gauss Newton (GGN)
        approximation to the Hessian and a mean-field approximation. Note that this
        optimizer does **not** support multiple model parameter groups. All model
        parameters must use the same optimizer parameters.
        model (nn.Module): network model
        train_set_size (int): number of data points in the full training set
            (objective assumed to be on the form (1/M)*sum(-log p))
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running average of gradients and sum of squared gradient (default: 0.999)
        prior_mu (FloatTensor, optional): mu of prior distribution (posterior of previous task)
            (default: None)
        prior_prec (float or FloatTensor, optional): prior precision on parameters
            (default: 1.0)
        prec_init (float, optional): initial precision for variational dist. q
            (default: 1.0)
        num_samples (float, optional): number of MC samples
            (default: 1)
    """

    def __init__(self, model, train_set_size, lr=1e-3, betas=(0.9, 0.999), prior_mu=None, prior_prec=1.0,
                 initial_prec=1.0, num_samples=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_mu is not None and not torch.is_tensor(prior_mu):
            raise ValueError("Invalid prior mu value (from previous task): {}".format(prior_mu))
        if torch.is_tensor(prior_prec):
            if (prior_prec < 0.0).all():
                raise ValueError("Invalid prior precision tensor: {}".format(prior_prec))
        else:
            if prior_prec < 0.0:
                raise ValueError("Invalid prior precision value: {}".format(prior_prec))
        if torch.is_tensor(initial_prec):
            if (initial_prec < 0.0).all():
                raise ValueError("Invalid initial precision tensor: {}".format(initial_prec))
        else:
            if initial_prec < 0.0:
                raise ValueError("Invalid initial precision value: {}".format(initial_prec))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if train_set_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, betas=betas, prior_mu=prior_mu, prior_prec=prior_prec, initial_prec=initial_prec,
                        num_samples=num_samples, train_set_size=train_set_size)
        super(VOGN, self).__init__(model.parameters(), defaults)

        self.train_modules = []
        self.set_train_modules(model)
        for module in self.train_modules:
            module.register_forward_hook(update_input)

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        device = parameters[0].device

        p = parameters_to_vector(parameters)
        # mean parameter of variational distribution.
        self.state['mu'] = p.clone().detach()

        # mean parameter of prior distribution.
        if torch.is_tensor(defaults['prior_mu']):
            self.state['prior_mu'] = defaults['prior_mu'].to(device)
        else:
            self.state['prior_mu'] = torch.zeros_like(p, device=device)

        # covariance parameter of variational distribution -- saved as a diagonal precision matrix.
        if torch.is_tensor(defaults['initial_prec']):
            self.state['precision'] = defaults['initial_prec'].to(device)
        else:
            self.state['precision'] = torch.ones_like(p, device=device) * defaults['initial_prec']

        # covariance parameter of prior distribution -- saved as a diagonal precision matrix.
        if torch.is_tensor(defaults['prior_prec']):
            self.state['prior_prec'] = defaults['prior_prec'].to(device)
        else:
            self.state['prior_prec'] = torch.ones_like(p, device=device) * defaults['prior_prec']

        self.state['momentum'] = torch.zeros_like(p, device=device)

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """

        if closure is None:
            raise RuntimeError(
                'For now, VOGN only supports that the model/loss can be reevaluated inside the step function')

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        lr = self.param_groups[0]['lr']
        momentum_beta = defaults['betas'][0]
        beta = defaults['betas'][1]
        momentum = self.state['momentum']

        mu = self.state['mu']
        precision = self.state['precision']
        prior_mu = self.state['prior_mu']
        prior_prec = self.state['prior_prec']

        grad_hat = torch.zeros_like(mu)
        ggn_hat = torch.zeros_like(mu)

        loss_list = []
        pred_list = []
        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            # Get loss and predictions
            loss, preds = closure()
            pred_list.append(preds)

            lc = []
            # Store the linear combinations
            for module in self.train_modules:
                lc.append(module.output)

            linear_grad = torch.autograd.grad(loss, lc)
            loss_list.append(loss.detach())

            grad = []
            ggn = []
            for i, module in enumerate(self.train_modules):
                G = linear_grad[i]
                A = module.input.clone().detach()
                M = A.shape[0]
                G *= M
                G2 = torch.mul(G, G)

                if isinstance(module, nn.Linear):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij,ik->jk', G, A))
                    ggn.append(torch.einsum('ij, ik->jk', G2, A2))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.Conv2d):
                    A = F.unfold(A, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding,
                                 stride=module.stride)
                    A2 = torch.mul(A, A)
                    _, k, hw = A.shape
                    _, c, _, _ = G.shape
                    G = G.view(M, c, -1)
                    G2 = G2.view(M, c, -1)
                    grad.append(torch.einsum('ijl,ikl->jk', G, A))
                    ggn.append(torch.einsum('ijl,ikl->jk', G2, A2))
                    if module.bias is not None:
                        A = torch.ones((M, 1, hw), device=A.device)
                        grad.append(torch.einsum('ijl,ikl->jk', G, A))
                        ggn.append(torch.einsum('ijl,ikl->jk', G2, A))

                if isinstance(module, nn.BatchNorm1d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ij->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.BatchNorm2d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ijkl->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ijkl->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ijkl->j', G))
                        ggn.append(torch.einsum('ijkl->j', G2))

            grad = parameters_to_vector(grad).div(M).detach()
            ggn = parameters_to_vector(ggn).div(M).detach()

            grad_hat.add_(grad)
            ggn_hat.add_(ggn)

        # Convert the parameter gradient to a single vector.
        grad_hat = grad_hat.mul(defaults['train_set_size'] / defaults['num_samples'])
        ggn_hat.mul_(defaults['train_set_size'] / defaults['num_samples'])

        # Add momentum
        momentum.mul_(momentum_beta).add_((1 - momentum_beta), grad_hat)

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Update precision matrix
        precision.mul_(beta).add_((1 - beta), ggn_hat + prior_prec)
        # Update mean vector
        mu.addcdiv_(-lr, momentum + torch.mul(mu - prior_mu, prior_prec), precision)
        # Update model parameters
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])

        return loss, pred_list

    def get_distribution_params(self):
        """Returns current mean and precision of variational distribution
           (usually used to save parameters from current task as prior for next task).
        """
        mu = self.state['mu'].clone().detach()
        precision = self.state['precision'].clone().detach()

        return mu, precision

    def get_mc_predictions(self, forward_function, inputs, ret_numpy=False, raw_noises=None, *args, **kwargs):
        """Returns Monte Carlo predictions.
        Arguments:
            forward_function (callable): The forward function of the model
                that takes inputs and returns the outputs.
            inputs (FloatTensor): The inputs to the model.
            mc_samples (int): The number of Monte Carlo samples.
            ret_numpy (bool): If true, the returned list contains numpy arrays,
                otherwise it contains torch tensors.
        """

        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        predictions = []
        precision = self.state['precision']
        mu = self.state['mu']
        if raw_noises is None:
            raw_noises = [torch.zeros_like(mu)]
        for raw_noise in raw_noises:
            # Sample a parameter vector:
            # raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)

            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            # Call the forward computation function
            outputs = forward_function(inputs, *args, **kwargs)
            if ret_numpy:
                outputs = outputs.data.cpu().numpy()
            predictions.append(outputs)
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])
        return predictions

    def _kl_gaussian(self, p_mu, p_sigma, q_mu, q_sigma):
        log_std_diff = torch.sum(torch.log(p_sigma ** 2) - torch.log(q_sigma ** 2))
        mu_diff_term = torch.sum((q_sigma ** 2 + (p_mu - q_mu) ** 2) / p_sigma ** 2)
        const = list(q_mu.size())[0]
        
        return 0.5 * (mu_diff_term - const + log_std_diff)

    def kl_divergence(self):
        prec0 = self.state['prior_prec']
        prec = self.state['precision']
        mu = self.state['mu']
        sigma = 1. / torch.sqrt(prec)
        mu0 = self.state['prior_mu']
        if torch.is_tensor(prec0):
            sigma0 = 1. / torch.sqrt(prec0)
        else:
            sigma0 = 1. / math.sqrt(prec0)
        kl = self._kl_gaussian(p_mu=mu0, p_sigma=sigma0, q_mu=mu, q_sigma=sigma)
        
        return kl
    
