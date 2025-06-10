import torch

class CosineScheduler():
    def __init__(self, clip = False):
        self.clip = clip

    def add_noise(self, input, eps, t):
        # takes:
        # clean input (B, ...) 
        # eps (B, ...)
        # t (B,)
        # returns noised input at timestep t
        B, = t.shape
        input_dims = len(input.shape[1:])
        alpha = self.alpha(t).view((B, *([1] * input_dims)))
        sigma = self.sigma(t).view((B, *([1] * input_dims)))
        output = input * alpha + eps * sigma
        return output
    
    def set_timesteps(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(1-1e-4, 1e-4, num_timesteps+1)

    def step(
            self, 
            eps_prediction, 
            t, 
            t_minus_one, 
            xt, 
            z,
            mask = None, # where mask is true, predicted x0 does not change from sample
            guidance_fn = None,
            ):
        # eps_prediction: epsilon_hat (B, ...)
        # t: float
        # t_minus_one: float
        # xt: (B, ...)
        # guidance function: x_hat -> guidance
        alpha_t_minus_one = self.alpha(t_minus_one)
        alpha_t = self.alpha(t)
        sigma_t_minus_one = self.sigma(t_minus_one)
        sigma_t = self.sigma(t)
        eta = self.eta(t, t_minus_one)
        predicted_x0 = (xt - sigma_t * eps_prediction)/alpha_t

        # apply mask
        predicted_x0 = torch.where(mask, xt, predicted_x0) if mask is not None else predicted_x0

        if self.clip: # consider moving this 
            predicted_x0 = torch.clip(predicted_x0, min=-4, max=4)
        
        # compute guidance if necessary
        g = guidance_fn(predicted_x0) if guidance_fn is not None else 0
        x0_guided = predicted_x0 + g
        
        xt_minus_one_direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_prediction
        extra_noise = eta * z

        xt_minus_one = alpha_t_minus_one * (x0_guided) + xt_minus_one_direction + extra_noise
        # apply mask
        xt_minus_one = torch.where(mask, xt, xt_minus_one) if mask is not None else xt_minus_one

        return xt_minus_one, predicted_x0

    def eta(self, t, t_minus_one):
        a = self.sigma(t_minus_one) / self.sigma(t)
        b = torch.sqrt(1-torch.square(self.alpha(t)/self.alpha(t_minus_one)))
        return a * b

    def alpha(self, t):
        return torch.cos((torch.pi/2)*t)
    
    def sigma(self, t):
        return torch.sin((torch.pi/2)*t)