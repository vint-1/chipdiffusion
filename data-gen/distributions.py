import torch
import torch.distributions as dist

def get_distribution(dist_type, dist_params):
    if dist_type == "uniform":
        return dist.Uniform(**dist_params)
    if dist_type == "normal":
        return dist.Normal(**dist_params)
    if dist_type == "log_uniform":
        return LogUniform(**dist_params)
    elif dist_type == "bernoulli":
        return dist.Bernoulli(**dist_params)
    elif dist_type == "cond_poisson":
        return ConditionalPoisson(**dist_params)
    elif dist_type == "cond_exp_bernoulli":
        return ConditionalExpBernoulli(**dist_params)
    elif dist_type == "cond_linear_bernoulli":
        return ConditionalLinearBernoulli(**dist_params)
    elif dist_type == "cond_exp_hierarchical_bernoulli":
        return ConditionalExpHierarchicalBernoulli(**dist_params)
    elif dist_type == "cond_sigmoid_hierarchical_bernoulli":
        return ConditionalSigmoidHierarchicalBernoulli(**dist_params)
    elif dist_type == "cond_thresh_bernoulli":
        return ConditionalThresholdBernoulli(**dist_params)
    elif dist_type == "cond_power_bernoulli":
        return ConditionalPowerBernoulli(**dist_params)
    elif dist_type == "clipped_exp":
        return ClippedExp(**dist_params)
    elif dist_type == "cond_binomial":
        return ConditionalBinomial(**dist_params)
    elif dist_type == "hierarchical_normal":
        return HierarchicalNormal(**dist_params)

class LogUniform:
    # uniformly sample in log space
    def __init__(self, low, high):
        self.dist = dist.Uniform(low=torch.log(torch.tensor(low)), high=torch.log(torch.tensor(high)))
    
    def sample(self, *args, **kwargs):
        sample = self.dist.sample(*args, **kwargs)
        return torch.exp(sample)

class ConditionalPoisson:
    def __init__(self, scale):
        self.scale = scale
    
    def sample(self, cond, sample_shape = None):
        rate = self.scale * cond
        distribution = dist.Poisson(rate)
        sample = distribution.sample(sample_shape)
        return sample
    
class ConditionalExpBernoulli:
    def __init__(self, scale, prob_clip=0.5, prob_multiplier=0.1):
        # apply clip before multiplier
        self.scale = scale
        self.prob_clip = prob_clip
        self.prob_multiplier = prob_multiplier
    
    def sample(self, cond, sample_shape = torch.Size([])):
        rate = cond / self.scale # positive number
        # prob = self.prob_multiplier * torch.clip(torch.exp(-rate), max=self.prob_clip) old version
        prob = torch.clip(self.prob_multiplier * torch.exp(-rate), max=self.prob_clip)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample

class ConditionalLinearBernoulli:
    def __init__(self, scale, prob_clip=0.5, prob_multiplier=0.1):
        # apply clip before multiplier
        self.scale = scale
        self.prob_clip = prob_clip
        self.prob_multiplier = prob_multiplier
    
    def sample(self, cond, sample_shape = torch.Size([])):
        rate = self.prob_multiplier * (1-(cond / self.scale)) # positive number
        prob = torch.clip(rate, max=self.prob_clip, min=0)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample
    
class ConditionalThresholdBernoulli:
    def __init__(self, threshold, p_1, p_2):
        # apply clip before multiplier
        self.threshold = threshold
        self.p_1 = p_1
        self.p_2 = p_2
    
    def sample(self, cond, sample_shape = torch.Size([])):
        prob = torch.where(cond > self.threshold, self.p_2, self.p_1)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample
    
class ConditionalExpHierarchicalBernoulli:
    def __init__(
            self, 
            scale_dist, 
            prob_clip=0.5, 
            prob_multiplier_factor=0.00605, 
            prob_multiplier_exp=2.1, 
            global_scale=True, 
            scale_axes=2,
            scale_cap_dist=None,
            ):
        # apply clip after multiplier
        # global scale determines whether scale factor is the same for all samples (true), or if individually iid sampled (false)
        # scale_axes=1 for fixed scale per vertex, 2 for fixed scale per source pin 
        if scale_cap_dist is None:
            self.scale_dist = get_distribution(**scale_dist)
            self.scale_cap_dist = None
        else:
            self.scale_dist_params = scale_dist
            self.scale_cap_dist = get_distribution(**scale_cap_dist)
        self.prob_clip = prob_clip
        self.prob_multiplier_factor = prob_multiplier_factor
        self.prob_multiplier_exp = prob_multiplier_exp
        self.global_scale = global_scale
        self.scale_axes = scale_axes
    
    def sample(self, cond, sample_shape = torch.Size([])):
        # sample scale cap if necessary
        if self.scale_cap_dist is None:
            scale_dist = self.scale_dist
        else:
            scale_cap = self.scale_cap_dist.sample().item()
            # update the scale_dist params high with sampled cap
            self.scale_dist_params.dist_params.high = scale_cap
            scale_dist = get_distribution(**self.scale_dist_params)
        # sample scale
        if self.global_scale:
            scale = scale_dist.sample() 
        else:
            scale = scale_dist.sample(cond.shape[:self.scale_axes]) # (V) or (V, T)
            scale = scale.view(*cond.shape[:self.scale_axes], *([1]*len(cond.shape[self.scale_axes:])))
        prob_multiplier = self.prob_multiplier_factor * (scale ** self.prob_multiplier_exp)
        rate = cond / scale # positive number
        # prob = self.prob_multiplier * torch.clip(torch.exp(-rate), max=self.prob_clip) old version
        prob = torch.clip(prob_multiplier * torch.exp(-rate), max=self.prob_clip)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample
    
class ConditionalSigmoidHierarchicalBernoulli:
    def __init__(
            self, 
            scale_dist,
            sigma_dist, 
            prob_clip=0.5, 
            prob_multiplier_factor=0.00605, 
            prob_multiplier_exp=2.1, 
            global_scale=True, 
            scale_axes=2,
            scale_cap_dist=None,
            ):
        # apply clip after multiplier
        # global scale determines whether scale factor is the same for all samples (true), or if individually iid sampled (false)
        # scale_axes=1 for fixed scale per vertex, 2 for fixed scale per source pin 
        self.sigma_dist = get_distribution(**sigma_dist)
        if scale_cap_dist is None:
            self.scale_dist = get_distribution(**scale_dist)
            self.scale_cap_dist = None
        else:
            self.scale_dist_params = scale_dist 
            self.scale_cap_dist = get_distribution(**scale_cap_dist)
        self.prob_clip = prob_clip
        self.prob_multiplier_factor = prob_multiplier_factor
        self.prob_multiplier_exp = prob_multiplier_exp
        self.global_scale = global_scale
        self.scale_axes = scale_axes
    
    def sample(self, cond, sample_shape = torch.Size([])):
        # sample scale cap if necessary
        if self.scale_cap_dist is None:
            scale_dist = self.scale_dist
        else:
            scale_cap = self.scale_cap_dist.sample().item()
            # update the scale_dist params high with sampled cap
            self.scale_dist_params.dist_params.high = scale_cap
            scale_dist = get_distribution(**self.scale_dist_params)
        # sample scale
        if self.global_scale:
            scale = scale_dist.sample()
        else:
            scale = scale_dist.sample(cond.shape[:self.scale_axes]) # (V) or (V, T)
            scale = scale.view(*cond.shape[:self.scale_axes], *([1]*len(cond.shape[self.scale_axes:])))
        # sample global temperature sigma
        sigma = self.sigma_dist.sample()
        
        # scaled logistic: multiplier/(1+torch.exp((cond-scale)/sigma))
        prob_multiplier = self.prob_multiplier_factor * (scale ** self.prob_multiplier_exp)
        rate = (cond-scale)/sigma # positive number
        prob = torch.clip(prob_multiplier / (1+torch.exp(rate)), max=self.prob_clip)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample

class ConditionalPowerBernoulli:
    def __init__(self, power, prob_clip=0.5, prob_multiplier=0.1):
        # apply clip before multiplier
        assert power >= 0, "power should be >=0"
        self.power = power
        self.prob_clip = prob_clip
        self.prob_multiplier = prob_multiplier
    
    def sample(self, cond, sample_shape = torch.Size([])):
        prob = torch.clip(self.prob_multiplier * torch.pow(cond, -self.power), max=self.prob_clip)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample

class ConditionalBinomial:
    # p is constant
    # number of trials is power-law function of instance area (A), p is typically [0.5, 0.8]
    # n = clip(64 * A^p, min=4)
    def __init__(self, binom_p, binom_min_n, t, p):
        # apply clip before multiplier
        self.binom_p = binom_p
        self.binom_min_n = binom_min_n
        self.t = t
        self.p = p
    
    def sample(self, cond):
        n = torch.clip(torch.ceil(self.t * torch.pow(cond, self.p)), min=self.binom_min_n)
        distribution = dist.Binomial(n, torch.full(cond.shape, self.binom_p))
        sample = distribution.sample().int()
        return sample

class ClippedExp:
    def __init__(self, scale, clip_min, clip_max):
        self.dist = dist.Exponential(rate = 1/scale)
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def sample(self, sample_shape = torch.Size([])):
        sample = self.dist.sample(sample_shape)
        return torch.clip(sample, self.clip_min, self.clip_max)
    
class HierarchicalNormal:
    def __init__(
            self, 
            loc_dist,
            scale_dist,
            ):
        self.loc_dist = get_distribution(**loc_dist)
        self.scale_dist = get_distribution(**scale_dist)
    
    def sample(self, sample_shape = torch.Size([])):
        # sample scale cap if necessary
        loc = self.loc_dist.sample().item()
        scale = self.scale_dist.sample().item()
        sample = dist.Normal(loc=loc, scale=scale).sample(sample_shape)
        return sample