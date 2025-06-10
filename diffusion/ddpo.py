import torch
import guidance

class DDPO():
    
    def __init__(self, model, reward_fn, batch_size, ema_factor=0.99, warmup_size=200):
        """
        model: nn.Module with reverse_samples() function that can output log probs
        reward_fn: callable
        batch_size: int
        """
        self.model = model
        self.reward_fn = reward_fn
        self.batch_size = batch_size

        self.ema_factor = ema_factor
        self.warmup_size = warmup_size

        self.rew_buffer = None
        self.ema_mean = None
        self.ema_std = None
    
    def loss(self, x, cond):
        # Note: here x is only used because reverse sampling requires it for port positions
        # Don't need intermediates
        # log_prob is (T, B) tensor with gradients
        x_0, _, log_prob = self.model.reverse_samples(self.batch_size, x, cond, intermediate_every = 0, output_log_prob = True)
        x_0 = x_0.detach()
        reward = self.reward_fn(x_0, cond).detach() # (B)

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / self.ema_std
        
        trajectory_log_prob = log_prob.mean(dim=0)
        loss = -torch.mean(trajectory_log_prob * advantage) # loss = neg reward
        metrics = {
            "reward": reward.detach().mean().cpu().numpy(),
            "reward_ema_mean": self.ema_mean.cpu().numpy(),
            "reward_ema_std": self.ema_std.cpu().numpy(),
        }
        return loss, metrics
    
    def update_moving_averages(self, new_reward):
        # new reward has shape (B)
        if self.rew_buffer is None:
            self.rew_buffer = new_reward
        buff_len = self.rew_buffer.shape[0]
        if buff_len < self.warmup_size:
            self.rew_buffer = torch.cat((self.rew_buffer, new_reward), dim=0)
            self.ema_mean = self.rew_buffer.mean()
            self.ema_std = self.rew_buffer.std()
        else:
            self.ema_mean = self.ema_factor * self.ema_mean + (1-self.ema_factor) * new_reward.mean()
            self.ema_std = self.ema_factor * self.ema_std + (1-self.ema_factor) * new_reward.std()

def legality_reward(x, cond):
    return -guidance.legality_guidance_potential(x, cond) # better legality means better reward

def hpwl_reward(x, cond):
    return -guidance.hpwl_guidance_potential(x, cond) # lower hpwl means higher reward

def get_reward_fn(legality_weight, hpwl_weight):
    def reward_fn(x, cond):
        return legality_weight * legality_reward(x, cond) + hpwl_weight * hpwl_reward(x, cond)
    return reward_fn