import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # Multi-dimension Gaussian distribution for continuous action space
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound  = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        # softplus is a smooth approximation to the ReLU, which can constrain the output to be always positive
        std = F.softplus(self.fc_std(x))    # std_var need to be always positive
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # re-parameterization trick for back-propogation
        # log value of probability of this sample result
        log_prob = dist.log_prob(normal_sample)
        # scale to (-1, +1)
        action = torch.tanh(normal_sample)
        # Question?
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # re-scale to set value
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        # input both state and action (follow DDPG)
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device) -> None:
        # policy network
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # first Q network
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # second Q network
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # first target Q network
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # second target Q network
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        # set the same params
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # set optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # using alpha log value to stablize the training process
        # temperature adjustment
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # target_entropy
        self.target_entropy = target_entropy

        # other hyper-params
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        """
        calc target Q value
        """
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        # target Q to ease bootstrapping
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # pendulum env reward space transform
        rewards = (rewards + 8.0) / 8.0

        # update twin Q Network
        td_target = self.calc_target(rewards, next_states, dones)
        # mini-batch?
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach())
        )

        # clear old grad + calc new grad + grad descent
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # update policy network
        new_actions, log_prob = self.actor(states)
        entropy = - log_prob
        # No bootstrapping here so we just use Q not target Q
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha (temperature)
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)







    

    


