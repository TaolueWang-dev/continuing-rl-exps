import numpy as np
from agents.base_agents import LFAControlAgent
from queue import PriorityQueue
import heapq
from utils.priorityqueue import PriorityQueueTop25Percent
def calculate_top_25_percent_average(data):
    data = sorted(data, key=lambda x: -x)
    length = len(data)
    top_25_percent_idx = max(1, int(length * 0.25))
    return sum(data[:top_25_percent_idx])/len(data[:top_25_percent_idx])

def get_top_25_element_avg(nums):
    n = len(nums)
    if n == 0:
        return 0
    k = max(1, int(len(nums) * 0.25))
    pq = PriorityQueue(k)
    for num in nums:
        if pq.full():
            if num > pq.queue[0]:
                pq.get()
                pq.put(num)
        else:
            pq.put(num)
    sum = 0
    for item in pq.queue:
        sum += item
    return sum/k




class DifferentialDiscountedQlearningAgent(LFAControlAgent):
    """
    Implements one-step discounted Q-learning with centering.
    """


    def __init__(self, **agent_args):
        super().__init__(**agent_args)

        # setting eta and gamma, the two main parameters of the algorithm
        self.eta = agent_args.get('eta', 0.1)
        self.gamma = agent_args.get('gamma', 0.95)
        self.pq =PriorityQueueTop25Percent
        self.reward_list = []

        # initializing the reward-rate estimate
        self.avg_reward_init = agent_args.get('avg_reward_init', 0)
        self.avg_reward = self.avg_reward_init

        self.pq = PriorityQueueTop25Percent()
        # initializing the step size for the reward-rate estimate
        self.beta_sequence = agent_args.get('beta_sequence', 'exponential')
        assert self.beta_sequence in ['unbiased_trick', 'exponential', '1byN']
        if self.robust_to_initialization:
            self.beta_sequence = 'unbiased_trick'
        self.beta_init = self.eta * self.alpha_init
        if self.tilecoder:
            self.beta_init *= self.num_tilings
        
        # using the TD vs conventional error for updating the reward rate
        self.conv_error = agent_args.get("conv_error", False)
        self.skip_exploratory_update = agent_args.get("skip_exploratory_update", False)
        # using the Sarsa target for computing the TD error
        self.sarsa_target = agent_args.get("sarsa_target", False)

    def _update_weights(self, reward, obs):
        past_sa = self._get_representation(self.past_obs, self.past_action)
        prediction = self._get_linear_value_approximation(past_sa)

        # compute the target
        next_action = None
        if self.sarsa_target:
            next_action = self._choose_action_egreedy(obs)      # ToDo: this breaks the exploratory_action logic
            q_next = self._get_value(obs, next_action)
        else:
            q_next = self._max_action_value(obs)
        # self.avg_reward = calculate_top_25_percent_average(self.reward_list)
        # self.reward_list.append(reward)
        self.pq.push(reward)
        # print(self.pq.pq_min.qsize())
        top25_reward_avg = self.pq.get_top25_avg()
        target = top25_reward_avg - self.avg_reward + self.gamma * q_next
        delta = target - prediction

        # update the reward-rate estimate
        if self.skip_exploratory_update and self.exploratory_action:
            updated_delta = delta
        else:
            avg_rew_delta = delta if not self.conv_error else (reward - self.avg_reward)
            if self.robust_to_initialization:
                old_avg_reward = self.avg_reward
                self.avg_reward += self.beta * avg_rew_delta
                updated_target = target + (old_avg_reward - self.avg_reward)
                updated_delta = updated_target - prediction
            else:
                self.avg_reward += self.beta * avg_rew_delta
                updated_delta = delta

        # update the value estimates
        if self.tilecoder:
            self.weights[past_sa] += self.alpha * updated_delta
        else:
            self.weights += self.alpha * updated_delta * past_sa

    def _initialize_step_size(self):
        """Initializes both the step sizes."""
        super()._initialize_step_size()
        if self.beta_sequence == 'unbiased_trick':
            self.o_b = self.beta_init if self.beta_init != 0 else 1
            self.beta = self.beta_init / self.o_b
        else:
            self.beta = self.beta_init

    def _update_step_size(self):
        """Updates both the step sizes per step."""
        super()._update_step_size()
        if self.beta_sequence == 'exponential':
            self.beta *= self.alpha_decay_rate
        elif self.beta_sequence == 'unbiased_trick':
            self.o_b = self.o_b + self.beta_init * (1 - self.o_b)
            self.beta = self.beta_init / self.o_b
        else:
            self.beta = self.beta_init / self.update_count if self.skip_exploratory_update else self.beta_init / self.timestep


class DifferentialDiscountedSarsaAgent(DifferentialDiscountedQlearningAgent):
    """
    Implements one-step discounted Sarsa with centering.
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.sarsa_target = True
        self.conv_error = True

if __name__ == '__main__':
    data = [10, 20, 15, 40, 50, 5, 25, 35]
    print(get_top_25_element_avg(data))