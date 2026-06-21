import random
from collections import defaultdict

class QAgent:
    def __init__(self, operators, alpha=0.5, gamma=0.7, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_rate=0.99, eta_reward=0.6, C=1.0):
        """
        Initialize the Q-Learning Agent.
        :param operators: Dictionary of {"operator_name": function_call}
        :param alpha: Learning rate (0 to 1). Higher = overrides old info faster.
        :param gamma: Discount factor (0 to 1). Higher = values long-term rewards.
        :param epsilon_decay_rate: Exponential decay multiplier (Beta = 0.99).
        """
        self.operators = operators
        self.action_names = list(operators.keys())
        
        self.alpha = alpha
        self.gamma = gamma
        self.eta_reward = eta_reward
        self.C = C

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        
        # Initialize Q-table: default Q-value is 0.0 for all unknown states
        self.q_table = defaultdict(lambda: {name: 0.0 for name in self.action_names})
        self.usage_count = {name: 0 for name in self.action_names}
        self.total_reward = {name: 0.0 for name in self.action_names}

        # Track the last action and state to apply the Bellman update properly
        self.last_state = None
        self.last_action = None

    def decay_epsilon(self):
        """Call this at the end of every ALNS iteration"""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay_rate

    def select_operator(self, state):
        """
        Selects an operator using an epsilon-greedy policy based on the current state.
        Returns the operator's name and its function.
        """
        # 1. Exploration: Spin the roulette wheel entirely randomly
        if random.random() < self.epsilon:
            selected_name = random.choice(self.action_names)
            
        # 2. Exploitation: Pick the best action for this specific state
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            # If there are ties (e.g., at the very start when all are 0.0), pick randomly among the best
            best_actions = [act for act, q in q_values.items() if q == max_q]
            selected_name = random.choice(best_actions)
            
        # Save state and action for the upcoming update
        self.last_state = state
        self.last_action = selected_name
        self.usage_count[selected_name] += 1

        return selected_name, self.operators[selected_name]

    def update_q_value(self, reward, next_state):
        """
        Applies the Temporal Difference (Bellman) update to the Q-Table.
        Must be called at the end of every ALNS iteration.
        """
        if self.last_state is None or self.last_action is None:
            return

        current_q = self.q_table[self.last_state][self.last_action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.total_reward[self.last_action] += reward
        
        self.q_table[self.last_state][self.last_action] = new_q