import random


class AOS:
    def __init__(self, operators, segment_size=100, reaction_factor=0.1):
        self.operators = operators  # Dictionary of {"name": function}
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        
        # Scoring rewards
        self.sigma_1 = 10  # New global best
        self.sigma_2 = 5   # Better than current
        self.sigma_3 = 2   # Accepted (worse than current but passes threshold)

        # State tracking
        self.weights = {name: 1.0 for name in operators}
        self.scores = {name: 0.0 for name in operators}
        self.times_used = {name: 0 for name in operators}
        self.iterations_in_segment = 0

    def select_operator(self):
        """Spins the roulette wheel and returns the operator name and function."""
        total_weight = sum(self.weights.values())
        probabilities = [self.weights[name] / total_weight for name in self.operators]

        selected_name = random.choices(
            population=list(self.operators.keys()),
            weights=probabilities,
            k=1
        )[0]
        
        self.times_used[selected_name] += 1
        return selected_name, self.operators[selected_name]

    def update_score(self, operator_name, score_type):
        """Adds the appropriate sigma reward to the operator."""
        if score_type == "global_best":
            self.scores[operator_name] += self.sigma_1
        elif score_type == "better":
            self.scores[operator_name] += self.sigma_2
        elif score_type == "accepted":
            self.scores[operator_name] += self.sigma_3
        

    def end_of_iteration_update(self):
        """Checks if a segment is over and recalculates weights if necessary."""
        self.iterations_in_segment += 1
        
        if self.iterations_in_segment >= self.segment_size:
            for name in self.operators:
                if self.times_used[name] > 0:
                    # Update weight formula
                    self.weights[name] = (1 - self.reaction_factor) * self.weights[name] + \
                                         self.reaction_factor * (self.scores[name] / self.times_used[name])
                
                # Reset segment tracking
                self.scores[name] = 0.0
                self.times_used[name] = 0
                
            self.iterations_in_segment = 0