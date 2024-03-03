import numpy as np
import hashlib
from model import Model

SCORES = {
    ('C', 'C'): (1, 1),
    ('C', 'D'): (3, 0),
    ('D', 'C'): (0, 3),
    ('D', 'D'): (2, 2)
}

class CASim(Model):
    def __init__(self):
        super().__init__()  # Ensuring proper initialization of the parent class.

        self.t = 0
        self.rule_set = []
        self.config = None

        self.population = []  # Population of strategies
        self.fitness_scores = []  # Fitness scores for each strategy

        # Setting parameters directly without make_param for clarity in this standalone snippet
        self.population_size = 100
        self.mutation_rate = 0.02
        self.chromosome_length = 8  # Adjusted to match binary string length from your original population initialization
        self.crossover_rate = 0.7

        # Directly initializing the population here
        self.population = self.initialize_population()
    def initialize_population(self):
        # Initialize your population of strategies here
        return [''.join(np.random.choice(['0', '1'])
                        for _ in range(self.chromosome_length))
                for _ in range(self.population_size)]

    def get_move(self, strategy, opponent_history):
        if strategy == 'always_cooperate':
            return 'C'
        elif strategy == 'always_defect':
            return 'D'
        elif strategy == 'tit_for_tat':
            return opponent_history[-1]
        else:
            # For binary-encoded strategies, use the last 3 moves to decide
            if len(opponent_history) >= 3:
                idx = int(''.join('1' if m == 'D' else '0' for m in opponent_history[-3:]), 2)
            else:
                idx = 0  # Default to cooperation if not enough history
            return 'D' if strategy[idx] == '1' else 'C'


    def play_game(self, strategy1, strategy2, rounds=10):
        moves1, moves2 = ['C'], ['C']  # Initialize with cooperation
        for _ in range(rounds):
            move1 = self.get_move(strategy1, moves2)
            move2 = self.get_move(strategy2, moves1)

            moves1.append(move1)
            moves2.append(move2)

        return sum(SCORES[(m1, m2)][0] for m1, m2 in zip(moves1[1:], moves2[1:])), \
            sum(SCORES[(m1, m2)][1] for m1, m2 in zip(moves1[1:], moves2[1:]))


    def evaluate_fitness(self):
        #
        self.fitness_scores = [0 for _ in range(self.population_size)]
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j:
                    score1, _ = self.play_game(
                        self.population[i], self.population[j])
                    self.fitness_scores[i] += score1

    def select(self):
        # Implement selection logic here
        # A simple selection method, where strategies with higher fitness
        # have a higher chance of being selected.
        selected_indices = np.random.choice(self.population_size,
                                            size=self.population_size,
                                            replace=True, p=np.array(
                                                self.fitness_scores)/
                                            sum(self.fitness_scores))
        self.population = [self.population[i] for i in selected_indices]

    def crossover_and_mutate(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(self.population, 2, replace=False)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.chromosome_length - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    def mutate(self, chromosome):
        return ''.join(bit if np.random.random() > self.mutation_rate else str(1 - int(bit)) for bit in chromosome)

    def evolve_strategies(self):
        # Wrapper function to evolve strategies using GA components
        self.evaluate_fitness()
        self.select()
        self.crossover_and_mutate()


    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.initialize_population()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                   cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                       for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)

    def test_strategies(self):
        # Define strategy pairs to test
        strategy_pairs = [
            ('always_cooperate', 'always_cooperate'),
            ('always_cooperate', 'always_defect'),
            ('always_defect', 'always_cooperate'),
            ('always_defect', 'always_defect'),
            ('always_cooperate', 'tit_for_tat'),
            ('tit_for_tat', 'always_defect'),
            ('tit_for_tat', 'tit_for_tat')
        ]

        # Run each pair and print results
        for strategy1, strategy2 in strategy_pairs:
            score1, score2 = self.play_game(strategy1, strategy2)
            print(f"Game between {strategy1} and {strategy2}: {score1} - {score2}")


if __name__ == '__main__':
    sim = CASim()
    sim.test_strategies()
    # from pycx_gui import GUI
    # cx = GUI(sim)
    # cx.start()
