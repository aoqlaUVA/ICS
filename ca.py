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
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.population = []  # Population of strategies
        self.fitness_scores = []  # Fitness scores for each strategy

        self.make_param('population_size', 100)
        self.make_param('mutation_rate', 0.02)
        self.make_param('chromosome_length', 4)
        self.make_param('crossover_rate', 0.7)
        self.make_param('population', 100, self.initialize_population)

    def initialize_population(self):
        # Initialize your population of strategies here
        return [''.join(np.random.choice(['0', '1'])
                        for _ in range(self.chromosome_length))
                for _ in range(self.population_size)]

    def play_game(self, strategy1, strategy2):
        # Assuming a simple strategy where the last bit decides the move
        # 'C' for cooperate, 'D' for defect
        move1 = 'C' if strategy1[-1] == '0' else 'D'
        move2 = 'C' if strategy2[-1] == '0' else 'D'
        return SCORES(move1, move2)

    def evaluate_fitness(self):
        #
        self.fitness_scores = [0 for _ in range(self.population_size)]
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j:
                    score1, score2 = self.play_game(
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
        # Implement crossover and mutation logic here

        pass

    def evolve_strategies(self):
        # Wrapper function to evolve strategies using GA components
        self.evaluate_fitness()
        self.select()
        self.crossover_and_mutate()

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        rev_inp = inp[::-1]
        sum = 0
        for i in range(len(rev_inp)):
            x = rev_inp[i] * self.k ** i
            sum = sum + int(x)

        rev_rule = self.rule_set[::-1]
        return rev_rule[sum]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        if self.random:
            np.random.seed(self.seed)
            init_row = np.random.randint(0, self.k, size=self.width)
        else:
            init_row = np.zeros(self.width, dtype=int)
            init_row[int(self.width / 2)] = self.k - 1
        return init_row

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


if __name__ == '__main__':
    sim = CASim()
    from pycx_gui import GUI
    cx = GUI(sim)
    cx.start()
