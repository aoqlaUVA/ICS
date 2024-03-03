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

        self.make_param('population_size', 100)
        self.make_param('mutation_rate', 0.02)
        self.make_param('chromosome_length', 10)
        self.make_param('crossover_rate', 0.7)
        self.make_param('num_rounds', 10)
        self.make_param('generations', 100)

        self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random strategies
        self.population = [self.encode_strategy(''.join(np.random.choice(['C', 'D']) for _ in range(self.chromosome_length))) 
                           for _ in range(self.population_size)]

    def encode_strategy(self, strategy):
        # Encode the strategy as a binary string
        return ''.join(['0' if decision == 'C' else '1' for decision in strategy])

    def decode_strategy(self, encoded_strategy):
        # Decode the binary string into a strategy
        return ''.join(['C' if gene == '0' else 'D' for gene in encoded_strategy])

    def encode_rule_table(self, strategy):
        rule_table = {}
        for i in range(2 ** (2 * self.num_rounds)):
            binary_string = bin(i)[2:].zfill(2 * self.num_rounds)
            history_pair = tuple(binary_string[j:j + 2] for j in range(0, len(binary_string), 2))
            rule_table[history_pair] = strategy(i)
        return rule_table
    
    def initialize_rule_tables(self):
        self.rule_tables = [self.encode_rule_table(strategy) for strategy in self.strategies]

    def play_prisoners_dilemma(self, rule_table1, rule_table2):
        score1 = score2 = 0
        history1 = history2 = []
        for _ in range(self.num_rounds):
            history_pair1 = tuple(history1[-1:]) + tuple(history2[-1:])
            history_pair2 = tuple(history2[-1:]) + tuple(history1[-1:])
            decision1 = rule_table1.get(history_pair1, 'C')
            decision2 = rule_table2.get(history_pair2, 'C')
            payoff1, payoff2 = SCORES[(decision1, decision2)]
            score1 += payoff1
            score2 += payoff2
            history1.append(decision1)
            history2.append(decision2)
        return score1, score2

    def run_tournament(self):
        self.initialize_rule_tables()
        self.scores = np.zeros((self.num_strategies, self.num_strategies))
        for i in range(self.num_strategies):
            for j in range(i, self.num_strategies):
                score1, score2 = self.play_prisoners_dilemma(self.rule_tables[i], self.rule_tables[j])
                self.scores[i, j] = score1
                self.scores[j, i] = score2
    
    def print_results(self):
        print("Baseline Performance Evaluation")
        print("--------------------------------")
        print("Strategy\tAverage Score")
        print("--------------------------------")
        for i in range(self.num_strategies):
            avg_score = np.mean(self.scores[i])
            print(f"{self.strategy_names[i]}\t{avg_score:.2f}")

    def tournament_selection(self, fitness_scores, tournament_size):
        # Perform tournament selection
        selected_indices = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(self.population_size, tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_index)
        return [self.population[i] for i in selected_indices]

    def single_point_crossover(self, parent1, parent2):
        # Perform single-point crossover
        crossover_point = np.random.randint(1, len(parent1))  # Choose a random crossover point
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def mutate_strategy(self, strategy, mutation_rate):
        # Perform mutation
        mutated_strategy = ''
        for bit in strategy:
            if np.random.rand() < mutation_rate:
                mutated_strategy += '1' if bit == '0' else '0'  # Flip the bit with the mutation rate probability
            else:
                mutated_strategy += bit
        return mutated_strategy

    def evolve_strategies(self):
        # Evolve the population through generations
        for _ in range(self.generations):
            self.evaluate_fitness()

            # Selection
            selected_strategies = self.tournament_selection(self.fitness_scores, tournament_size=5)

            # Crossover
            new_population = []
            for i in range(0, len(selected_strategies), 2):
                parent1 = selected_strategies[i]
                parent2 = selected_strategies[i + 1]
                offspring1, offspring2 = self.single_point_crossover(parent1, parent2)
                new_population.extend([offspring1, offspring2])

            # Mutation
            self.population = [self.mutate_strategy(strategy, self.mutation_rate) for strategy in new_population]

    def evaluate_fitness(self):
        # Evaluate the fitness of each strategy based on their performance in the Prisoner's Dilemma game
        self.fitness_scores = []
        for strategy in self.population:
            total_score = 0
            for opponent_strategy in self.population:
                score, _ = self.play_prisoners_dilemma(strategy, opponent_strategy)
                total_score += score
            self.fitness_scores.append(total_score)

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

# if __name__ == '__main__':
#     sim = CASim()
#     from pycx_gui import GUI
#     cx = GUI(sim)
#     cx.start()

if __name__ == "__main__":
    sim = CASim()
    sim.run_tournament()
    sim.print_results()