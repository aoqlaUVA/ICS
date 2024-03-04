import numpy as np
import hashlib
from model import Model

SCORES = {
    ('C', 'C'): (1, 1),
    ('C', 'D'): (3, 0),
    ('D', 'C'): (0, 3),
    ('D', 'D'): (2, 2)
}


class PrisonersDilemma(Model):
    def __init__(self):
        super().__init__()
        self.make_param('population_size', 100)
        self.make_param('mutation_rate', 0.02)
        self.make_param('chromosome_length', 8)
        self.make_param('crossover_rate', 0.7)
        self.make_param('rounds', 10)
        self.population = self.initialize_population()
        self.fitness_scores = np.zeros(self.population_size)

    def initialize_population(self):
        # Initialize population with random strategies
        return [''.join(np.random.choice(['0', '1']) for _ in
                        range(self.chromosome_length)) for _ in
                range(self.population_size)]

    def play_game(self, strategy1, strategy2):
        moves1, moves2 = ['C'], ['C']  # Initialize with cooperation
        for _ in range(self.rounds):
            move1 = self.get_move(strategy1, moves2[-1])  # Use the last move of the opponent
            move2 = self.get_move(strategy2, moves1[-1])  # Use the last move of the opponent
            moves1.append(move1)
            moves2.append(move2)

        # Calculate scores after excluding the initial move
        scores_1 = sum(SCORES[(m1, m2)][0] for m1, m2 in zip(moves1[1:], moves2[1:]))
        scores_2 = sum(SCORES[(m1, m2)][1] for m1, m2 in zip(moves1[1:], moves2[1:]))
        return scores_1, scores_2

    def get_move(self, strategy, opponent_history):
        if strategy == 'always_cooperate':
            return 'C'
        elif strategy == 'always_defect':
            return 'D'
        elif strategy == 'tit_for_tat':
            if not opponent_history:
                return 'C'
            return opponent_history[-1]
        elif strategy == 'defect_2_percent_tit_for_tat':
            if not opponent_history:
                return 'C'
            if np.random.random() < 0.02:
                return 'D'
            return opponent_history[-1]
        elif strategy == 'grudge':
            if 'D' in opponent_history:
                return 'D'
            return 'C'
        elif strategy == 'nasty_tit_for_tat':
            if not opponent_history:
                return 'D'
            return opponent_history[-1]


        # else:
        #     # For binary-encoded strategies, use the last 3 moves to decide
        #     if len(opponent_history) >= 3:
        #         idx = int(
        #             ''.join('1' if m == 'D' else '0' for m in
        #                     opponent_history[-3:]), 2)
        #     else:
        #         idx = 0  # Default to cooperation if not enough history
        #     return 'D' if strategy[idx] == '1' else 'C'

    def evaluate_fitness(self):
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                score1, score2 = self.play_round(self.population[i],
                                                 self.population[j], rounds=200)
                self.fitness_scores[i] += score1
                self.fitness_scores[j] += score2

    def select(self):
        probabilities = self.fitness_scores / np.sum(self.fitness_scores)
        selected_indices = np.random.choice(range(
            self.population_size), size=self.population_size, replace=True, p=probabilities)
        self.population = [self.population[i] for i in selected_indices]
        self.fitness_scores = np.zeros(self.population_size)

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome):
        mutated = ''.join('1' if (bit == '0' and np.random.rand() <
                                  self.mutation_rate) else '0'
                          if (bit == '1' and
                              np.random.rand() < self.mutation_rate)
                          else bit for bit in chromosome)
        return mutated

    def crossover_and_mutate(self):
        next_generation = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = self.population[i], self.population[i+1]
            child1, child2 = self.crossover(parent1, parent2)
            next_generation.extend([self.mutate(child1), self.mutate(child2)])
        self.population = next_generation

    def evolve_strategies(self):
        # Wrapper function to evolve strategies using GA components
        self.evaluate_fitness()
        self.select()
        self.crossover_and_mutate()

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
        strategies = [
            'always_cooperate',
            'always_defect',
            'tit_for_tat',
            'defect_2_percent_tit_for_tat',
            'grudge',
            'nasty_tit_for_tat',
        ]

        # Generate all possible pairings including self-pairings
        strategy_pairs = [(s1, s2) for s1 in strategies for s2 in strategies]

        # Run each pair and print results
        for strategy1, strategy2 in strategy_pairs:
            score1, score2 = self.play_game(strategy1, strategy2)
            print(
                f"Game between {strategy1} and {strategy2}: {score1} - {score2}")


if __name__ == '__main__':
    prison_dilemma = PrisonersDilemma()
    prison_dilemma.test_strategies()
    # from pycx_gui import GUI
    # cx = GUI(sim)
    # cx.start()
