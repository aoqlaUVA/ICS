# import numpy as np
# import hashlib
# from model import Model

# SCORES = {
#     ('C', 'C'): (1, 1),
#     ('C', 'D'): (3, 0),
#     ('D', 'C'): (0, 3),
#     ('D', 'D'): (2, 2)
# }


# class PrisonersDilemma(Model):
#     def __init__(self):
#         super().__init__()
#         self.make_param('population_size', 100)
#         self.make_param('mutation_rate', 0.02)
#         self.make_param('chromosome_length', 8)
#         self.make_param('crossover_rate', 0.7)
#         self.make_param('rounds', 10)
#         self.population = self.initialize_population()
#         self.fitness_scores = np.zeros(self.population_size)

#     def initialize_population(self):
#         # Initialize population with random strategies
#         return [''.join(np.random.choice(['0', '1']) for _ in
#                         range(self.chromosome_length)) for _ in
#                 range(self.population_size)]

#     def play_game(self, strategy1, strategy2):
#         moves1, moves2 = ['C'], ['C']  # Initialize with cooperation
#         for _ in range(self.rounds):
#             move1 = self.get_move(strategy1, moves2[-1])
#             move2 = self.get_move(strategy2, moves1[-1])
#             moves1.append(move1)
#             moves2.append(move2)

#         # Calculate scores after excluding the initial move
#         scores_1 = sum(SCORES[(m1, m2)][0] for m1, m2 in zip(moves1[1:], moves2[1:]))
#         scores_2 = sum(SCORES[(m1, m2)][1] for m1, m2 in zip(moves1[1:], moves2[1:]))
#         return scores_1, scores_2

#     def get_move(self, strategy, opponent_history):
#         if strategy == 'always_cooperate':
#             return 'C'
#         elif strategy == 'always_defect':
#             return 'D'
#         elif strategy == 'tit_for_tat':
#             if not opponent_history:
#                 return 'C'
#             return opponent_history[-1]
#         elif strategy == 'defect_2_percent_tit_for_tat':
#             if not opponent_history:
#                 return 'C'
#             if np.random.random() < 0.02:
#                 return 'D'
#             return opponent_history[-1]
#         elif strategy == 'grudge':
#             if 'D' in opponent_history:
#                 return 'D'
#             return 'C'
#         elif strategy == 'nasty_tit_for_tat':
#             if not opponent_history:
#                 return 'D'
#             return opponent_history[-1]


#         # else:
#         #     # For binary-encoded strategies, use the last 3 moves to decide
#         #     if len(opponent_history) >= 3:
#         #         idx = int(
#         #             ''.join('1' if m == 'D' else '0' for m in
#         #                     opponent_history[-3:]), 2)
#         #     else:
#         #         idx = 0  # Default to cooperation if not enough history
#         #     return 'D' if strategy[idx] == '1' else 'C'

#     def evaluate_fitness(self):
#         for i in range(self.population_size):
#             for j in range(i + 1, self.population_size):
#                 score1, score2 = self.play_round(self.population[i],
#                                                  self.population[j], rounds=200)
#                 self.fitness_scores[i] += score1
#                 self.fitness_scores[j] += score2

#     def select(self):
#         probabilities = self.fitness_scores / np.sum(self.fitness_scores)
#         selected_indices = np.random.choice(range(
#             self.population_size), size=self.population_size, replace=True, p=probabilities)
#         self.population = [self.population[i] for i in selected_indices]
#         self.fitness_scores = np.zeros(self.population_size)

#     def crossover(self, parent1, parent2):
#         if np.random.random() < self.crossover_rate:
#             point = np.random.randint(1, self.chromosome_length - 1)
#             child1 = parent1[:point] + parent2[point:]
#             child2 = parent2[:point] + parent1[point:]
#             return child1, child2
#         return parent1, parent2

#     def mutate(self, chromosome):
#         mutated = ''.join('1' if (bit == '0' and np.random.rand() <
#                                   self.mutation_rate) else '0'
#                           if (bit == '1' and
#                               np.random.rand() < self.mutation_rate)
#                           else bit for bit in chromosome)
#         return mutated

#     def crossover_and_mutate(self):
#         next_generation = []
#         for i in range(0, self.population_size, 2):
#             parent1, parent2 = self.population[i], self.population[i+1]
#             child1, child2 = self.crossover(parent1, parent2)
#             next_generation.extend([self.mutate(child1), self.mutate(child2)])
#         self.population = next_generation

#     def evolve_strategies(self):
#         # Wrapper function to evolve strategies using GA components
#         self.evaluate_fitness()
#         self.select()
#         self.crossover_and_mutate()

#     def draw(self):
#         """Draws the current state of the grid."""

#         import matplotlib
#         import matplotlib.pyplot as plt

#         plt.cla()
#         if not plt.gca().yaxis_inverted():
#             plt.gca().invert_yaxis()
#         plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
#                    cmap=matplotlib.cm.binary)
#         plt.axis('image')
#         plt.title('t = %d' % self.t)

#     def step(self):
#         """Performs a single step of the simulation by advancing time (and thus
#         row) and applying the rule to determine the state of the cells."""
#         self.t += 1
#         if self.t >= self.height:
#             return True

#         for patch in range(self.width):
#             # We want the items r to the left and to the right of this patch,
#             # while wrapping around (e.g. index -1 is the last item on the row).
#             # Since slices do not support this, we create an array with the
#             # indices we want and use that to index our grid.
#             indices = [i % self.width
#                        for i in range(patch - self.r, patch + self.r + 1)]
#             values = self.config[self.t - 1, indices]
#             self.config[self.t, patch] = self.check_rule(values)

#     def test_strategies(self):
#         # Define strategy pairs to test
#         strategies = [
#             'always_cooperate',
#             'always_defect',
#             'tit_for_tat',
#             'defect_2_percent_tit_for_tat',
#             'grudge',
#             'nasty_tit_for_tat',
#         ]

#         # Generate all possible pairings including self-pairings
#         strategy_pairs = [(s1, s2) for s1 in strategies for s2 in strategies]

#         # Run each pair and print results
#         for strategy1, strategy2 in strategy_pairs:
#             score1, score2 = self.play_game(strategy1, strategy2)
#             print(
#                 f"Game between {strategy1} and {strategy2}: {score1} - {score2}")


# if __name__ == '__main__':
#     prison_dilemma = PrisonersDilemma()
#     prison_dilemma.test_strategies()
#     # from pycx_gui import GUI
#     # cx = GUI(sim)
#     # cx.start()

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
        Model.__init__(self)

        self.make_param('population_size', 100)
        self.make_param('mutation_rate', 0.02)
        self.make_param('chromosome_length', 10)
        self.make_param('crossover_rate', 0.7)
        self.make_param('num_rounds', 50)
        self.make_param('generations', 100)

        self.strategies = [
            "Always Defect",
            "Always Cooperate",
            "Tit-for-Tat",
            "Grudge",
            "Nasty Tit-for-Tat",
            "Suspicious Tit for Tat",
            # "Tit for Two Tats",
            # "Two Tits for Tat",
            "Discriminating Altruist",
            "Pavlov"
        ]
        self.num_strategies = len(self.strategies)
        self.rule_tables = []

    def encode_rule_table(self, strategy):
        rule_table = {}
        if strategy == "Always Defect":
            rule_table[('C', 'C')] = 'D'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'D'
            rule_table[('D', 'D')] = 'D'
        elif strategy == "Always Cooperate":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'C'
            rule_table[('D', 'C')] = 'C'
            rule_table[('D', 'D')] = 'C'
        elif strategy == "Tit-for-Tat":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'C'
            rule_table[('D', 'D')] = 'D'
        elif strategy == "Grudge":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'D'
            rule_table[('D', 'D')] = 'D'
        elif strategy == "Nasty Tit-for-Tat":
            rule_table[('C', 'C')] = 'D'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'C'
            rule_table[('D', 'D')] = 'D'
        elif strategy == "Suspicious Tit for Tat":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'C'
            rule_table[('D', 'D')] = 'D'
        # elif strategy == "Tit for Two Tats":
        #     rule_table[('C', 'C')] = 'C'
        #     rule_table[('C', 'D')] = 'C'
        #     rule_table[('D', 'C')] = 'C'
        #     rule_table[('D', 'D')] = 'D'
        # elif strategy == "Two Tits for Tat":
        #     rule_table[('C', 'C')] = 'C'
        #     rule_table[('C', 'D')] = 'D'
        #     rule_table[('D', 'C')] = 'C'
        #     rule_table[('D', 'D')] = 'C'
        elif strategy == "Discriminating Altruist":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'D'
            rule_table[('D', 'D')] = 'D'
        elif strategy == "Pavlov":
            rule_table[('C', 'C')] = 'C'
            rule_table[('C', 'D')] = 'D'
            rule_table[('D', 'C')] = 'C'
            rule_table[('D', 'D')] = 'D'
        return rule_table

    def initialize_rule_tables(self):
        self.rule_tables = [self.encode_rule_table(
            strategy) for strategy in self.strategies]

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = ''.join(np.random.choice(
                ['C', 'D']) for _ in range(self.chromosome_length))
            population.append(chromosome)
        return population

    def evaluate_fitness(self, population):
        fitness_scores = []
        for strategy in population:
            rule_table = self.decode_rule_table(strategy)
            score = self.run_tournament_with_rule_table(rule_table)
            fitness_scores.append(score)
        return fitness_scores

    def select_fittest(self, population, fitness_scores):
        # Sort indices in descending order of fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_population = [population[i]
                               for i in sorted_indices[:self.population_size]]
        return selected_population

    def crossover(self, population):
        num_parents = len(population)
        num_offspring = self.population_size - num_parents
        offspring = []
        for _ in range(num_offspring):
            parent1, parent2 = np.random.choice(
                population, size=2, replace=False)
            # Choose a random crossover point
            crossover_point = np.random.randint(1, self.chromosome_length)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.append(offspring1)
            offspring.append(offspring2)
        return offspring

    def mutate(self, population):
        mutated_population = []
        for strategy in population:
            mutated_strategy = ''
            for bit in strategy:
                if np.random.rand() < self.mutation_rate:
                    # Flip the bit with the mutation rate probability
                    mutated_strategy += 'D' if bit == 'C' else 'C'
                else:
                    mutated_strategy += bit
            mutated_population.append(mutated_strategy)
        return mutated_population

    def evolve_strategies(self):
        population = self.initialize_population()
        all_fitness_scores = []
        for _ in range(self.generations):
            fitness_scores = self.evaluate_fitness(population)
            selected_population = self.select_fittest(
                population, fitness_scores)
            offspring = self.crossover(selected_population)
            mutated_offspring = self.mutate(offspring)
            population = selected_population + mutated_offspring
            all_fitness_scores.append(fitness_scores)
        return population, all_fitness_scores

    def decode_rule_table(self, chromosome):
        rule_table = {}
        for i in range(len(chromosome)):
            row = i // 2
            col = i % 2
            move = chromosome[i]
            if col == 0:
                current_key = ('C', 'C') if row == 0 else ('C', 'D')
            else:
                current_key = ('D', 'C') if row == 0 else ('D', 'D')
            rule_table[current_key] = move
        return rule_table

    def run_tournament_with_rule_table(self, rule_table):
        score1 = score2 = 0
        history1 = history2 = ['']
        for _ in range(self.num_rounds):
            decision1 = rule_table.get((history1[-1], history2[-1]), 'C')
            decision2 = rule_table.get((history2[-1], history1[-1]), 'C')
            payoff1, payoff2 = SCORES[(decision1, decision2)]
            score1 += payoff1
            score2 += payoff2
            history1.append(decision1)
            history2.append(decision2)
        return score1

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
#     sim = PrisonersDilemma()
#     from pycx_gui import GUI
#     cx = GUI(sim)
#     cx.start()


if __name__ == "__main__":
    sim = PrisonersDilemma()
    evolved_population, all_fitness_scores = sim.evolve_strategies()
    print("Evolved Population:")
    for strategy in evolved_population:
        print(strategy)
    print("Fitness Scores over Generations:")
    for gen, fitness_scores in enumerate(all_fitness_scores):
        print(f"Generation {gen + 1}: {fitness_scores}")
