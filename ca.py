# Name: Ayoub Oqla & Samuel Ampadu
# Studentnr: 14281171 & 13186523
# BSc informatica
import numpy as np
import hashlib
from model import Model
import matplotlib.pyplot as plt

SCORES = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.make_param('population_size', 100)
        self.make_param('mutation_rate', 0.2)
        self.make_param('chromosome_length', 16)
        self.make_param('num_rounds', 16)
        self.make_param('generations', 100)
        self.make_param('inherit', 10)

        self.population = []

        self.scores = None

        self.averages = []

        self.strategies = [
            "Always Defect",
            "Always Cooperate",
            "Tit-for-Tat",
            "Grudge",
            "Nasty Tit-for-Tat",
            "Suspicious Tit for Tat",
            "Tit for Two Tats",
            "Discriminating Altruist",
            "Pavlov",
            "Grimtrigger"
        ]
        self.num_strategies = len(self.strategies)
        self.rule_tables = []

    # Stores the rule tables of the 10 chosen strategies.
    def encode_rule_table(self, strategy):
        rule_table = {}

        if strategy == "Always Defect":
            rule_table[('', '', '', '')] = 'D'
            rule_table[('', '', 'C', 'C')] = 'D'
            rule_table[('', '', 'C', 'D')] = 'D'
            rule_table[('', '', 'D', 'C')] = 'D'
            rule_table[('', '', 'D', 'D')] = 'D'
            rule_table[('C', 'C', 'C', 'C')] = 'D'
            rule_table[('C', 'C', 'C', 'D')] = 'D'
            rule_table[('C', 'C', 'D', 'C')] = 'D'
            rule_table[('C', 'C', 'D', 'D')] = 'D'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'D'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'D'
            rule_table[('D', 'C', 'C', 'D')] = 'D'
            rule_table[('D', 'C', 'D', 'C')] = 'D'
            rule_table[('D', 'C', 'D', 'D')] = 'D'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Always Cooperate":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'C'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'C'
            rule_table[('C', 'D', 'C', 'D')] = 'C'
            rule_table[('C', 'D', 'D', 'C')] = 'C'
            rule_table[('C', 'D', 'D', 'D')] = 'C'
            rule_table[('D', 'C', 'C', 'C')] = 'C'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'C'
            rule_table[('D', 'D', 'C', 'C')] = 'C'
            rule_table[('D', 'D', 'C', 'D')] = 'C'
            rule_table[('D', 'D', 'D', 'C')] = 'C'
            rule_table[('D', 'D', 'D', 'D')] = 'C'
        elif strategy == "Tit-for-Tat":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'D'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'D'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'D'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'C'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'C'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Grudge":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'D'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'D'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'D'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'D'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'D'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'D'
            rule_table[('D', 'C', 'C', 'D')] = 'D'
            rule_table[('D', 'C', 'D', 'C')] = 'D'
            rule_table[('D', 'C', 'D', 'D')] = 'D'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Nasty Tit-for-Tat":
            rule_table[('', '', '', '')] = 'D'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'C'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'D'
            rule_table[('C', 'C', 'C', 'D')] = 'D'
            rule_table[('C', 'C', 'D', 'C')] = 'D'
            rule_table[('C', 'C', 'D', 'D')] = 'D'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'C'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'D'
            rule_table[('D', 'C', 'C', 'D')] = 'D'
            rule_table[('D', 'C', 'D', 'C')] = 'D'
            rule_table[('D', 'C', 'D', 'D')] = 'D'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Suspicious Tit for Tat":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'C'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'C'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'C'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'C'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Tit for Two Tats":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'C'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'C'
            rule_table[('C', 'D', 'C', 'D')] = 'C'
            rule_table[('C', 'D', 'D', 'C')] = 'C'
            rule_table[('C', 'D', 'D', 'D')] = 'C'
            rule_table[('D', 'C', 'C', 'C')] = 'C'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'C'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'C'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Discriminating Altruist":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'C'
            rule_table[('', '', 'D', 'C')] = 'C'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'D'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'D'
            rule_table[('D', 'C', 'C', 'D')] = 'D'
            rule_table[('D', 'C', 'D', 'C')] = 'D'
            rule_table[('D', 'C', 'D', 'D')] = 'D'
            rule_table[('D', 'D', 'C', 'C')] = 'D'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Pavlov":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'D'
            rule_table[('', '', 'D', 'C')] = 'D'
            rule_table[('', '', 'D', 'D')] = 'C'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'D'
            rule_table[('C', 'C', 'D', 'D')] = 'D'
            rule_table[('C', 'D', 'C', 'C')] = 'D'
            rule_table[('C', 'D', 'C', 'D')] = 'D'
            rule_table[('C', 'D', 'D', 'C')] = 'D'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'D'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'D'
            rule_table[('D', 'D', 'C', 'C')] = 'C'
            rule_table[('D', 'D', 'C', 'D')] = 'D'
            rule_table[('D', 'D', 'D', 'C')] = 'D'
            rule_table[('D', 'D', 'D', 'D')] = 'D'
        elif strategy == "Grimtrigger":
            rule_table[('', '', '', '')] = 'C'
            rule_table[('', '', 'C', 'C')] = 'C'
            rule_table[('', '', 'C', 'D')] = 'D'
            rule_table[('', '', 'D', 'C')] = 'D'
            rule_table[('', '', 'D', 'D')] = 'D'
            rule_table[('C', 'C', 'C', 'C')] = 'C'
            rule_table[('C', 'C', 'C', 'D')] = 'C'
            rule_table[('C', 'C', 'D', 'C')] = 'C'
            rule_table[('C', 'C', 'D', 'D')] = 'C'
            rule_table[('C', 'D', 'C', 'C')] = 'C'
            rule_table[('C', 'D', 'C', 'D')] = 'C'
            rule_table[('C', 'D', 'D', 'C')] = 'C'
            rule_table[('C', 'D', 'D', 'D')] = 'D'
            rule_table[('D', 'C', 'C', 'C')] = 'C'
            rule_table[('D', 'C', 'C', 'D')] = 'C'
            rule_table[('D', 'C', 'D', 'C')] = 'C'
            rule_table[('D', 'C', 'D', 'D')] = 'C'
            rule_table[('D', 'D', 'C', 'C')] = 'C'
            rule_table[('D', 'D', 'C', 'D')] = 'C'
            rule_table[('D', 'D', 'D', 'C')] = 'C'
            rule_table[('D', 'D', 'D', 'D')] = 'D'

        return rule_table
    
    def initialize_rule_tables(self):
        self.rule_tables = [self.encode_rule_table(strategy) for strategy in self.strategies]

    def initialize_population(self):
        population = [''.join(np.random.choice(['C', 'D']) for _ in range(self.chromosome_length)) for _ in range(self.population_size)]
        return population
    
    def evaluate_fitness(self, population):
        fitness_scores = []
        for strategy in population:
            rule_table = self.decode_rule_table(strategy)
            score = self.run_tournament_with_rule_table(rule_table)
            fitness_scores.append(score)
        return fitness_scores

    def select_fittest(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort indices in descending order of fitness
        selected_population = [population[i] for i in sorted_indices[:self.population_size]]
        return selected_population[:self.inherit]
    
    def crossover(self, population):
        offspring = []
        while len(offspring) < self.population_size - len(population):
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            crossover_point = np.random.randint(1, self.chromosome_length)
            offspring += [parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]]
        return offspring
    
    def mutate(self, population):
        mutated_population = [''.join(['D' if np.random.rand() < self.mutation_rate and bit == 'C' else 'C' if np.random.rand() < self.mutation_rate and bit == 'D' else bit for bit in strategy]) for strategy in population]
        return mutated_population
    
    def evolve_strategies(self):
        fitness_scores = self.evaluate_fitness(self.population)
        selected_population = self.select_fittest(self.population, fitness_scores)
        offspring = self.crossover(selected_population)
        mutated_offspring = self.mutate(offspring)
        self.population = selected_population + mutated_offspring
        self.scores = fitness_scores
        self.averages.append(np.mean(fitness_scores))


    # Helper function. Makes rule tables of the chromosomes of the population.
    def decode_rule_table(self, chromosome):
        situations = [(a, b, c, d) for a in ['C', 'D'] for b in ['C', 'D'] for c in ['C', 'D'] for d in ['C', 'D']]
        rule_table = {situation: chromosome[i] for i, situation in enumerate(situations)}
        return rule_table
    
    # Plays the game player vs opponents (10 strategies)
    def run_tournament_with_rule_table(self, rule_table):
        score = 0
        for opponent_strategy in self.strategies:
            opponent_rule_table = self.encode_rule_table(opponent_strategy)
            history1 = history2 = ['', '']
            for _ in range(self.num_rounds):
                decision1 = rule_table.get(tuple(history1[-2:] + history2[-2:]), 'D')
                decision2 = opponent_rule_table.get(tuple(history2[-2:] + history1[-2:]))
                payoff1, payoff2 = SCORES[(decision1, decision2)]
                score += payoff1
                history1.append(decision1)
                history2.append(decision2)
        return score

    
    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.averages = []
        self.initialize_population()

    def draw(self):
        """Draws the current state of the grid."""
        import matplotlib.pyplot as plt

        plt.cla()
        x = np.arange(len(self.averages))
        plt.bar(x, self.averages)
        plt.ylabel('Score')
        plt.title('Average Scores of Strategies')
        plt.show()

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.generations:
            return True

        self.evolve_strategies()

    def print_results(self):
        print("Baseline Performance Evaluation")
        print("--------------------------------")
        print("Strategy\tAverage Score")
        print("--------------------------------")
        for i in range(0, self.population_size - 1):
            print(f"{i}\t{self.averages[i]}")

if __name__ == "__main__":
    sim = CASim()
    sim.initialize_rule_tables()
    sim.population = sim.initialize_population()
    from pycx_gui import GUI
    cx = GUI(sim)
    cx.start()
    sim.print_results()
