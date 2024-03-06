# Name: Ayoub Oqla & Samuel Ampadu
# Studentnr: 14281171 & 13186523
# BSc informatica
import numpy as np
import matplotlib.pyplot as plt
import ca

def run_simulation(population_size, mutation_rate, generations, inherit, title):
    sim = ca.CASim()
    sim.population_size = population_size
    sim.mutation_rate = mutation_rate
    sim.generations = generations
    sim.inherit = inherit
    sim.initialize_rule_tables()
    sim.population = sim.initialize_population()
    
    for _ in range(generations):
        sim.evolve_strategies()

    plt.bar(range(len(sim.averages)), sim.averages)
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Average Score')
    plt.show()

if __name__ == "__main__":
    parameters = [
        (100, 0.2, 100, 5, "Population: 100, Mutation rate: 0.2, Generations: 100, Inherit: 5"),
        (100, 0.0, 100, 5, "Population: 100, Mutation rate: 0.0, Generations: 100, Inherit: 5"),
        (50, 0.35, 300, 10, "Population: 50, Mutation rate: 0.35, Generations: 300, Inherit: 10"),
    ]
    
    for population_size, mutation_rate, generations, inherit, title in parameters:
        run_simulation(population_size, mutation_rate, generations, inherit, title)
