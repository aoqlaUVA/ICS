# Name: Ayoub Oqla
# Studentnr: 14281171
# BSc informatica
# Be patient when running, might take 1-3 minutes :).
import matplotlib.pyplot as plt
import numpy as np
import ca

def make_instance(lambda_value):
    sim = ca.CASim()
    sim.lambda_value = lambda_value
    sim.r = 2
    sim.k = 4
    sim.width = 200
    sim.height = 200
    sim.reset()
    return sim

def run_experiment(start_lambda, end_lambda, lambda_step):
    lambda_values = np.arange(start_lambda, end_lambda, lambda_step)
    transient_lengths = []

    for lambda_value in lambda_values:
        sim = make_instance(lambda_value)

        while not sim.step():
            pass  # Run the CA until it stops or repeats

        transient_lengths.append(sim.transient_length if sim.transient_length is not None else 0)

    return lambda_values, transient_lengths

lambda_values, transient_lengths = run_experiment(0.0, 1.0, 0.01)
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, transient_lengths, marker='o')
plt.title('Transient Length vs. Lambda')
plt.xlabel('Lambda Value')
plt.ylabel('Transient Length')
plt.grid(True)
plt.show()
