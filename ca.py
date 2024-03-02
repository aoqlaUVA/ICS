import numpy as np
import hashlib
from model import Model

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None
        self.quiescent_state = 0
        self.state_hashes = set()
        self.transient_length = None
        self.make_param('pop_size', 100)
        self.make_param('mutation', 0.02)
        self.make_param('crossover', 0.7)
        self.make_param('generation', 100)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))
    
    def table_walk_through_for_lambda(self, lambda_prime):
        """
        Changes the rule table based on the lambda parameter.
        
        Args:
        lambda_prime (float): The target lambda value to increase to.
        """
        rule_set_size = self.k ** (2 * self.r + 1)
        self.rule_set = np.zeros(rule_set_size)
        self.quiescent_state = np.random.choice(list(range(self.k)))
        # Calculate the number of states that need to change
        num_states_to_change = int(lambda_prime * (self.k ** (2 * self.r + 1)))
        # Select the states to change
        states_to_change = np.random.choice(range(self.k ** (2 * self.r + 1)), num_states_to_change, replace=False)
        # Set the new states for the selected transitions
        for state in states_to_change:
            # Choose a random non-quiescent state
            non_quiescent_states = [s for s in range(self.k) if s != self.quiescent_state]
            self.rule_set[state] = np.random.choice(non_quiescent_states)
        # Update the current lambda value
        self.lambda_value = lambda_prime

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
        self.table_walk_through_for_lambda(self.lambda_value)

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
