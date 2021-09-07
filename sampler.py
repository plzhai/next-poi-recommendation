import numpy as np

class UniformNegativeSampler():
    def __init__(self, nloc):
        self.n_loc = nloc

    def sample(self, seq_length, k):
        neg_samples, probs = np.random.randint(1, self.n_loc, [seq_length, k]), np.ones((seq_length, k), dtype=np.float32)
        return neg_samples, probs
