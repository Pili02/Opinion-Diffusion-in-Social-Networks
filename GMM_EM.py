import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.weights = np.random.rand(n_components)
        self.weights /= np.sum(self.weights)  # Normalize weights
        self.means = np.random.rand(n_components, n_features) * 10  # Random means
        self.covariances = np.array(
            [np.eye(n_features) * np.random.rand() for _ in range(n_components)]
        )  # Positive definite covariance matrices

    def sample(self):
        """Samples a point from the mixture model."""
        component = np.random.choice(self.n_components, p=self.weights)
        sample = np.random.multivariate_normal(
            self.means[component], self.covariances[component]
        )
        return sample, component

    def update_gmm(self, dataset, num_iterations=100):
        """Expectation-Maximization (EM) for GMM update."""
        n_samples = dataset.shape[0]
        if n_samples == 0:
            return  # No data to update

        epsilon = 1e-6  # Prevent division by zero

        # E-M Algorithm
        for _ in range(num_iterations):
            # **E-step: Compute Responsibilities**
            responsibilities = np.zeros((n_samples, self.n_components))
            for idx, x_val in enumerate(dataset):
                probs = np.array(
                    [
                        self.weights[k]
                        * multivariate_normal.pdf(
                            x_val, self.means[k], self.covariances[k]
                        )
                        for k in range(self.n_components)
                    ]
                )
                sum_probs = np.sum(probs)
                if sum_probs == 0:
                    responsibilities[idx] = (
                        np.ones(self.n_components) / self.n_components
                    )  # Assign equal weights
                else:
                    responsibilities[idx] = probs / sum_probs

            # **M-step: Update Parameters**
            class_responsibilities = (
                np.sum(responsibilities, axis=0) + epsilon
            )  # Prevent division by zero
            self.weights = class_responsibilities / n_samples
            self.weights /= np.sum(self.weights)  # Normalize weights

            for k in range(self.n_components):
                weighted_sum = np.sum(
                    responsibilities[:, k].reshape(-1, 1) * dataset, axis=0
                )
                self.means[k] = weighted_sum / class_responsibilities[k]

                # Compute new covariance
                diff = dataset - self.means[k]
                weighted_cov = np.dot(
                    (responsibilities[:, k].reshape(-1, 1) * diff).T, diff
                )
                self.covariances[k] = (
                    weighted_cov / class_responsibilities[k]
                    + np.eye(self.n_features) * epsilon
                )  # Ensure positive-definiteness


num_agents = 10
num_features = 1
n_components = 4
num_iterations = 100

# Initialize agents with GMMs
agents = [GMM(n_components, num_features) for _ in range(num_agents)]

remember_agents = []
for iteration in range(num_iterations):
    new_agents = []
    for i in range(num_agents):
        agent1 = copy.deepcopy(agents[i])
        neighbors_opinion = []

        # Collect samples from all other agents
        for j in range(num_agents):
            if i != j:
                sample, _ = agents[j].sample()
                neighbors_opinion.append(sample)

        neighbors_opinion = np.array(neighbors_opinion)
        agent1.update_gmm(neighbors_opinion, num_iterations=100)  # Run EM for 100 steps
        new_agents.append(agent1)

    remember_agents.append(new_agents)
    agents = new_agents

# Plot the Evolution of Means Over Iterations**
x_values = range(num_iterations)
for j in range(num_agents):
    y_values = [remember_agents[i][j].means[0][0] for i in range(num_iterations)]
    plt.plot(x_values, y_values, label=f"Agent {j + 1}")

plt.xlabel("Iteration")
plt.ylabel("Mean Value")
plt.legend(title="Agents", loc="upper right")
plt.show()
