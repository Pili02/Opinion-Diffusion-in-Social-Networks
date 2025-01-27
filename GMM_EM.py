import numpy as np
import random as rand
import matplotlib.pyplot as plt
from sklearn import mixture
import copy
from scipy.stats import multivariate_normal


class GMM:
    weights = []
    means = []
    covariances = []

    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.weights = np.zeros(n_components)
        self.means = np.zeros((n_components, n_features))
        self.covariances = np.zeros((n_components, n_features, n_features))

    def normalize(self):
        self.weights /= np.sum(self.weights)

    def sample(self):
        component = np.random.choice(self.n_components, p=self.weights)

        mean = self.means[component]
        cov = self.covariances[component]
        sample = np.random.multivariate_normal(mean, cov)

        return sample, component


num_agents = 10
num_features = 1
n_components = 4
alpha = 0.1
num_iterations = 100
max_cov_value = 5.0  # upper limit for covariances
max_weight_value = 0.5  # upper limit for weights
max_mean_value = 10.0  # upper limit for means

# we will assign random values to these agents
agents = [GMM(n_components, num_features) for i in range(num_agents)]

for agent in agents:
    agent.weights = np.random.rand(n_components)
    # agent.weights = np.clip(
    #     agent.weights, 0, max_weight_value
    # )
    agent.normalize()
for agent in agents:
    agent.means = np.random.rand(n_components, num_features) * max_mean_value
for agent in agents:
    agent.covariances = np.random.rand(n_components, num_features, num_features)
    agent.covariances = np.array(
        [np.dot(A, A.T) for A in agent.covariances]
    )  # To create positive semi-definite matrices
    # Now apply the upper limit
    agent.covariances = np.clip(agent.covariances, 0, max_cov_value)


remember_agents = []
for num in range(num_iterations):
    new_agents = []
    for i in range(num_agents):
        # print("i: ", i)
        agent1 = copy.deepcopy(agents[i])
        neibhoursOpinion = []
        for j in range(num_agents):
            if i == j:
                continue
            sample, component = agents[j].sample()
            neibhoursOpinion.append(sample)
        X = np.array(neibhoursOpinion)
        r = np.zeros((len(X), agent1.n_components))

        # E-step: calculate responsibilities
        for idx, x_val in enumerate(X):
            probs = np.array(
                [
                    agent1.weights[k]
                    * multivariate_normal.pdf(
                        x_val, agent1.means[k], agent1.covariances[k]
                    )
                    for k in range(agent1.n_components)
                ]
            )
            r[idx] = probs / np.sum(probs)

    remember_agents.append(new_agents)
    agents = new_agents

x_values = range(num_iterations)
for j in range(num_agents):
    y_values = [remember_agents[i][j].means[0] for i in range(num_iterations)]
    plt.plot(x_values, y_values, label=f"Agent {j + 1}")

plt.xlabel("Iteration")
plt.ylabel("Mean Value")

plt.legend(title="Agents", loc="upper right")
plt.show()
