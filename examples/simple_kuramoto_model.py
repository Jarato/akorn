import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 3  # Number of oscillators
dt = 0.01  # Time step
T = 10000  # Total number of time steps
omega = np.random.uniform(0.8, 1.2, N)  # Natural frequencies
K = np.random.uniform(0.1, 0.5, (N, N))  # Random coupling matrix

# Initial phases
theta = np.random.uniform(0, 2*np.pi, N)

# Simulation
history = np.zeros((T, N))

for t in range(T):
    history[t] = theta  # Store phases
    dtheta = omega + np.sum(K * np.sin(np.subtract.outer(theta, theta)), axis=1)
    theta += dt * dtheta # Euler update
    theta = theta % (2*np.pi)

# Plot results
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(history[:, i], label=f'Oscillator {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Phase (radians)')
plt.title('Kuramoto Model Simulation')
plt.legend()
plt.show()
