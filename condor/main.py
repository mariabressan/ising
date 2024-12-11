import numpy as np
import pickle
import argparse

def energy(lattice, N, J):
    ### Total energy with periodic boundary conditions ###
    E = 0
    for i in range(N):
        for j in range(N):
            s = lattice[i, j]
            nb = lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + lattice[i, (j+1)%N] + lattice[i, (j-1)%N]
            E -= s * nb
    return E * J / 2  # /2 to avoid double counting

def metropolis(lattice, N, T, J, n_sweeps, measure_every=1):
    ### Run Metropolis algorithm ###
    total_sites = N * N
    energies = []
    magnetizations = []
    
    for sweep in range(n_sweeps):
        for _ in range(total_sites):
            a, b = np.random.randint(0, N, 2)
            neighbors = (
                lattice[(a+1)%N, b] + lattice[(a-1)%N, b] +
                lattice[a, (b+1)%N] + lattice[a, (b-1)%N]
            )
            dE = 2 * J * lattice[a, b] * neighbors
            
            # Metropolis
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                lattice[a, b] *= -1

        # Measurements
        if sweep % measure_every == 0:
            E = energy(lattice, N, J)
            M = np.sum(lattice)
            energies.append(E)
            magnetizations.append(M)
    
    return np.array(energies), np.array(magnetizations)

def simulate(L, T):
    ### Main simulation ###
    J = 1  # Interaction strength
    n_thermalization = 10**5  # Thermalization sweeps
    n_measurement = 3 * 10**5  # Measurement sweeps
    measure_every = 10  # Measure every 10 sweeps
    
    results= {}
    N = L
    lattice = 2 * np.random.randint(2, size=(N, N)) - 1  # Random initial state
    
    # Thermalize the system
    metropolis(lattice, N, T, J, n_thermalization)

    # Perform measurements
    energies, magnetizations = metropolis(
        lattice, N, T, J, n_measurement, measure_every
    )
    
    # Calculate observables
    avg_E = np.mean(energies) / (N**2)
    avg_E2 = np.mean(energies**2) / (N**2)
    specific_heat = (avg_E2 - avg_E**2) / (T**2)

    avg_M = np.mean(magnetizations) / (N**2)
    avg_M2 = np.mean(magnetizations**2) / (N**2)
    susceptibility = (avg_M2 - avg_M**2) / T

    # Store results
    results = {
        'E': avg_E,
        'C': specific_heat,
        'M': avg_M,
        'Chi': susceptibility
    }

    # Save results for lattice size
    with open(f'L{L}_T{T}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

np.random.seed(42)
parser = argparse.ArgumentParser(description='Optional: specify variables')
parser.add_argument('-L',type=float)
parser.add_argument('-T',type=float)
args = parser.parse_args()
L = args.L
T = args.T
simulate(L,T)