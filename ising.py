import os
import numpy as np
import pickle
import time
from numba import jit

@jit
def energy(lattice, N, J):
    """Calculate total energy with periodic boundary conditions."""
    E = 0
    for i in range(N):
        for j in range(N):
            s = lattice[i, j]
            nb = lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + lattice[i, (j+1)%N] + lattice[i, (j-1)%N]
            E -= s * nb
    return E * J / 2  # Divide by 2 to avoid double counting

@jit
def metropolis(lattice, N, T, J, n_sweeps, measure_every):
    """Run Metropolis algorithm."""
    total_sites = N * N

    total_energy = 0
    total_energy2 = 0
    total_mag = 0
    total_mag2 = 0
    n_measurements = 0

    for sweep in range(n_sweeps):
        for _ in range(total_sites):
            a, b = np.random.randint(0, N, 2)
            neighbors = (
                lattice[(a+1)%N, b] + lattice[(a-1)%N, b] +
                lattice[a, (b+1)%N] + lattice[a, (b-1)%N]
            )
            dE = 2 * J * lattice[a, b] * neighbors
            
            # Metropolis criterion
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                lattice[a, b] *= -1

        # Measurements
        if sweep % measure_every == 0:
            E = energy(lattice, N, J)
            M = np.sum(lattice)
            total_energy += E
            total_energy2 += E**2
            total_mag += abs(M)
            total_mag2 += M**2
            n_measurements += 1

    # Average measurements
    avg_energy = total_energy / n_measurements
    avg_energy2 = total_energy2 / n_measurements
    avg_mag = total_mag / n_measurements
    avg_mag2 = total_mag2 / n_measurements
    
    return avg_energy, avg_energy2, avg_mag, avg_mag2

def simulate(L, T):
    """Main simulation routine."""
    J = 1  # Interaction strength
    n_thermalization = 10**5  # Thermalization sweeps
    n_measurement = 3 * 10**5  # Measurement sweeps
    measure_every = 10  # Measure every 10 sweeps
    
    results = {}
    lattice = 2 * np.random.randint(2, size=(L, L)) - 1  # Random initial state
    
    # Thermalize
    metropolis(lattice, L, T, J, n_thermalization, measure_every=1)

    # Perform measurements
    avg_E, avg_E2, avg_M, avg_M2 = metropolis(lattice, L, T, J, n_measurement, measure_every)
    
    # Calculate observables
    specific_heat = (avg_E2 - avg_E**2) / (T**2)
    susceptibility = (avg_M2 - avg_M**2) / T
    #susceptibility = N*(avg_M2/N - avg_M**2) / T

    # Store results
    results = {
        'E': avg_E / (L**2),
        'C': specific_heat / (L**2),
        'M': avg_M / (L**2),
        'Chi': susceptibility / (L**2)
    }

    # Ensure output directory exists
    output_dir = f'out/{L}/'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(f'{output_dir}/{T:.5f}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

# Main Simulation 
L_sizes = [16]  # [10,16,24,36] Example lattice sizes
T_values = np.arange(0.015, 4.5 + 0.015, 0.015)  # Temperature range
T_values = [round(T, 5) for T in T_values]

time_i = time.time()
for L in L_sizes:
    print(f"L={L}")
    for i, T in enumerate(T_values):
        simulate(L, T)
        elapsed = (time.time() - time_i) / 60
        expected = elapsed / (i + 1) * len(T_values)
        progress = (i + 1) / len(T_values) * 100
        print(f"   T={T:.5f}, Done {elapsed:.2f} min ({progress:.1f}%), Expected Total: {expected:.2f} min")
