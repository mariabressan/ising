import numpy as np

L_sizes = [10, 16, 24, 36]  # Lattice sizes
T_values = np.arange(0.015, 4.5 + 0.015, 0.015)  # Temp range

with open('args.txt', 'w') as f:
    for L in L_sizes:
        for T in T_values:
            f.write(f'-L={L} -T={round(T,4)}\n')