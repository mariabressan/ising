import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from glob import glob

colors = ["blue", "red", "green", "purple"]
ylims = {"C":3,"Chi":40,"chi_scaling":0.06}

def load():
    results = {}
    for lattice_dir in glob("output/*"):
        L = int(os.path.basename(lattice_dir))  
        results[L] = {}
        for file in glob(f"{lattice_dir}/*.pkl"):
            T = float(os.path.basename(file).replace('.pkl', '')) 
            with open(file, 'rb') as f:
                results[L][T] = pickle.load(f)
    return results

def critical(results, L, var):
    Ts = np.zeros(300)
    varns = np.zeros(300)
    for j, T in enumerate(results[L]):
        Ts[j] = T
        varns[j] = results[L][T][var]
    window = (Ts > 2) & (Ts < 3)
    T, X = Ts[window], varns[window]
    peak = np.argmax(X)
    return T[peak], X[peak]

def fit_line(xx, yy):
    try:
        guess_m = (yy[1]-yy[0])/(xx[1]-xx[0])
        guess_b = 0
        guess = np.array([guess_m, guess_b])

        lin = lambda x, m, b: m*x + b

        popt, pcov = optimize.curve_fit(lin, xx, yy, p0 = guess)
        m, b = popt
        dm, db = pcov[0][0], pcov[1][1]
        fit_func = lin(xx, m, b)

    except IndexError:
        guess_m = 0
        m, b, dm, db, fit_func = 0, 0, 0, 0, 0
    return(m, b, dm, db, fit_func)

def plot_vars(results, lattice_sizes):
    for var, varn in zip(["E","M","C","Chi"],["E/N","M","$C_v$","$\chi$"]):

        fig, ax = plt.subplots()

        for i, L in enumerate(lattice_sizes):
            T = np.arange(0.015, 4.5 + 0.015, 0.015)
            X = np.zeros(len(T))
            for j in range(len(X)):
                X[j] = results[L][round(T[j],5)][var]

            if var in ["C", "Chi"]:
                Tc, Xc = critical(results, L, var)
                ax.plot(T, X, color=colors[i], label=f"L = {L}: $T_c$ = {round(Tc,3)}, {varn}$_c$ = {round(Xc,3)}")
                ax.set(ylim=(0,ylims[var]))
            else:
                ax.plot(T, X, color=colors[i], label=f"L = {L}")
            
        ax.set(xlabel="Temperature [J]", ylabel=varn)
        ax.grid()
        ax.legend()
        fig.savefig(f"plots/{var}.png")

        plt.clf()

def Tc(results, lattice_sizes, var):
        fig, ax = plt.subplots()

        critical_temperatures = []
        x = [] # = L^-1

        for i, L in enumerate(lattice_sizes):
            Tc, Chic = critical(results, L, var)
            critical_temperatures.append(Tc); x.append(1/L)
            ax.scatter(1/L, Tc, color=colors[i], label=f"L = {L}")

        critical_temperatures, x = np.array(critical_temperatures), np.array(x)

        m, b, _, _, fit_func = fit_line(x, critical_temperatures)

        if var == "Chi":
            ax.plot(x, fit_func, color="black", label=f"$T_c$ = {round(b,3)}")
            ax.set(xlabel=r"$L^{-1}$", ylabel=r"$T_{c}(L)$")
        elif var == "C":
            ax.plot(x, fit_func, color="black", label=f"$T_c'$ = {round(b,3)}")
            ax.set(xlabel=r"$L^{-1}$", ylabel=r"$T_{c}'(L)$")
        else:
            print("Variable error")
        ax.grid()
        ax.legend()
        fig.savefig(f"plots/Tc_{var}.png")

        plt.clf()

def gamma_nu(results, lattice_sizes):
    fig, ax = plt.subplots()

    log_Xc = []
    log_L = []
    
    for i, L in enumerate(lattice_sizes):
        _, Xc = critical(results, L, "Chi")
        log_Xc.append(np.log(Xc)); log_L.append(np.log(L))
        ax.scatter(np.log(L), np.log(Xc), color=colors[i], label=f"L = {L}") 

    m, b, _, _, fit_func = fit_line(np.array(log_L), np.array(log_Xc))
    ax.plot(log_L, fit_func, color="black", label=r"$\gamma$/$\nu$ =" + f"{round(m,3)}, $f(0)$ = {round(b,3)}")
    ax.set(xlabel=r"$\ln(L)$", ylabel=r"$\ln(\chi_c)$")
    ax.legend()
    fig.savefig("plots/gamma_nu.png")

    plt.clf()

def chi_scaling(results, lattice_sizes):

    log_Xc = []
    log_L = []
    
    for i, L in enumerate(lattice_sizes):
        _, Xc = critical(results, L, "Chi")
        log_Xc.append(np.log(Xc)); log_L.append(np.log(L))

    log_Xc, log_L = np.array(log_Xc), np.array(log_L)
    gamma_nu, _, _, _, _ = fit_line(log_L, log_Xc)

    fig, ax = plt.subplots()

    for i, L in enumerate(lattice_sizes):
        T = np.zeros(300)
        X = np.zeros(300)
        for j,t in enumerate(results[L]):
            T[j] = t
            X[j] = results[L][t]["Chi"]
        Tc, _ = critical(results, L, "Chi")
        
        ax.plot(L*(T-Tc), np.array(L**(-gamma_nu))*np.array(X), linestyle="--", color=colors[i], label=f"L = {L}") 
    
    ax.set(xlabel=r"$L^{1/\nu}(T-T_c(L))$", ylabel=r"$L^{-\gamma/\nu}\chi$",ylim=(0,ylims["chi_scaling"]))
    ax.grid()
    ax.legend()

    fig.savefig("plots/chi_scaling.png")
    plt.clf()

def find_intersection(x, y1, y2):
    window = (x > 2) & (x < 3)
    where = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    valid_where = where[window[where]]
    return x[valid_where], y1[valid_where], y2[valid_where]

def beta_nu(results, lattice_sizes):

    beta_nu = 0.25

    fig, ax = plt.subplots()

    M_small = np.zeros(300)
    M_large = np.zeros(300)
    for i,t in enumerate(results[lattice_sizes[0]]):
        M_small[i] = results[lattice_sizes[0]][t]["M"]
        M_large[i] = results[lattice_sizes[-1]][t]["M"]
        
    for i, L in enumerate(lattice_sizes):
        T = np.zeros(300)
        M = np.zeros(300)
        for j,t in enumerate(results[L]):
            T[j] = t
            M[j] = results[L][t]["M"]
        ax.plot(T, L**(beta_nu)*np.array(M), color=colors[i], label=f"L = {L}") 
    
    Tc, Mc, _ = find_intersection(T, lattice_sizes[0]**(beta_nu)*M_small, 
                                    lattice_sizes[-1]**(beta_nu)*M_large)
    
    ax.plot(Tc, Mc, marker="x", color="black", label = f"$T_c$ = {round(Tc[0], 3)}")

    ax.set(xlabel="Temperature [J]", ylabel=r"$L^{\beta/\nu}M$")
    ax.legend()
    fig.savefig("plots/beta_nu.png")
    plt.clf()

def M_scaling(results, lattice_sizes):
    beta_nu = 0.25    
    fig, ax = plt.subplots()

    M_small = np.zeros(300)
    M_large = np.zeros(300)
    T = np.zeros(300)
    for i,t in enumerate(results[lattice_sizes[0]]):
        T[i] = t
        M_small[i] = results[lattice_sizes[0]][t]["M"]
        M_large[i] = results[lattice_sizes[-1]][t]["M"]
    Tc, _, _ = find_intersection(T, 
                                    lattice_sizes[0]**(beta_nu)*M_small, 
                                    lattice_sizes[-1]**(beta_nu)*M_large)

    for i, L in enumerate(lattice_sizes):
        M = np.zeros(300)
        for j,t in enumerate(results[L]):
            M[j] = results[L][t]["M"]
        ax.plot(L*(T-Tc), L**(beta_nu)*M, color=colors[i], label=f"L = {L}") 

    ax.set(xlabel=r"$L^{1/\nu}(T-T_c(L))$", ylabel=r"$L^{\beta/\nu}M$")
    ax.legend()
    fig.savefig("plots/M_scale.png")
    plt.clf()

def C_div(results, lattice_sizes):
    fig, ax = plt.subplots()

    Ccs = np.zeros(len(lattice_sizes))
    log_L = np.zeros(len(lattice_sizes))

    for i, L in enumerate(lattice_sizes):
        _, Cc = critical(results, L, "C")
        Ccs[i] = Cc
        log_L[i] = np.log(L)
        ax.scatter(np.log(L), Cc, color=colors[i], label=f"L = {L}")

    Ccs, log_L = np.array(Ccs), np.array(log_L)
    
    ax.set(xlabel=r"$\ln(L)$", ylabel=r"$C_c(L)$")
    ax.legend()
    fig.savefig("plots/C_div.png")
    plt.clf()

def entropy(results, lattice_sizes):
    fig, ax = plt.subplots() 

    for i, L in enumerate(lattice_sizes):
        T = np.zeros(300)
        C = np.zeros(300)
        for j, t in enumerate(results[L]):
            T[j] = t
            C[j] = results[L][t]["C"]
        S = np.cumsum(np.diff(T)[0]*C/T)
        ax.plot(T, S, color=colors[i], label=f"L = {L}")

    ax.axhline(y = np.log(2), label=r"$\ln(2)$")

    ax.set(xlabel="Temperature [J]", ylabel=r"$S(T)$", xlim=(T.min(), T.max()))
    ax.legend()
    fig.savefig("plots/entropy.png")
    plt.clf()

def free_energy(results, lattice_sizes):
    fig, ax = plt.subplots() 

    for i, L in enumerate(lattice_sizes):
        T, C, E = np.zeros(300), np.zeros(300), np.zeros(300)
        for j, t in enumerate(results[L]):
            T[j] = t
            C[j] = results[L][t]["C"]
            E[j] = results[L][t]["E"]
        S = np.cumsum(np.diff(T)[0]*C/T)
        ax.plot(T, E-T*S, color=colors[i], label=f"L = {L}")

    ax.set(xlabel="Temperature [J]", ylabel=r"$F = E-TS$")
    ax.legend()
    fig.savefig("plots/free_energy.png")
    plt.clf()
    
def main():
    results = load()
    lattice_sizes = [10, 16, 24, 36]
    plot_vars(results, lattice_sizes)
    Tc(results, lattice_sizes, "C")
    Tc(results, lattice_sizes, "Chi")
    gamma_nu(results, lattice_sizes)
    chi_scaling(results,lattice_sizes)
    beta_nu(results, lattice_sizes)
    M_scaling(results, lattice_sizes)
    C_div(results, lattice_sizes)
    entropy(results, lattice_sizes)
    free_energy(results,lattice_sizes)

if __name__ == "__main__":
    main()
