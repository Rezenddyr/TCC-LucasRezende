import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import os
import time

METODO = "DE_MM"

arquivo = open("GRN5.txt", 'r')
x, A, B, C, D, E = [], [], [], [], [], []
for linha in arquivo:
    elementos = linha.split()
    x.append(float(elementos[0].strip()))
    A.append(float(elementos[1].strip()))
    B.append(float(elementos[2].strip()))
    C.append(float(elementos[3].strip()))
    D.append(float(elementos[4].strip()))
    E.append(float(elementos[5].strip()))
arquivo.close()

maximo_A = max(A)
maximo_B = max(B)
maximo_C = max(C)
maximo_D = max(D)
maximo_E = max(E)

A_ORIGINAL = copy.deepcopy(A)
B_ORIGINAL = copy.deepcopy(B)
C_ORIGINAL = copy.deepcopy(C)
D_ORIGINAL = copy.deepcopy(D)
E_ORIGINAL = copy.deepcopy(E)

dobra_pontos = copy.deepcopy(x)
Y0 = [A[0], B[0], C[0], D[0], E[0]]

# tau(5) + k(7) + vmax(5) = 17
IND_SIZE  = 17
TAU_SIZE  = 5
K_SIZE    = 7
VMAX_SIZE = 5

BOUNDS = (
    [(0.1,   5.0)]   * TAU_SIZE  +
    [(0.001, 0.999)] * K_SIZE    +
    [(1.0,   10.0)]  * VMAX_SIZE
)


def mm(v, k):
    if v <= 0.0:
        return 0.0
    return v / (v + k)


def twoBody(y, t,
            tauA, tauB, tauC, tauD, tauE,
            kA, kB, kC, kD, kEB, kED, kEE,
            vmaxA, vmaxB, vmaxC, vmaxD, vmaxE):

    ydot = np.empty((5,))

    ydot[0] = (vmaxA * (1 - mm(y[4]/maximo_E, kA)) - y[0]/maximo_A) / tauA
    ydot[1] = (vmaxB * mm(y[0]/maximo_A, kB)       - y[1]/maximo_B) / tauB
    ydot[2] = (vmaxC * mm(y[1]/maximo_B, kC)       - y[2]/maximo_C) / tauC
    ydot[3] = (vmaxD * mm(y[2]/maximo_C, kD)       - y[3]/maximo_D) / tauD

    HB = mm(y[1]/maximo_B, kEB)
    HD = mm(y[3]/maximo_D, kED)
    HE = mm(y[4]/maximo_E, kEE)
    ydot[4] = (vmaxE * ((HB * HD) + (HD * HE)) - y[4]/maximo_E) / tauE

    return ydot


def organiza_pontos(sol):
    pA, pB, pC, pD, pE = [], [], [], [], []
    for pt in sol:
        pA.append(pt[0]); pB.append(pt[1]); pC.append(pt[2])
        pD.append(pt[3]); pE.append(pt[4])
    return pA, pB, pC, pD, pE


def calcula_diferenca(pA, pB, pC, pD, pE):
    dif = 0
    for i in range(len(pA)):
        dif += abs(A_ORIGINAL[i] - pA[i])
        dif += abs(B_ORIGINAL[i] - pB[i])
        dif += abs(C_ORIGINAL[i] - pC[i])
        dif += abs(D_ORIGINAL[i] - pD[i])
        dif += abs(E_ORIGINAL[i] - pE[i])
    return dif


def avalia(ind):
    try:
        sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        return calcula_diferenca(*organiza_pontos(sol))
    except Exception:
        return float('inf')


def plota_resultados(ind, pasta, seed):
    sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind))
    pA, pB, pC, pD, pE = organiza_pontos(sol)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Resultados — {METODO}', fontsize=14)

    dados = [(pA, A_ORIGINAL, 'A'), (pB, B_ORIGINAL, 'B'),
             (pC, C_ORIGINAL, 'C'), (pD, D_ORIGINAL, 'D'),
             (pE, E_ORIGINAL, 'E')]

    for idx, (pred, orig, nome) in enumerate(dados):
        ax = axs[idx // 3][idx % 3]
        ax.plot(x, pred, label=f'{nome} predito')
        ax.plot(x, orig,  label=f'{nome} real')
        ax.set_title(f'Variável {nome}')
        ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Concentração')
        ax.legend()

    axs[1][2].axis('off')
    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300)
    plt.show()
    print(f"Gráfico salvo em: {caminho}")


def main():
    seed = int(time.time())

    pasta = METODO
    os.makedirs(pasta, exist_ok=True)

    arquivo_resultados = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')

    inicio = time.time()
    with open(arquivo_resultados, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"strategy=best1bin  popsize=15  F=0.8  CR=0.75  polish=True\n")
        f.write(f"BOUNDS: tau=[0.1, 5.0]  K=[0.001, 0.999]  Vmax=[1.0, 10.0]\n")

    gen_log = [0]

    def callback(xk, convergence=None):
        gen_log[0] += 1
        if gen_log[0] % 100 == 0:
            apt = avalia(xk)
            print(f"GEN: {gen_log[0]}")
            print(f"Individuo: \n{list(xk)}")
            print(f"APTIDAO: \n{apt}")
            with open(arquivo_resultados, "a") as f:
                f.write(f"GEN: {gen_log[0]}\nIndividuo: \n{list(xk)}\nAPTIDAO: \n{apt}\n")

    result = differential_evolution(
        avalia,
        BOUNDS,
        strategy='best1bin',
        popsize=15,
        mutation=0.8,
        recombination=0.75,
        polish=True,
        seed=seed,
        callback=callback,
        maxiter=10000,
        tol=0,
        updating='immediate'
    )

    melhor_ind  = list(result.x)
    menor_valor = result.fun

    tempo_total = time.time() - inicio
    horas    = int(tempo_total // 3600)
    minutos  = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"\n--- RESULTADO FINAL ---")
    print(f"Erro (norma-1): {menor_valor}")
    print(f"Parâmetros: {melhor_ind}")
    print(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")

    with open(arquivo_resultados, "a") as f:
        f.write(f"\n--- RESULTADO FINAL ---\n")
        f.write(f"Erro (norma-1): {menor_valor}\n")
        f.write(f"Parâmetros: {melhor_ind}\n")
        f.write(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s\n")

    plota_resultados(melhor_ind, pasta, seed)


if __name__ == "__main__":
    main()
