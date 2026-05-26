import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import math
import os
import time

METODO = "DE_GK"

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

# tau(5) + v2(7) + J(14) = 26
TAU_SIZE = 5
V2_SIZE  = 7
J_SIZE   = 14

MIN_TAU = 0.1
MAX_TAU = 5.0
MIN_V2  = 0.01
MAX_V2  = 1.5
MIN_J   = 0.001
MAX_J   = 0.99

BOUNDS = (
    [(MIN_TAU, MAX_TAU)] * TAU_SIZE +
    [(MIN_V2,  MAX_V2)]  * V2_SIZE  +
    [(MIN_J,   MAX_J)]   * J_SIZE
)


def GK(v1, v2, J1, J2):
    if v1 <= 0.0:
        return 0.0
    B = v2 - v1 + v1 * J2 + v2 * J1
    disc = B * B - 4.0 * (v2 - v1) * v1 * J2
    if disc < 0.0:
        disc = 0.0
    denom = B + math.sqrt(disc)
    if denom <= 1e-12:
        return 1.0
    return 2.0 * v1 * J2 / denom


def twoBody(y, t,
            tauA, tauB, tauC, tauD, tauE,
            v2_EA, J1_EA, J2_EA,
            v2_AB, J1_AB, J2_AB,
            v2_BC, J1_BC, J2_BC,
            v2_CD, J1_CD, J2_CD,
            v2_BE, J1_BE, J2_BE,
            v2_DE, J1_DE, J2_DE,
            v2_EE, J1_EE, J2_EE):

    NA = y[0] / maximo_A
    NB = y[1] / maximo_B
    NC = y[2] / maximo_C
    ND = y[3] / maximo_D
    NE = y[4] / maximo_E

    ydot = np.empty((5,))

    ydot[0] = (1.0 - GK(NE, v2_EA, J1_EA, J2_EA) - NA) / tauA
    ydot[1] = (GK(NA, v2_AB, J1_AB, J2_AB) - NB) / tauB
    ydot[2] = (GK(NB, v2_BC, J1_BC, J2_BC) - NC) / tauC
    ydot[3] = (GK(NC, v2_CD, J1_CD, J2_CD) - ND) / tauD

    GKB = GK(NB, v2_BE, J1_BE, J2_BE)
    GKD = GK(ND, v2_DE, J1_DE, J2_DE)
    GKE = GK(NE, v2_EE, J1_EE, J2_EE)
    ydot[4] = (GKB * GKD + GKD * GKE - NE) / tauE

    return ydot


def organiza_pontos(solucao):
    pA, pB, pC, pD, pE = [], [], [], [], []
    for pontos in range(len(solucao)):
        pA.append(solucao[pontos][0])
        pB.append(solucao[pontos][1])
        pC.append(solucao[pontos][2])
        pD.append(solucao[pontos][3])
        pE.append(solucao[pontos][4])
    return pA, pB, pC, pD, pE


def calcula_diferenca(pA, pB, pC, pD, pE):
    difTotal = 0
    for i in range(len(pA)):
        difTotal += abs(A_ORIGINAL[i] - pA[i])
        difTotal += abs(B_ORIGINAL[i] - pB[i])
        difTotal += abs(C_ORIGINAL[i] - pC[i])
        difTotal += abs(D_ORIGINAL[i] - pD[i])
        difTotal += abs(E_ORIGINAL[i] - pE[i])
    return difTotal


def extrai_params(ind):
    return (
        ind[0],  ind[1],  ind[2],  ind[3],  ind[4],
        ind[5],  ind[12], ind[13],
        ind[6],  ind[14], ind[15],
        ind[7],  ind[16], ind[17],
        ind[8],  ind[18], ind[19],
        ind[9],  ind[20], ind[21],
        ind[10], ind[22], ind[23],
        ind[11], ind[24], ind[25],
    )


def avalia(ind):
    try:
        p = extrai_params(ind)
        sol = odeint(twoBody, Y0, dobra_pontos, args=p)
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        pA, pB, pC, pD, pE = organiza_pontos(sol)
        return calcula_diferenca(pA, pB, pC, pD, pE)
    except Exception:
        return float('inf')


def plota_resultados(ind, pasta, seed):
    p = extrai_params(ind)
    sol = odeint(twoBody, Y0, dobra_pontos, args=p)
    pA, pB, pC, pD, pE = organiza_pontos(sol)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Resultados — {METODO}', fontsize=14)

    dados = [(pA, A_ORIGINAL, 'A'), (pB, B_ORIGINAL, 'B'),
             (pC, C_ORIGINAL, 'C'), (pD, D_ORIGINAL, 'D'),
             (pE, E_ORIGINAL, 'E')]

    for idx, (pred, orig, nome) in enumerate(dados):
        ax = axs[idx // 3][idx % 3]
        ax.plot(x, pred, label=f'{nome} predito')
        ax.plot(x, orig, label=f'{nome} real')
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
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  v2=[{MIN_V2}, {MAX_V2}]  J=[{MIN_J}, {MAX_J}]\n")

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
