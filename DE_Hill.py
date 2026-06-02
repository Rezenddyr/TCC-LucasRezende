import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import math
import os
import time

METODO = "DE_Hill"

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

# tau(5) + k(7) + n(7) + vmax(5) = 24
IND_SIZE  = 24
TAU_SIZE  = 5
K_SIZE    = 7
N_SIZE    = 7
VMAX_SIZE = 5

MIN_TAU  = 0.1
MAX_TAU  = 5.0
MIN_K    = 0.1
MAX_K    = 1.0
MIN_N    = 1.0
MAX_N    = 25.0
MIN_VMAX = 0.1
MAX_VMAX = 2.0

# Parâmetros da Evolução Diferencial
NP = 50   # tamanho da população
F  = 0.8  # fator de escala (mutação)
CR = 0.9  # taxa de cruzamento (crossover)

POPULACAO = []
APTIDAO   = []


def get_bounds(indx):
    if indx < TAU_SIZE:
        return MIN_TAU, MAX_TAU
    elif indx < TAU_SIZE + K_SIZE:
        return MIN_K, MAX_K
    elif indx < TAU_SIZE + K_SIZE + N_SIZE:
        return MIN_N, MAX_N
    else:
        return MIN_VMAX, MAX_VMAX


def twoBody(y, t, tauA, kA, nA, vmaxA,
                   tauB, kB, nB, vmaxB,
                   tauC, kC, nC, vmaxC,
                   tauD, kD, nD, vmaxD,
                   tauE, kEB, kED, kEE, nEB, nED, nEE, vmaxE):
    ydot = np.empty((5,))

    ydot[0] = (vmaxA * (1 - (pow(y[4]/maximo_E, nA) / (pow(y[4]/maximo_E, nA) + pow(kA, nA))))
               - y[0]/maximo_A) / tauA

    ydot[1] = (vmaxB * (pow(y[0]/maximo_A, nB) / (pow(y[0]/maximo_A, nB) + pow(kB, nB)))
               - y[1]/maximo_B) / tauB

    ydot[2] = (vmaxC * (pow(y[1]/maximo_B, nC) / (pow(y[1]/maximo_B, nC) + pow(kC, nC)))
               - y[2]/maximo_C) / tauC

    ydot[3] = (vmaxD * (pow(y[2]/maximo_C, nD) / (pow(y[2]/maximo_C, nD) + pow(kD, nD)))
               - y[3]/maximo_D) / tauD

    HB = pow(y[1]/maximo_B, nEB) / (pow(y[1]/maximo_B, nEB) + pow(kEB, nEB))
    HD = pow(y[3]/maximo_D, nED) / (pow(y[3]/maximo_D, nED) + pow(kED, nED))
    HE = pow(y[4]/maximo_E, nEE) / (pow(y[4]/maximo_E, nEE) + pow(kEE, nEE))
    ydot[4] = (vmaxE * ((HB * HD) + (HD * HE)) - y[4]/maximo_E) / tauE

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
    return (ind[0],  ind[1],  ind[2],  ind[3],  ind[4],
            ind[5],  ind[6],  ind[7],  ind[8],
            ind[9],  ind[10], ind[11],
            int(ind[12]), int(ind[13]), int(ind[14]), int(ind[15]),
            int(ind[16]), int(ind[17]), int(ind[18]),
            ind[19], ind[20], ind[21], ind[22], ind[23])


def avalia(ind):
    try:
        p = extrai_params(ind)
        sol = odeint(twoBody, Y0, dobra_pontos,
                     args=(p[0], p[5], p[12], p[19],
                           p[1], p[6], p[13], p[20],
                           p[2], p[7], p[14], p[21],
                           p[3], p[8], p[15], p[22],
                           p[4], p[9], p[10], p[11], p[16], p[17], p[18], p[23]))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        pA, pB, pC, pD, pE = organiza_pontos(sol)
        return calcula_diferenca(pA, pB, pC, pD, pE)
    except Exception:
        return float('inf')


def cria_individuo():
    return [r.uniform(*get_bounds(i)) for i in range(IND_SIZE)]


def mutacao_de(idx_alvo):
    # DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
    indices = list(range(NP))
    indices.remove(idx_alvo)
    r1, r2, r3 = r.sample(indices, 3)
    mutante = []
    for j in range(IND_SIZE):
        lo, hi = get_bounds(j)
        v = POPULACAO[r1][j] + F * (POPULACAO[r2][j] - POPULACAO[r3][j])
        mutante.append(max(lo, min(hi, v)))
    return mutante


def cruzamento_de(alvo, mutante):
    # Cruzamento binomial: garante ao menos um gene do mutante (j_rand)
    trial = copy.deepcopy(alvo)
    j_rand = r.randint(0, IND_SIZE - 1)
    for j in range(IND_SIZE):
        if r.random() < CR or j == j_rand:
            trial[j] = mutante[j]
    return trial


def plota_resultados(ind, pasta, seed):
    p = extrai_params(ind)
    sol = odeint(twoBody, Y0, dobra_pontos,
                 args=(p[0], p[5], p[12], p[19],
                       p[1], p[6], p[13], p[20],
                       p[2], p[7], p[14], p[21],
                       p[3], p[8], p[15], p[22],
                       p[4], p[9], p[10], p[11], p[16], p[17], p[18], p[23]))
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
    plt.close()
    print(f"Gráfico salvo em: {caminho}")


def main(seed):
    r.seed(seed)
    POPULACAO.clear()
    APTIDAO.clear()

    pasta = METODO
    os.makedirs(pasta, exist_ok=True)

    arquivo_resultados = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')

    inicio = time.time()
    with open(arquivo_resultados, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"NP: {NP}  F: {F}  CR: {CR}\n")
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  K=[{MIN_K}, {MAX_K}]"
                f"  N=[{MIN_N}, {MAX_N}]  Vmax=[{MIN_VMAX}, {MAX_VMAX}]\n")

    for _ in range(NP):
        POPULACAO.append(cria_individuo())
    for ind in POPULACAO:
        APTIDAO.append(avalia(ind))

    for gen in range(10001):
        for i in range(NP):
            mutante = mutacao_de(i)
            trial   = cruzamento_de(POPULACAO[i], mutante)
            apt_trial = avalia(trial)
            # seleção gulosa: substitui apenas se o trial for melhor
            if apt_trial <= APTIDAO[i]:
                POPULACAO[i] = trial
                APTIDAO[i]   = apt_trial

        if gen % 100 == 0:
            menor_valor  = min(APTIDAO)
            indice_menor = APTIDAO.index(menor_valor)
            print(f"GEN: {gen}")
            print(f"Individuo: \n{POPULACAO[indice_menor]}")
            print(f"APTIDAO: \n{menor_valor}")
            with open(arquivo_resultados, "a") as f:
                f.write(f"GEN: {gen}\nIndividuo: \n{POPULACAO[indice_menor]}\nAPTIDAO: \n{menor_valor}\n")

    menor_valor  = min(APTIDAO)
    indice_menor = APTIDAO.index(menor_valor)

    tempo_total = time.time() - inicio
    horas    = int(tempo_total // 3600)
    minutos  = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"\n--- RESULTADO FINAL ---")
    print(f"Erro (norma-1): {menor_valor}")
    print(f"Parâmetros: {POPULACAO[indice_menor]}")
    print(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")

    with open(arquivo_resultados, "a") as f:
        f.write(f"\n--- RESULTADO FINAL ---\n")
        f.write(f"Erro (norma-1): {menor_valor}\n")
        f.write(f"Parâmetros: {POPULACAO[indice_menor]}\n")
        f.write(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s\n")

    plota_resultados(POPULACAO[indice_menor], pasta, seed)


SEEDS = [
    1778434285, 1778461231, 1778490666, 1778578247, 1778663936,
    1778719796, 1778749666, 1778837425, 1778893788, 1778981565,
]

if __name__ == "__main__":
    for s in SEEDS:
        main(s)
