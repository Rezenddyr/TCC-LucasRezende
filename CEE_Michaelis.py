import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import math
import os
import time

METODO = "ES_Michaelis"


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


IND_SIZE     = 12
TAU_SIZE     = 5
KM_SIZE      = 7

MIN_TAU      = 0.1
MAX_TAU      = 5.0
MIN_KM       = 0.01
MAX_KM       = 500.0
MIN_STRATEGY = 0.1
MAX_STRATEGY = 10.0

POPULACAO      = []
APTIDAO        = []
APTIDAO_FILHOS = []


def twoBody(y, t, tauA, tauB, tauC, tauD, tauE,
                   Km_AB, Km_BC, Km_BE, Km_CD, Km_DE, Km_EA, Km_EE):
    ydot = np.empty((5,))

    NA = y[0] / maximo_A
    NB = y[1] / maximo_B
    NC = y[2] / maximo_C
    ND = y[3] / maximo_D
    NE = y[4] / maximo_E

    ydot[0] = (Km_EA * (1 - NE) / (Km_EA + NE) - NA) / tauA

    ydot[1] = (NA * (Km_AB + 1) / (Km_AB + NA) - NB) / tauB

    ydot[2] = (NB * (Km_BC + 1) / (Km_BC + NB) - NC) / tauC

    ydot[3] = (NC * (Km_CD + 1) / (Km_CD + NC) - ND) / tauD

    num_E = (ND * (Km_DE + 1) *
             (-NB * NE * (Km_BE + 1) * (Km_EE + 1)
              + NB * (Km_BE + 1) * (Km_EE + NE)
              + NE * (Km_BE + NB) * (Km_EE + 1)))
    den_E = (Km_BE + NB) * (Km_DE + ND) * (Km_EE + NE)
    ydot[4] = (num_E / den_E - NE) / tauE

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
            ind[5],  ind[6],  ind[7],  ind[8],  ind[9],  ind[10], ind[11])


def avalia(ind):
    p = extrai_params(ind)
    sol = odeint(twoBody, Y0, dobra_pontos,
                 args=(p[0], p[1], p[2], p[3], p[4],
                       p[5], p[6], p[7], p[8], p[9], p[10], p[11]))
    pA, pB, pC, pD, pE = organiza_pontos(sol)
    return calcula_diferenca(pA, pB, pC, pD, pE)



def cria_individuo():
    ind = []
    for _ in range(TAU_SIZE):
        ind.append(r.uniform(MIN_TAU, MAX_TAU))
    for _ in range(KM_SIZE):
        ind.append(r.uniform(MIN_KM, MAX_KM))
    for _ in range(IND_SIZE):
        ind.append(r.uniform(MIN_STRATEGY, MAX_STRATEGY))
    return ind



def mutESLogNormal(ind, c, indpb):
    t  = c / math.sqrt(2.0 * math.sqrt(IND_SIZE))
    t0 = c / math.sqrt(2.0 * IND_SIZE)
    n  = r.gauss(0, 1)
    t0_n = t0 * n

    for indx in range(IND_SIZE):
        if r.random() < indpb:
            s_idx = indx + IND_SIZE
            s_old = copy.deepcopy(ind[s_idx])
            v_old = copy.deepcopy(ind[indx])

            if indx < TAU_SIZE:
                lo, hi = MIN_TAU, MAX_TAU
            else:
                lo, hi = MIN_KM, MAX_KM

            tentativas = 0
            while True:
                ind[s_idx] = s_old * math.exp(t0_n + t * r.gauss(0, 1))
                ind[s_idx] = min(ind[s_idx], MAX_STRATEGY)
                ind[indx]  = v_old * ind[s_idx] * r.gauss(0, 1)
                tentativas += 1
                if lo <= ind[indx] <= hi or tentativas > 50:
                    if not (lo <= ind[indx] <= hi):
                        ind[s_idx] = s_old
                        ind[indx]  = v_old
                    break

    return ind


def varOr(populacao, lambda_):
    offspring = []
    for _ in range(lambda_):
        pai = copy.deepcopy(r.choice(populacao))
        filho = mutESLogNormal(pai, 1.0, 0.03)
        offspring.append(filho)
    return offspring


def selTournament(offspring, mu, tournsize):
    indices_validos = list(range(len(APTIDAO_FILHOS)))
    chosen, chosen_apt = [], []
    for _ in range(mu):
        aspirants = r.sample(indices_validos, min(tournsize, len(indices_validos)))
        aptidoes  = [APTIDAO_FILHOS[i] for i in aspirants]
        menor_erro = min(aptidoes)
        idx_melhor = aspirants[aptidoes.index(menor_erro)]
        chosen.append(offspring[idx_melhor])
        chosen_apt.append(menor_erro)
    return chosen, chosen_apt



def plota_resultados(ind, pasta):
    p = extrai_params(ind)
    sol = odeint(twoBody, Y0, dobra_pontos,
                 args=(p[0], p[1], p[2], p[3], p[4],
                       p[5], p[6], p[7], p[8], p[9], p[10], p[11]))
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
    caminho = os.path.join(pasta, f'graficos_{METODO}.png')
    plt.savefig(caminho, dpi=300)
    plt.show()
    print(f"Gráfico salvo em: {caminho}")



def main():
    seed = int(time.time())
    r.seed(seed)
    MU, LAMBDA = 15, 105

    pasta = METODO
    os.makedirs(pasta, exist_ok=True)

    arquivo_resultados = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')

    inicio = time.time()
    with open(arquivo_resultados, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  Km=[{MIN_KM}, {MAX_KM}]\n")

    for _ in range(MU):
        POPULACAO.append(cria_individuo())
    for ind in POPULACAO:
        APTIDAO.append(avalia(ind))

    for gen in range(10001):
        offspring = varOr(POPULACAO, LAMBDA)

        APTIDAO_FILHOS.clear()
        for ind in offspring:
            dif = avalia(ind)
            if not math.isnan(dif):
                APTIDAO_FILHOS.append(dif)

        novos_pais, novas_aptidoes = selTournament(offspring, MU, 3)

        POPULACAO.clear()
        POPULACAO.extend(novos_pais)
        APTIDAO.clear()
        APTIDAO.extend(novas_aptidoes)

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
    horas   = int(tempo_total // 3600)
    minutos = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"\n--- RESULTADO FINAL ---")
    print(f"Erro (norma-1): {menor_valor}")
    print(f"Parâmetros: {POPULACAO[indice_menor][:IND_SIZE]}")
    print(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")

    with open(arquivo_resultados, "a") as f:
        f.write(f"\n--- RESULTADO FINAL ---\n")
        f.write(f"Erro (norma-1): {menor_valor}\n")
        f.write(f"Parâmetros: {POPULACAO[indice_menor][:IND_SIZE]}\n")
        f.write(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s\n")

    plota_resultados(POPULACAO[indice_menor], pasta)


if __name__ == "__main__":
    main()
