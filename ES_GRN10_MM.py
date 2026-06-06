import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import math
import os
import time

METODO = "ES_GRN10_MM"

# ---------------------------------------------------------------------------
# Rede GRN10 - 10 genes (A..J) - cinetica de Michaelis-Menten (MM)
# Estrategia Evolutiva (mu, lambda) com auto-adaptacao log-normal.
#
# Topologia (regulador -> alvo):
#   A <- J (repressao)         B <- E
#   C <- B, F, A (logico)      D <- F
#   E <- J (repressao)         F <- A
#   G <- B, F, A (logico)      H <- F
#   I <- G E H (produto)       J <- I
# ---------------------------------------------------------------------------

arquivo = open("GRN10.txt", 'r')
x = []
cols = [[] for _ in range(10)]
for linha in arquivo:
    e = linha.split()
    if len(e) < 11:
        continue
    x.append(float(e[0]))
    for i in range(10):
        cols[i].append(float(e[i + 1]))
arquivo.close()

ORIG = [copy.deepcopy(c) for c in cols]
MAXS = np.array([max(c) for c in cols])
dobra_pontos = copy.deepcopy(x)
Y0 = [c[0] for c in cols]

NOMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

AJ, BE, CB, CF, CA, DF, EJ, FA, GB, GF, GA, HF, IG, IH, JI = range(15)

# tau(10) + k(15) + vmax(10) = 35
IND_SIZE = 35
MIN_STRATEGY = 0.1
MAX_STRATEGY = 10.0

POPULACAO      = []
APTIDAO        = []
APTIDAO_FILHOS = []


def get_bounds(indx):
    if indx < 10:   return 0.1,   5.0     # tau (10)
    if indx < 25:   return 0.001, 0.999   # k   (15)
    return 1.0, 10.0                        # vmax (10)


def mm(v, k):
    if v <= 0.0:
        return 0.0
    return v / (v + k)


def sum6(a, b, c):
    return (a * (1 - b) * (1 - c) + (1 - a) * b * (1 - c) +
            (1 - a) * (1 - b) * c + a * (1 - b) * c +
            (1 - a) * b * c + a * b * c)


def twoBody(y, t, p):
    tau  = p[0:10]
    k    = p[10:25]
    vmax = p[25:35]

    N = y / MAXS

    def M(e, val):
        return mm(val, k[e])

    ydot = np.empty((10,))
    ydot[0] = (vmax[0] * (1.0 - M(AJ, N[9])) - N[0]) / tau[0]
    ydot[1] = (vmax[1] * M(BE, N[4]) - N[1]) / tau[1]
    ydot[2] = (vmax[2] * sum6(M(CB, N[1]), M(CF, N[5]), M(CA, N[0])) - N[2]) / tau[2]
    ydot[3] = (vmax[3] * M(DF, N[5]) - N[3]) / tau[3]
    ydot[4] = (vmax[4] * (1.0 - M(EJ, N[9])) - N[4]) / tau[4]
    ydot[5] = (vmax[5] * M(FA, N[0]) - N[5]) / tau[5]
    ydot[6] = (vmax[6] * sum6(M(GB, N[1]), M(GF, N[5]), M(GA, N[0])) - N[6]) / tau[6]
    ydot[7] = (vmax[7] * M(HF, N[5]) - N[7]) / tau[7]
    ydot[8] = (vmax[8] * M(IG, N[6]) * M(IH, N[7]) - N[8]) / tau[8]
    ydot[9] = (vmax[9] * M(JI, N[8]) - N[9]) / tau[9]
    return ydot


def organiza_pontos(sol):
    return [list(sol[:, i]) for i in range(10)]


def calcula_diferenca(preds):
    dif = 0.0
    for i in range(10):
        oi, pi = ORIG[i], preds[i]
        for j in range(len(pi)):
            dif += abs(oi[j] - pi[j])
    return dif


def avalia(ind):
    try:
        sol = odeint(twoBody, Y0, dobra_pontos,
                     args=(np.asarray(ind[:IND_SIZE]),))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        return calcula_diferenca(organiza_pontos(sol))
    except Exception:
        return float('inf')


def cria_individuo():
    ind = [r.uniform(*get_bounds(i)) for i in range(IND_SIZE)]
    ind += [r.uniform(MIN_STRATEGY, MAX_STRATEGY) for _ in range(IND_SIZE)]
    return ind


def mutESLogNormal(ind, c, indpb):
    t    = c / math.sqrt(2.0 * math.sqrt(IND_SIZE))
    t0   = c / math.sqrt(2.0 * IND_SIZE)
    t0_n = t0 * r.gauss(0, 1)
    for indx in range(IND_SIZE):
        if r.random() < indpb:
            s_idx = indx + IND_SIZE
            s_old, v_old = ind[s_idx], ind[indx]
            lo, hi = get_bounds(indx)
            tentativas = 0
            while True:
                ind[s_idx] = min(s_old * math.exp(t0_n + t * r.gauss(0, 1)), MAX_STRATEGY)
                ind[indx]  = v_old * ind[s_idx] * r.gauss(0, 1)
                tentativas += 1
                if lo <= ind[indx] <= hi or tentativas > 50:
                    if not (lo <= ind[indx] <= hi):
                        ind[s_idx], ind[indx] = s_old, v_old
                    break
    return ind


def varOr(populacao, lambda_):
    offspring = []
    for _ in range(lambda_):
        filho = mutESLogNormal(copy.deepcopy(r.choice(populacao)), 1.0, 0.03)
        offspring.append(filho)
    return offspring


def selTournament(offspring, mu, tournsize):
    indices = list(range(len(APTIDAO_FILHOS)))
    chosen, chosen_apt = [], []
    for _ in range(mu):
        asp   = r.sample(indices, min(tournsize, len(indices)))
        apts  = [APTIDAO_FILHOS[i] for i in asp]
        menor = min(apts)
        chosen.append(offspring[asp[apts.index(menor)]])
        chosen_apt.append(menor)
    return chosen, chosen_apt


def plota_resultados(ind, pasta, seed):
    sol = odeint(twoBody, Y0, dobra_pontos, args=(np.asarray(ind[:IND_SIZE]),))
    preds = organiza_pontos(sol)

    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f'Resultados - {METODO}', fontsize=14)
    for idx in range(10):
        ax = axs[idx // 5][idx % 5]
        ax.plot(x, preds[idx], label=f'{NOMES[idx]} predito')
        ax.plot(x, ORIG[idx],  label=f'{NOMES[idx]} real')
        ax.set_title(f'Variavel {NOMES[idx]}')
        ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Concentracao')
        ax.legend()

    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Grafico salvo em: {caminho}")


def main(seed):
    POPULACAO.clear(); APTIDAO.clear(); APTIDAO_FILHOS.clear()
    r.seed(seed)
    MU, LAMBDA = 15, 105

    pasta = METODO
    os.makedirs(pasta, exist_ok=True)
    arq_res = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')
    inicio  = time.time()
    with open(arq_res, "w") as f:
        f.write(f"SEED: {seed}\nMETODO: {METODO}\nIND_SIZE: {IND_SIZE}\n")
        f.write(f"BOUNDS: tau=[0.1, 5.0]  K=[0.001, 0.999]  Vmax=[1.0, 10.0]\n")

    for _ in range(MU):
        POPULACAO.append(cria_individuo())
    for ind in POPULACAO:
        APTIDAO.append(avalia(ind))

    for gen in range(10001):
        offspring = varOr(POPULACAO, LAMBDA)
        APTIDAO_FILHOS.clear()
        for ind in offspring:
            APTIDAO_FILHOS.append(avalia(ind))
        novos_pais, novas_aptidoes = selTournament(offspring, MU, 3)
        POPULACAO.clear(); POPULACAO.extend(novos_pais)
        APTIDAO.clear();   APTIDAO.extend(novas_aptidoes)

        if gen % 100 == 0:
            mv = min(APTIDAO); mi = APTIDAO.index(mv)
            print(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi][:IND_SIZE]}\nAPTIDAO: \n{mv}")
            with open(arq_res, "a") as f:
                f.write(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi][:IND_SIZE]}\nAPTIDAO: \n{mv}\n")

    mv = min(APTIDAO); mi = APTIDAO.index(mv)
    tt = time.time() - inicio
    msg = (f"\n--- RESULTADO FINAL ---\nErro (norma-1): {mv}\n"
           f"Parametros: {POPULACAO[mi][:IND_SIZE]}\n"
           f"Tempo total: {int(tt//3600):02d}h {int((tt%3600)//60):02d}m {int(tt%60):02d}s\n")
    print(msg)
    with open(arq_res, "a") as f:
        f.write(msg)
    plota_resultados(POPULACAO[mi], pasta, seed)


SEEDS = [
    1778434285, 1778461231, 1778490666, 1778578247, 1778663936,
    1778719796, 1778749666, 1778837425, 1778893788, 1778981565,
]

if __name__ == "__main__":
    for s in SEEDS:
        main(s)
