import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

# Função para definir os parâmetros do problema
def test_3():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds
    rand = random.random()  # random [0,1]
    population = 100
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female
    y = "-100*(z[1]-z[0]**2)**2 - (1 - z[0]**2)**2"
    n = 2
    bounds = np.array([[-100, 100],
                       [-100, 100]])
    lim = 200
    pf = 0.7

# Resultados armazenados em uma lista de tuplas (fitness, solução)
results = [
    (-65.46893430797135, [2.69000086, 6.72054731]),
    (-75758.21053478548, [10.09693194, 80.08705662]),
    (-67.77083084876658, [-1.46830799, 2.89533613]),
    (-3665.149265526493, [-7.44438689, 52.89198555]),
    (-4244.255310504946, [7.73677802, 60.82171371]),
    (-8293.736122395556, [2.7855888, -0.33371964]),
    (-429.36703687059065, [-4.61161814, 21.69847129]),
    (-2993.4931628912586, [7.4625918, 55.53321478]),
    (-14.110983841460818, [2.05098232, 4.64032872]),
    (-9401.939851045896, [-6.22359373, 43.94147606]),
    (-5111.8892863307565, [3.25174181, 20.30071572]),
    (-45193.52363077942, [-4.28968905, -2.78599665]),
    (-7159.476133548059, [8.93984991, 80.53386525]),
    (-1151.960932120084, [5.3451441, 31.47106577]),
    (-3294.2995415998926, [-0.24110327, -5.57917422]),
    (-2060.4602445445194, [-3.99508374, 19.42621098]),
    (-3839.627959556711, [-7.83826885, 58.16158391]),
    (-6588.976859928442, [8.74311544, 79.49658051]),
    (-25943.167358430655, [-8.92716183, 91.65880012]),
    (-5817.117039921438, [6.99416386, 54.62910482]),
    (-4508.966207676642, [-3.5270114, 4.96960709]),
    (-503.03569976454247, [3.21126268, 8.30219032]),
    (-1094.2077137753429, [-3.81696269, 11.55244362]),
    (-29864.789515128115, [-1.4721949, -15.11368]),
    (-116.15430178252487, [-0.67211892, -1.06114965]),
    (-1372.81687628471, [6.23069655, 38.29591512]),
    (-37893.315684578774, [-7.18949266, 33.18677379]),
    (-24.68688954483435, [-2.37615307, 6.05496341]),
    (-11261.075841267295, [1.68764685, -7.06112414]),
    (-8573.914988833054, [7.10311007, 60.6893102]),
    (-3686.0072577503734, [1.5645646, 8.4303676]),
    (-7377.865060383277, [4.48107223, 12.37775482]),
    (-3087.462429036814, [-5.46065532, 35.18171365]),
    (-42795.0205744262, [10.58658338, 98.76641388]),
    (-87577.69393726377, [9.95955487, 72.46101438]),
    (-502.3535375324368, [0.95423122, -1.34430264]),
    (-900.1622970610399, [-3.22454731, 13.24699552]),   
    (-96.35968785008879, [-0.04738652, -0.87462346]),
    (-53719.05015352091, [7.30585647, 33.93004245]),
    (-8523.597636001998, [1.96108694, -5.62054415]),
    (-358.61687570179816, [2.79097861, 6.53218043]),
    (-16.834674626798567, [-1.5884783, 1.65447895]),
    (-13.38881303841319, [1.56672603, 2.12218491]),
    (-29.75716639914962, [-1.68439886, 3.13936187]),
    (-1120.8760877833558, [5.35506867, 29.44653946]),
    (-2396.969079738671, [4.37234247, 17.40630668]),
    (-1750.9008898589739, [-1.14695867, 6.29724952]),
    (-740.76895344748, [-3.6661054, 14.09360555]),
    (-82363.71104556629, [4.36201785, 47.75182306]),
    (-15571.690270657225, [-10.09236872, 95.0206371]),
    (-30718.547768029537, [-3.85720852, -2.54384416])
]

# Separando os resultados em listas de fitness e soluções
fitness_values = [result[0] for result in results]
solutions = [result[1] for result in results]

# Criando o gráfico
plt.figure(figsize=(10, 6))
sns.histplot(fitness_values, kde=True, bins=10)
plt.title('Distribuição dos Valores de Fitness')
plt.xlabel('Fitness')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# Criando o gráfico
plt.figure(figsize=(10, 6))
sns.histplot(fitness_values, kde=True, bins=10)
plt.title('Distribuição dos Valores de Fitness')
plt.xlabel('Fitness')
plt.ylabel('Frequência')
plt.grid(True)
plt.axvline(x=10, color='r', linestyle='--', label='10 segundos')
plt.legend()
plt.show()