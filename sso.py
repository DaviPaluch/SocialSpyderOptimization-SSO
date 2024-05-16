import random
import numpy as np
from numpy import linalg as la
import math
import colored
from termcolor import colored
from statistics import median

# by default  algorithm is maximize

Male = True
Female = False

D = True
ND = False


class Spider:
    def __init__(self, s, s_next, weight, gender, fitness):
        self.s = s
        self.weight = weight
        self.fitness = fitness
        self.gender = gender
        self.s_next = s_next
        if gender:
            self.group = D

    def print_out(self):
        print("position = "+str(self.s))
        print("fitness = " + str(self.fitness))
        print("weight = " + str(self.weight))
        print("position next = "+str(self.s_next))
        print("fitness next = "+str(f(self.s_next)))
        if self.gender:
            print("Gender = Male")
            if self.group == D:
                print("Group = D")
            elif self.group == ND:
                print("Group = ND")
        else:
            print("Gender = Female")
        print()

    def set_group(self, group):
        self.group = group


def f(a):
    z = []
    z.extend(a)
    return eval(y)


def calculate_weight(fitness, best, worst):
    return (fitness - worst) / (best - worst)


def distance_euclidean(i, j):
    return la.norm(i - j, 2)


def distance_manhattan(pa, pb):
    return la.norm(pa - pb, 1)


def vibrations(spider_i, spider_j):
    return spider_j.weight * math.exp(-distance_euclidean(spider_i.s, spider_j.s) ** 2)


def maximum():  # return the spider with the best fitness
    maxi = spiders[0]
    for i in range(population):
        if spiders[i].fitness > maxi.fitness:
            maxi = spiders[i]
    return maxi


def minimum():  # return the spider with the worst fitness
    mini = spiders[0]
    for i in range(population):
        if spiders[i].fitness < mini.fitness:
            mini = spiders[i]
    return mini


def median_all_spiders():  # median fitness values of spiders
    increase = 0
    for i in range(population):
        increase = increase + spiders[i].fitness
    return increase / population


def probability():
    arr = np.array([0] * int(100 * pf) + [1] * int(100 - 100 * pf))
    np.random.shuffle(arr)
    rand = random.choice(arr)
    if rand == 0:
        return True
    else:
        return False


#  finds the nearest neighbor based on conditions
def nearest_spider(spider, treaty):
    near_distance = math.inf
    near_spider = spiders[0]
    for i in range(len(spiders)):
        if near_distance > distance_euclidean(spiders[i].s, spider.s) and not np.array_equal(spiders[i].s, spider.s):
            conditions = {}
            for j in range(len(treaty)):
                conditions[j] = eval(treaty[j])
            if all(conditions[j] for j in range(len(conditions))):
                near_spider = spiders[i]
                near_distance = distance_euclidean(spiders[i].s, spider.s)
    if near_distance == math.inf:
        near_spider = spider
    return near_spider


def type_1_female(fs, v_i_n, s_n, v_i_b, s_b, a, b, d, rand):
    return fs + a * v_i_n * (s_n - fs) + b * v_i_b * (s_b - fs) + d * (rand - 1/2)


def type_2_female(fs, v_i_n, s_n, v_i_b, s_b, a, b, d, rand):
    return fs - a * v_i_n * (s_n - fs) - b * v_i_b * (s_b - fs) + d * (rand - 1 / 2)


def type_1_male(ms, fs, v_m_f, a, d, rand):
    return ms + a * v_m_f * (fs - ms) + d * (rand - 1/2)


def type_2_male(ms, a):
    return ms + a * (weighted_mean_male() - ms)


# the weighted mean of the male spider population
def weighted_mean_male():
    total = np.array(n)
    total_weight = 0
    for x in range(population):
        if spiders[x].gender == Male:
            total = total + spiders[x].weight * spiders[x].s
            total_weight = total_weight + spiders[x].weight
    #print(total)
    #print(total_weight)
    return total / total_weight


def median_male_spider():
    return median([value.weight for value in spiders if value.gender == Male])


def total_weight_male():
    total = 0
    for x in range(population):
        total += spiders[x].weight
    return total


def update_fitness():
    for x in range(population):
        spiders[x].fitness = f(spiders[x].s)


def update_positions():
    for x in range(population):
        spiders[x].s = spiders[x].s_next


def update_weight(best, worst):
    for x in range(population):
        spiders[x].weight = calculate_weight(spiders[x].fitness, best, worst)
        #print("spider "+str(x)+" w = "+str(spiders[x].weight)+" s="+str(spiders[x].s))


def update_group(means):
    for x in range(population):
        if spiders[x].gender == Male:
            if spiders[x].weight > means:
                spiders[x].group = D
            elif spiders[x].weight <= means:
                spiders[x].group = ND


# mating radius
def radius():
    r = 0
    for i in range(n):
        r += bounds[i, 1] - bounds[i, 0]
    return r / (2 * n)


def seat(spider):
    for x in range(population):
        if np.array_equal(spiders[x].s, spider.s) and spiders[x].weight == spider.weight \
                and spiders[x].gender == spider.gender:
            return x


def create_population():
    for x in range(population):
        s = np.zeros(n)
        for x1 in range(n):
            s[x1] = np.random.uniform(bounds[x1, 0], bounds[x1, 1])
        if population_female > x:
            spiders.append(Spider(s, s, 0, Female, 0))
        else:
            spiders.append(Spider(s, s, 0, Male, 0))


def check(spider_new):
    for x in range(population):
        if np.array_equal(spider_new, spiders[x].s):
            return True
    return False


def social_spider_optimization():
    global spiders
    spiders = []
    create_population()
    number_of_iterations = 0
    max_fes = 10000 * D
    r = radius()
    max_error = 1e-8
    max_fes_reached = False
    error_achieved = False
    max_all = -np.inf
    max_s = np.ones(n)
    while not max_fes_reached and not error_achieved:
        update_positions()
        update_fitness()
        best = maximum()
        if max_all < best.fitness:
            max_all = best.fitness
            max_s = best.s
        worst = minimum()
        update_weight(best.fitness, worst.fitness)
        means = median_male_spider()
        update_group(means)
        for x in range(population):
            a = random.random()
            b = random.random()
            d = random.random()
            rand = random.random()
            if spiders[x].gender == Female:
                near_spider = nearest_spider(spiders[x], ["spiders[i].weight > spider.weight"])
                if probability():
                    spiders[x].s_next = type_1_female(spiders[x].s, vibrations(spiders[x], near_spider),
                                                      near_spider.s,
                                                      vibrations(spiders[x], best), best.s, a, b, d, rand)
                else:
                    spiders[x].s_next = type_2_female(spiders[x].s, vibrations(spiders[x], near_spider),
                                                      near_spider.s,
                                                      vibrations(spiders[x], best), best.s, a, b, d, rand)
            elif spiders[x].gender == Male:
                if spiders[x].weight > means:
                    near_w = nearest_spider(spiders[x], ["spiders[i].gender == Female"])
                    spiders[x].s_next = type_1_male(spiders[x].s, near_w.s, vibrations(spiders[x], near_w), a, d, rand)
                else:
                    spiders[x].s_next = type_2_male(spiders[x].s, a)
        for m in range(population_male):
            if spiders[population_female + m].group == D:
                sp = []
                likely = []
                for w in range(population_female):
                    if distance_euclidean(spiders[population_female + m].s, spiders[w].s) < r:
                        sp.append(spiders[w])
                if len(sp) != 0:
                    sp.append(spiders[population_female + m])
                    total_weight = 0
                    for j in range(len(sp)):
                        total_weight = total_weight + sp[j].weight
                    likely.append(sp[0].weight / total_weight)
                    for j in range(len(sp) - 1):
                        likely.append((sp[j + 1].weight / total_weight) + likely[j])
                    spider_new = np.zeros(n)
                    for j in range(n):
                        number = random.random()
                        for k in range(len(sp)):
                            if number < likely[k]:
                                spider_new[j] = sp[k].s[j]
                                break
                    if check(spider_new):
                        for same in range(len(sp)):
                            if np.array_equal(spider_new, sp[same].s):
                                sp.remove(sp[same])
                                random_pos = random.randint(0, n - 1)
                                random_sp = random.randint(0, len(sp) - 1)
                                spider_new[random_pos] = sp[random_sp].s[random_pos]
                                break
                    if f(spider_new) > worst.fitness:
                        worst.s = spider_new
                        worst.s_next = spider_new
                        worst.fitness = f(spider_new)
                        worst = minimum()
                        best = maximum()
                        update_weight(best.fitness, worst.fitness)
                        means = median_male_spider()
                        update_group(means)
        number_of_iterations += 1
        max_fes_reached = number_of_iterations >= max_fes
        error_achieved = best.fitness < max_error
    maximize = maximum()
    return maximize.fitness, maximize.s_next




# Rosenbrock function	maximize=0  (1,1)
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


# Number of executions per problem
executions_per_problem = 51

for test in range(executions_per_problem):
    test_3()
    best_fitness, best_s = social_spider_optimization()
    print('\n' + colored("Τest " + str(test + 1), 'blue') + '\n' + "f(max) = "+str(best_fitness)+" max = "+str(best_s))
