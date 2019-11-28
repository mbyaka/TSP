import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import random


class City:
    def __init__(self, name, x, y):
        self.x = float(x)
        self.y = float(y)
        self.name = name

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + self.name + "," + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            dst = float(self.routeDistance())
            if dst == 0:
                self.fitness = 1
            else:
                self.fitness = 1 / dst
        return self.fitness


class Result:
    def __init__(self, Index, Best, Tour):
        self.Index = Index
        self.Best = Best
        self.Tour = Tour

    def __repr__(self):
        return "(" + str(self.Index) + "," + str(self.Best) + ")"


class Analysis:
    def __init__(self, p_size, e_size, m_rate, gCount, Best, Mean, StdDev, Tour):
        self.Best = Best
        self.Mean = Mean
        self.StdDev = StdDev
        self.Tour = Tour
        self.pSize = p_size
        self.eSize = e_size
        self.mRate = m_rate
        self.gCount = gCount

    def __repr__(self):
        return "(" + \
               "pSize: " + str(self.pSize) + "," + \
               "eSize: " + str(self.eSize) + "," + \
               "mRate: " + str(self.mRate) + "," + \
               "gCount: " + str(self.gCount) + "," + \
               "Best: " + str(self.Best) + "," + \
               "Mean: " + str(self.Mean) + "," + \
               "StdDev: " + str(self.StdDev) + \
               ")"


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for j in range(0, len(popRanked)):
            if pick <= df.iat[j, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    childP1 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    mpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(mpool, eliteSize)
    nGeneration = mutatePopulation(children, mutationRate)
    return nGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    # print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    results = []
    rank = rankRoutes(pop)
    results.append(Result(Index=0, Best=(1 / rank[0][1]), Tour=pop[rank[0][0]]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        rank = rankRoutes(pop)
        results.append(Result(Index=i + 1, Best=(1 / rank[0][1]), Tour=pop[rank[0][0]]))

    # plt.plot(ls)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()

    # print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))

    # bestRouteIndex = rankRoutes(pop)[0][0]
    # bestRoute = pop[bestRouteIndex]

    # print(min((o.Best for o in results)))

    # print(bestRoute)

    return getOptimum(results)


def getOptimum(res):
    dist = 9999999
    opt = {}

    for r in res:
        # bests.append(r.Best)
        if r.Best < dist:
            opt = r
            dist = r.Best

    return opt


def getAnalysis(res):
    best = getOptimum(res)

    bests = []
    for r in res:
        bests.append(r.Best)

    mean = np.average(bests)
    std = np.std(bests)

    return best, mean, std


def main():
    cityList = []

    with open('iller.csv', encoding="Windows-1254", ) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            cityList.append(City(name=row[1], x=row[2], y=row[3]))

    pSize = [10, 20, 30]
    #eSize = [10, 20, 30]
    mRate = [0.01, 0.1, 0.5]
    generationCount = [20, 100, 300]

    analysis = []


    data = []

    for pS in pSize:
        #for eS in eSize:
        for mR in mRate:
            for gC in generationCount:
                results = []
                for tttt in range(0, 100):
                    res = geneticAlgorithm(population=cityList, popSize=pS, eliteSize=pS, mutationRate=mR,
                                           generations=gC)
                    results.append(res)

                b, m, s = getAnalysis(results)
                analysis.append(
                    Analysis(p_size=pS, e_size=pS, m_rate=mR, gCount=gC, Best=b, Mean=m, StdDev=s, Tour=b.Tour))

                jj = {"Best": analysis[-1].Best.Best,
                      "Mean": analysis[-1].Mean,
                      "StdDev": analysis[-1].StdDev,
                      "pSize": analysis[-1].pSize,
                      "eSize": analysis[-1].eSize,
                      "mRate": analysis[-1].mRate,
                      "gCount": analysis[-1].gCount}

                data.append(jj)
                print(str(pS) + " " + str(pS) + " " + str(mR) + " " + str(gC))

    f = open("output_plots/out.json", "a")
    f.write(json.dumps(data))
    f.close()


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


main()
