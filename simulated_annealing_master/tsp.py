import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NodeGenerator:
    def __init__(self, width, height, nodesNumber):
        self.width = width
        self.height = height
        self.nodesNumber = nodesNumber

    def generate(self):
        xs = np.array([37, 37.764751, 38.750714, 39.626922, 38.36869, 40.64991, 39.92077, 36.88414, 41.110481, 41.18277, 37.856041, 39.648369, 41.581051, 37.881168, 40.255169, 40.056656, 39.062635, 38.393799, 40.575977, 37.461267, 40.266864, 40.155312, 40.601343, 40.550556, 37.77652, 37.91441, 40.843849, 41.681808, 38.680969, 39.75, 39.9, 39.776667, 37.06622, 40.912811, 40.438588, 37.583333, 36.401849, 39.887984, 37.764771, 41.00527, 38.41885, 37.585831, 41.2061, 37.17593, 40.616667, 41.38871, 38.73122, 39.846821, 41.733333, 39.14249, 36.718399, 40.85327, 37.866667, 39.416667, 38.35519, 38.619099, 37.321163, 36.8, 37.215278, 38.946189, 38.69394, 37.966667, 40.983879, 37.213026, 41.02005, 40.693997, 41.292782, 37.933333, 42.02314, 39.747662, 37.159149, 37.418748, 40.983333, 40.316667, 41.00145, 39.307355, 38.682301, 38.48914, 40.65, 39.818081, 41.456409])
        ys = np.array([35.321333, 38.278561, 30.556692, 43.021596, 34.03698, 35.83532, 32.85411, 30.70563, 42.702171, 41.818292, 27.841631, 27.88261, 32.460979, 41.13509, 40.22488, 30.066524, 40.76961, 42.12318, 31.578809, 30.066524, 29.063448, 26.41416, 33.613421, 34.955556, 29.08639, 40.230629, 31.15654, 26.562269, 39.226398, 39.5, 41.27, 30.520556, 37.38332, 38.38953, 39.508556, 43.733333, 36.34981, 44.004836, 30.556561, 28.97696, 27.12872, 36.937149, 32.62035, 33.228748, 43.1, 33.78273, 35.478729, 33.515251, 27.216667, 34.17091, 37.12122, 29.88152, 32.483333, 29.983333, 38.30946, 27.428921, 40.724477, 34.633333, 28.363611, 41.753893, 34.685651, 34.683333, 37.876411, 36.176261, 40.523449, 30.435763, 36.33128, 41.95, 35.153069, 37.017879, 38.796909, 42.491834, 27.516667, 36.55, 39.7178, 39.438778, 29.40819, 43.40889, 29.266667, 34.81469, 31.798731])

        return np.column_stack((xs, ys))


class SimulatedAnnealing:
    def __init__(self, coords, temp, alpha, stopping_temp, stopping_iter):
        ''' animate the solution over time

            Parameters
            ----------
            coords: array_like
                list of coordinates
            temp: float
                initial temperature
            alpha: float
                rate at which temp decreases
            stopping_temp: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates

        '''

        self.coords = coords
        self.sample_size = len(coords)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = vectorToDistMatrix(coords)
        self.curr_solution = nearestNeighbourSolution(self.dist_matrix)
        self.best_solution = self.curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        print('Intial weight: ', self.curr_weight)

    def weight(self, sol):
        '''
        Calcuate weight
        '''
        #return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def acceptance_probability(self, candidate_weight):
        '''
        Acceptance probability as described in:
        https://stackoverflow.com/questions/19757551/basics-of-simulated-annealing-in-python
        '''
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        '''
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        '''
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        '''
        Annealing process with 2-opt
        described here: https://en.wikipedia.org/wiki/2-opt
        '''
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)

            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
              round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')

    def animateSolutions(self):
        animateTSP(self.solution_history, self.coords)

    def plotLearning(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()


def vectorToDistMatrix(coords):
    '''
    Create the distance matrix
    '''
    lst = []
    for c1 in coords:
        matrix = []
        for c2 in coords:
            x1 = c1[0] - c2[0]
            x2 = c1[1] - c2[1]

            matrix.append(np.sqrt(x1*x1 + x2 * x2))
        lst.append(np.array(matrix))

    return np.array(lst)
    #return np.sqrt((np.square(coords[:, np.newaxis] - coords).sum(axis=2)))


def nearestNeighbourSolution(dist_matrix):
    '''
    Computes the initial solution (nearest neighbour strategy)
    '''
    node = random.randrange(len(dist_matrix))
    result = [node]

    nodes_to_visit = list(range(len(dist_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result


def animateTSP(history, points):
    ''' animate the solution over time

        Parameters
        ----------
        hisotry : list
            history of the solutions chosen by the algorith
        points: array_like
            points with the coordinates
    '''

    ''' approx 1500 frames for animation '''
    key_frames_mult = len(history) // 15

    fig, ax = plt.subplots()

    ''' path is a line coming through all the nodes '''
    line, = plt.plot([], [], lw=2)

    def init():
        ''' initialize node dots on graph '''
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        plt.plot(x, y, 'co')

        ''' draw axes slighty bigger  '''
        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

        '''initialize solution to be empty '''
        line.set_data([], [])
        return line,

    def update(frame):
        ''' for every frame update the solution on the graph '''
        x = [points[i, 0] for i in history[frame] + [history[frame][0]]]
        y = [points[i, 1] for i in history[frame] + [history[frame][0]]]
        line.set_data(x, y)
        return line

    ''' animate precalulated solutions '''

    ani = FuncAnimation(fig, update, frames=range(0, len(history), key_frames_mult),
                        init_func=init, interval=3, repeat=False)

    plt.show()




def main():
    lst = []
    for i in range(0,50):
        '''set the simulated annealing algorithm params'''
        temp = 10000
        stopping_temp = 0.003
        alpha = 0.995
        stopping_iter = 10000000

        '''set the dimensions of the grid'''
        size_width = 81
        size_height = 81

        '''set the number of nodes'''
        population_size = 81

        '''generate random list of nodes'''
        nodes = NodeGenerator(size_width, size_height, population_size).generate()

        '''run simulated annealing algorithm with 2-opt'''
        sa = SimulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter)
        sa.anneal()

        '''animate'''
        sa.animateSolutions()

        '''show the improvement over time'''
        sa.plotLearning()

        lst.append(sa.curr_weight)
    print(np.average(lst))
if __name__ == "__main__":
    main()
