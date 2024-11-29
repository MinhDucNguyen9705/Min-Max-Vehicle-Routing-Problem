import sys
import random 
import time 
import matplotlib.pyplot as plt 
import numpy as np 
from GA import GeneticAlgorithm

class HybridAlgorithm(GeneticAlgorithm):

    def __init__(self, generations=10000, population_size=10, mutation_rate=0.7, keep_parents=0, T=10):
        self.generations = generations              # Số lượng thế hệ
        self.population_size = population_size      # Số lượng cá thể trong mỗi thế hệ
        self.mutation_rate = mutation_rate          # Tỉ lệ đột biến
        self.keep_parents = keep_parents            # Số lượng cá thể được giữ lại sau mỗi thế hệ
        self.T = T                                  # Nhiệt độ ban đầu

    def simulated_annealing(self, chromosome, new_chromosome):
        f1 = self.fitness(chromosome)               # Điểm số của cá thể hiện tại
        f2 = self.fitness(new_chromosome)           # Điểm số của cá thể mới
        delta_f = f2 - f1                           
        if delta_f < 0:
            return True                             # Chấp nhận cá thể mới
        else:
            P = np.exp(-delta_f/self.T)
            r = random.random()                     # Sinh số ngẫu nhiên trong khoảng [0, 1]
            if r<=P:
                return True                         # Chấp nhận cá thể mới
            else:
                return False                        # Không chấp nhận cá thể mới

    def generate_new_population(self):
        self.population.sort(key=lambda x: self.fitness(x))         # Sắp xếp cá thể theo điểm số tăng dần
        selected, selection_p = self.selection(self.population)     # Chọn lọc cá thể
        total_p = sum(selection_p)
        normalized_p = np.array(selection_p) / total_p     # Chuẩn hóa tỉ lệ chọn lọc
        sorted_selected = sorted(zip(selected, normalized_p), key=lambda x: x[1], reverse=True)       # Sắp xếp cá thể theo tỉ lệ chọn lọc giảm dần
        selected, normalized_p = map(list, zip(*sorted_selected))            
        new_population = []                                         # Danh sách chứa cá thể mới
        if self.keep_parents > 0:                                   # Giữ lại một số cá thể tốt nhất
            new_population = selected[:min(self.keep_parents, len(selected))]
        while len(new_population) < len(self.population):               # Lai ghép ra các cá thể mới
            parents_idx = np.random.choice(len(selected), size=2,                      # Chọn lọc hai cá thể cha mẹ
                                    replace=False, p=normalized_p)
            parent1, parent2 = [selected[parents_idx[0]], selected[parents_idx[1]]]
            child1, child2 = self.crossover([parent1, parent2])                        
            if self.simulated_annealing(parent1, child1):
                new_population += [child1]
            if self.simulated_annealing(parent2, child2):
                new_population += [child2]
        for i in range (len(new_population)):                           # Đột biến các cá thể mới
            mutation_rate = random.random()
            if mutation_rate > self.mutation_rate:
                new_population[i] = self.mutation(new_population[i])
        return new_population[:min(self.population_size*10, len(new_population))]

    def run(self, verbose=0):
        self.history = []                           # Lịch sử giá trị tốt nhất của mỗi thế hệ
        self.best_solution = []                     # Lời giải tốt nhất
        self.best_score = float('inf')              # Điểm tốt nhất
        self.not_increase_epochs = 0                # Số lượng thế hệ không cải thiện điểm số
        self.epochs = 0                             # Số lượng thế hệ hiện tại
        self.best_generation = 0                    # Thế hệ chứa lời giải tốt nhất
        self.time = 0                               # Thời gian chạy thuật toán
        for i in range (self.generations):
            # self.T = max(3*2**(-self.epochs/self.generations), 0.3)
            self.T = 0.95*self.T
            start = time.time()
            self.population = self.generate_new_population()
            score = float('inf')
            best_chromosome = []
            for chromosome in self.population:
                new_score = self.fitness(chromosome)
                if new_score < score:
                    score = new_score
                    best_chromosome = chromosome
            self.history.append(score)
            if score < self.best_score:
                self.best_score = score
                self.best_solution = best_chromosome
                self.not_increase_epochs = 0
                self.best_generation = self.epochs
            else:
                self.not_increase_epochs += 1
                if self.not_increase_epochs == 3000:
                    break
            self.epochs += 1
            end = time.time()
            self.time += end-start
            if verbose==1:
                print('-------------------------------------------------')
                print(f"Generation {self.epochs}: .Current best score: {score}. All time best score: {self.best_score}")
                print(f"Elapsed time: {self.time}")
                print(f"Temperature: {self.T}")
                print(f"{len(self.population)}")
        return self.best_score, self.best_solution, self.epochs, self.best_generation

    def evaluate(self, runs=10):
        scores = []
        for i in range (runs):
            score, _, _, _ = self.run()
            scores.append(score)
            print(f"Run {i+1}: Best score: {score}. Number of epochs: {self.epochs}. Time: {self.time}")
        return scores

if __name__ == "__main__":
    saga = HybridAlgorithm(generations=1000, population_size=50, mutation_rate=0.02, keep_parents=10, T=10000)
    saga.load_input('input.txt')
    best_score, best_solution, epochs, best_generation = saga.run(verbose=1)
    # scores = ga.evaluate(10)
    # scores = np.array(scores)
    # print(f"Mean: {np.mean(scores)}")
    # print(f"Std: {np.std(scores)}")
    # print(best_score)
    # print(best_solution)
    # print(epochs)
    print(f"Best score: {best_score}\nBest solution is {best_solution}\nFound at generation {best_generation}")
    zero_idx = [i for i in range (len(best_solution)) if best_solution[i] == 0]
    for i in range (len(zero_idx)-1):
        print(zero_idx[i+1]-zero_idx[i]-1)
        toPrint = best_solution[zero_idx[i]:zero_idx[i+1]]
        for j in range (len(toPrint)):
            print(toPrint[j], end = " ")
        print()
    plt.plot(saga.history)
    plt.show()