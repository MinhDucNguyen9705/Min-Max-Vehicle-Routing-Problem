import sys
import random 
import time 
import matplotlib.pyplot as plt 
import numpy as np 

class GeneticAlgorithm():

    def __init__(self, generations=10000, population_size=10, mutation_rate=0.7, keep_parents=0):
        self.generations = generations              # Số lượng thế hệ
        self.population_size = population_size      # Số lượng cá thể trong mỗi thế hệ
        self.mutation_rate = mutation_rate          # Tỉ lệ đột biến
        self.keep_parents = keep_parents            # Số lượng cá thể được giữ lại sau mỗi thế hệ

    def load_input(self, filename):
        with open (filename, 'r') as f:
            n, k = list(map(int, f.readline().split()))
            d = []
            for i in range (n+1):
                d.append(list(map(float, f.readline().split())))
        self.n = n          # Số lượng node
        self.k = k          # Số lượng bưu tá
        self.d = d          # Ma trận khoảng cách   
        self.population = self.generate_initial_population()   # Khởi tạo thế hệ đầu tiên

    #[0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]
    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            # Cá thể có dạng 0....0....0....0, quãng đường mỗi bưu tá đi sẽ là các node từ số 0 này đến số 0 kế tiếp
            chromosome = [0]    # Khởi tạo cá thể
            added = set()       # Set chứa các node đã thêm
            count = 0           # Đếm số lượng xe đã thêm
            while count < self.k-1:      # Thêm quãng đường di chuyển của k-1 bưu tá đầu vào cá thể
                rand = random.randint(1, self.n)     # Chọn ngẫu nhiên một node từ 1->n
                end = random.random()           # Chọn ngẫu nhiên một số từ 0->1
                if rand not in added:           # Nếu node chưa được thêm vào cá thể thì thêm vào
                    added.add(rand)
                    chromosome += [rand]
                if 1-end >= end:                # Nếu end < 0.5 thì thêm node 0 vào cá thể, tức là kết thúc quãng đường của một bưu tá
                    chromosome += [0]
                    count += 1
            for i in range (1, self.n+1):            # Thêm quãng đường của bưu tá cuối cùng, với các node còn lại
                if i not in added:
                    chromosome += [i]
                    added.add(i)
            chromosome += [0]
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        curr_max = 0         # Quãng đường dài nhất
        curr_score = 0         # Quãng đường hiện tại
        for i in range (1, len(chromosome)):
            curr_score += self.d[chromosome[i-1]][chromosome[i]]
            if chromosome[i] == 0:
                if curr_score > curr_max:
                    curr_max = curr_score
                curr_score = 0
        return curr_max

    def selection(self, population):
        fitnesses = []    # Danh sách điểm số của từng cá thể
        for chromosome in population:
            fitnesses.append(self.fitness(chromosome))
        total = sum(fitnesses)     # Tổng điểm số của tất cả cá thể
        p = [0 for i in range (len(fitnesses))]     # Danh sách xác suất chọn của từng cá thể
        accumulated_p = [0 for i in range (len(fitnesses))]    # Danh sách xác suất tích lũy
        #acculated_p[i] = p[0]+...+p[i] 
        for i in range(len(fitnesses)):
            p[i] = fitnesses[i]/total
            accumulated_p[i] = accumulated_p[i-1] + p[i]      
        selected = []                    # Danh sách cá thể được chọn
        selected_p = []                  # Danh sách xác suất cá thể được chọn
        for i in range (self.keep_parents):
            selected.append(population[i])
            selected_p.append(p[i])
        alpha = random.random()                     # Chọn ngẫu nhiên một số từ 0->1
        for i in range(self.keep_parents, len(fitnesses)):
            if alpha < p[i]:                        # Nếu p[0]+p[1]+...+p[i-1] < alpha < p[0]+p[1]+..+p[i] thì chọn cá thể thứ i
                selected.append(population[i])
                selected_p.append(p[i])
        return selected, selected_p

    # def selection(self, population):
    #     fitnesses = []    # Danh sách điểm số của từng cá thể
    #     for chromosome in population:
    #         fitnesses.append(self.fitness(chromosome))
    #     total = sum(fitnesses)     # Tổng điểm số của tất cả cá thể
    #     p = [fitnesses[i]/total for i in range (len(fitnesses))]     # Danh sách xác suất chọn của từng cá thể

    def crossover(self, parents):
        length = len(parents[0])               # Độ dài của cá thể

        zero_idx_0 = [i for i in range (length) if parents[0][i] == 0]     # Danh sách vị trí của node 0 
        zero_idx_1 = [i for i in range (length) if parents[1][i] == 0]     

        length_non_zero_0 = [zero_idx_0[i+1] - zero_idx_0[i] - 1 for i in range (len(zero_idx_0)-1)]     # Danh sách số node giữa các điểm 0
        length_non_zero_1 = [zero_idx_1[i+1] - zero_idx_1[i] - 1 for i in range (len(zero_idx_1)-1)]

        idx_0 = random.randint(0, len(length_non_zero_0)-1)         # Khoảng được chọn để crossover trong danh sách trên
        max_length_0 = length_non_zero_0[idx_0]                     # Độ dài của khoảng được chọn
        idx_1 = random.randint(0, len(length_non_zero_1)-1)
        max_length_1 = length_non_zero_1[idx_1]

        max_length = min(max_length_0, max_length_1)                # Độ dài lớn nhất có thể crossover
        if max_length <= 1:
            return parents
        
        window_size = random.randint(1, max_length)                 # Độ dài của crossover
        
        start_0 = random.randint(zero_idx_0[idx_0]+1, zero_idx_0[idx_0+1]-window_size)   # Vị trí bắt đầu của crossover
        start_1 = random.randint(zero_idx_1[idx_1]+1, zero_idx_1[idx_1+1]-window_size)

        end_0 = start_0 + window_size               # Vị trí kết thúc của crossover
        end_1 = start_1 + window_size 

        child_0 = parents[0][:start_0] + parents[1][start_1:end_1] + parents[0][end_0:]     # Crossover
        child_1 = parents[1][:start_1] + parents[0][start_0:end_0] + parents[1][end_1:]

        not_added_0 = set(range(1, self.n+1))       # Danh sách node chưa được thêm vào cá thể
        not_added_1 = set(range(1, self.n+1))

    #Bố: [0, 1, 2, 3, 0, 8, 9, 5, 0, 8, 7, 9, 0]
    #Mẹ: [0, 6, 4, 6, 0, 3, 4, 5, 0, 1, 2, 9, 0]
    #Con: [0, 1, 2, 3, 0, 6, 4, 5, 0, 8, 7, 9, 0]
    #Con: [0, 8, 9, 6, 0, 3, 4, 5, 0, 1, 2, 9, 0]

        def validate(child, not_added, start, end):   # Kiểm tra trùng lặp trong cá thể hậu crossover và thêm node chưa được thêm vào cá thể
            for i in range (start, end):           # Các node trong khoảng crossover sẽ được giữ nguyên
                not_added.remove(child[i])
            for i in range (start):             # Các node còn lại sẽ được kiểm tra và chỉnh sửa
                if child[i] == 0:
                    continue 
                if child[i] in not_added:
                    not_added.remove(child[i])
                else:
                    for node in not_added:
                        child[i] = node
                        not_added.remove(node)
                        break
            for i in range (end, length):
                if child[i] == 0:
                    continue 
                if child[i] in not_added:
                    not_added.remove(child[i])
                else:
                    for node in not_added:
                        child[i] = node
                        not_added.remove(node)
                        break
            return child
        
        child_0 = validate(child_0, not_added_0, start_0, end_0)
        child_1 = validate(child_1, not_added_1, start_1, end_1)
        
        return child_0, child_1   

    def mutation(self, chromosome):
        for i in range (1):
            swap_idx = random.sample(range(1, len(chromosome)-1), 2)       # Chọn ngẫu nhiên 2 node để đổi chỗ
            chromosome[swap_idx[0]], chromosome[swap_idx[1]] = chromosome[swap_idx[1]], chromosome[swap_idx[0]]
        return chromosome

    def generate_new_population(self):
        self.population.sort(key=lambda x: self.fitness(x))  
        selected, selected_p = self.selection(self.population)
        selected_p = [i/sum(selected_p) for i in selected_p]
        selected = [val for val, prob in sorted(zip(selected, selected_p), key=lambda x: x[1], reverse=True)]
        selected_p = sorted(selected_p, reverse=True)
        new_population = []
        # if self.keep_parents > 0:
        #     new_population = selected[:min(self.keep_parents, len(selected))]
        while len(new_population) < len(self.population):
            # if len(selected) < 2:
            #     parents = random.sample(self.population, 2)
            # else:
            parents_idx = np.random.choice(len(selected), size=2, 
                                    replace=False, p=selected_p)
            parents = [selected[parents_idx[0]], selected[parents_idx[1]]]
            new_population += self.crossover(parents)

        for i in range (len(new_population)):
            mutation_rate = random.random()
            if mutation_rate > self.mutation_rate:
                new_population[i] = self.mutation(new_population[i])
            # f1 = self.fitness(new_population[i])
            # mutated_chromosome = self.mutation(new_population[i])
            # f2 = self.fitness(mutated_chromosome)
            # delta_f = f2 - f1
            # if delta_f < 0:
            #     new_population[i] = mutated_chromosome
            # else:
            #     P = np.exp(-delta_f/T)
            #     r = random.random()
            #     if r<=P:
            #         new_population[i] = mutated_chromosome
            #     else:
            #         continue
        return new_population

    def run(self, verbose=0):
        self.history = []                           # Lịch sử giá trị tốt nhất của mỗi thế hệ
        self.best_solution = []                     # Lời giải tốt nhất
        self.best_score = float('inf')              # Điểm tốt nhất
        self.not_increase_epochs = 0                # Số lượng thế hệ không cải thiện điểm số
        self.epochs = 0                             # Số lượng thế hệ hiện tại
        self.best_generation = 0                    # Thế hệ chứa lời giải tốt nhất
        self.time = 0                               # Thời gian chạy thuật toán
        for i in range (self.generations):
            start = time.time()
            # selected = self.selection(self.population)
            # self.population.sort(key=lambda x: self.fitness(x))   
            # new_population = []
            # if self.keep_parents > 0:
            #     new_population = selected[:min(self.keep_parents, len(selected))]
            # while len(new_population) < len(self.population):
            #     if len(selected) < 2:
            #         parents = random.sample(self.population, 2)
            #     else:
            #         parents = random.sample(selected, 2)
            #     new_population += self.crossover(parents)
            # for i in range (len(new_population)):
            #     mutation_rate = random.random()
            #     if mutation_rate > self.mutation_rate:
            #         new_population[i] = self.mutation(new_population[i])
            # self.population = new_population 
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
        return self.best_score, self.best_solution, self.epochs, self.best_generation

    def evaluate(self, runs=10):
        scores = []
        for i in range (runs):
            score, _, _, _ = self.run()
            scores.append(score)
            print(f"Run {i+1}: Best score: {score}. Number of epochs: {self.epochs}. Time: {self.time}")
        return scores

if __name__ == "__main__":
    ga = GeneticAlgorithm(generations=1000, population_size=50, mutation_rate=0.02, keep_parents=10)
    ga.load_input('input.txt')
    best_score, best_solution, epochs, best_generation = ga.run(verbose=1)
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
    plt.plot(ga.history)
    plt.show()