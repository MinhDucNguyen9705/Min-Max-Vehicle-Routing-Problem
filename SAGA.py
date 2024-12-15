import sys
import random 
import time 
import matplotlib.pyplot as plt 
import math

class GeneticAlgorithm():

    def __init__(self, generations=10000, population_size=10, mutation_rate=0.01, keep_parents=0, time_limit=100, T=100):
        self.generations = generations              # Số lượng thế hệ
        self.population_size = population_size      # Số lượng cá thể trong mỗi thế hệ
        self.mutation_rate = mutation_rate          # Tỉ lệ đột biến
        self.keep_parents = keep_parents            # Số lượng cá thể được giữ lại sau mỗi thế hệ
        self.time_limit = time_limit                # Thời gian giới hạn chạy thuật toán
        self.time = 0
        self.T = T

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            if a <= 0:
                raise ValueError("`a` must be greater than 0 when it is an integer.")
            a = list(range(a))
        
        if p is not None:
            if len(p) != len(a):
                raise ValueError("`p` must have the same length as `a`.")
            if not (abs(sum(p) - 1.0) < 1e-9):
                raise ValueError("The probabilities in `p` must sum to 1.")
        else:
            p = [1 / len(a)] * len(a)
        
        def weighted_sample(population, probabilities, k):
            cumulative_probs = []
            cumulative_sum = 0
            for prob in probabilities:
                cumulative_sum += prob
                cumulative_probs.append(cumulative_sum)
            
            samples = []
            for _ in range(k):
                r = random.random() 
                for i, cp in enumerate(cumulative_probs):
                    if r <= cp:
                        samples.append(population[i])
                        break
            return samples

        num_samples = 1 if size is None else (size if isinstance(size, int) else int(size[0] * size[1]))

        if replace:
            result = weighted_sample(a, p, num_samples)
        else:
            if num_samples > len(a):
                raise ValueError("Cannot take a larger sample than population when 'replace=False'.")
            result = random.sample(a, num_samples)  

        if size is None:
            return result[0] 
        elif isinstance(size, tuple):
            return [result[i:i+size[1]] for i in range(0, len(result), size[1])] 
        else:
            return result 

    def load_input_file(self, filename):
        with open (filename, 'r') as f:
            n, k = list(map(int, f.readline().split()))
            d = []
            for i in range (n+1):
                d.append(list(map(float, f.readline().split())))
        self.n = n          # Số lượng node
        self.k = k          # Số lượng bưu tá
        self.d = d          # Ma trận khoảng cách   
        start = time.time()
        self.population = self.generate_greedy_initial_population()+self.generate_greedy_initial_population()
        end = time.time()
        self.time += end-start

    def load_input_line(self):
        n, k = list(map(int, input().split()))
        d = []
        for i in range (n+1):
            d.append(list(map(float, input().split())))
        self.n = n          # Số lượng node
        self.k = k          # Số lượng bưu tá
        self.d = d          # Ma trận khoảng cách
        start = time.time()   
        self.population = self.generate_greedy_initial_population() + self.generate_greedy_initial_population()
        end = time.time()
        self.time = end - start
    #[0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]

    def generate_greedy_initial_population(self):
        population = []
        for i in range (self.population_size//2):

            routes = [[] for _ in range(self.k)]
            unassigned = list(range(1, self.n+1))
            random.shuffle(unassigned)
            
            while unassigned:
                for vehicle in range(self.k):
                    if not unassigned:
                        break
                    
                    if not routes[vehicle]:
                        routes[vehicle].append(0)

                    current = routes[vehicle][-1]
                    nearest = min(unassigned, 
                        key=lambda p: self.d[current][p] if p in unassigned else float('inf')
                    )
                    
                    routes[vehicle].append(nearest)
                    unassigned.remove(nearest)
                # print(len(unassigned))
            population.append([point for route in routes for point in route]+[0])
            # print(len(population))
        return population
    
    def generate_initial_population(self):
        population = []
        for i in range (self.population_size//2):
            temp = [[] for i in range (self.k)]
            nodes = [i for i in range (1, self.n+1)]
            random.shuffle(nodes)
            for j in range (1, self.n+1):
                temp[random.randint(0, self.k-1)].append(nodes[j-1])
            for j in range (self.k):
                temp[j] += [0]
            chromosome = [0]
            for j in range (self.k):
                chromosome += temp[j]
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        curr_max = 0         # Quãng đường dài nhất
        curr_score = 0         # Quãng đường hiện tại
        for i in range (1, len(chromosome)):
            # curr_score += self.d[chromosome[i-1]][chromosome[i]]
            if chromosome[i] == 0:
                curr_max = max(curr_max, curr_score)
                # print(curr_score)
                curr_score = 0
                continue
            curr_score += self.d[chromosome[i-1]][chromosome[i]]
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
        if max_length < 1:
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
            duplicated = set()
            for i in range (start, end):           # Các node trong khoảng crossover sẽ được giữ nguyên
                not_added.remove(child[i])
            for i in range (start):             # Các node còn lại sẽ được kiểm tra và chỉnh sửa
                if child[i] == 0:
                    continue 
                if child[i] in not_added:
                    not_added.remove(child[i])
                else:
                    duplicated.add(i)
                #     for node in not_added:
                #         child[i] = node
                #         not_added.remove(node)
                #         break
            for i in range (end, length):
                if child[i] == 0:
                    continue 
                if child[i] in not_added:
                    not_added.remove(child[i])
                else:
                    duplicated.add(i)
                #     for node in not_added:
                #         child[i] = node
                #         not_added.remove(node)
                #         break
            for idx in duplicated:
                for value in not_added:
                    child[idx] = value 
                    not_added.remove(value)
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

#     def generate_new_population(self):
#         self.population.sort(key=lambda x: self.fitness(x))  
#         selected, selected_p = self.selection(self.population)
#         selected_p = [i/sum(selected_p) for i in selected_p]
#         selected = [val for val, prob in sorted(zip(selected, selected_p), key=lambda x: x[1], reverse=True)]
#         selected_p = sorted(selected_p, reverse=True)
#         new_population = []
#         if self.keep_parents > 0:
#             new_population = sorted(self.population[:min(self.keep_parents, len(self.population))], key=lambda x: self.fitness(x), reverse=True)
#         while len(new_population) < len(self.population):
#             parents_idx = self.choice(len(selected), size=2, 
#                                     replace=False, p=selected_p)
#             parents = [selected[parents_idx[0]], selected[parents_idx[1]]]
#             new_population += self.crossover(parents)

#         for i in range (len(new_population)):
#             mutation_rate = random.random()
#             if mutation_rate > self.mutation_rate:
#                 new_population[i] = self.mutation(new_population[i])
#         return new_population[:self.population_size]

#     def evaluate(self, runs=10):
#         scores = []
#         for i in range (runs):
#             score, _, _, _ = self.run()
#             scores.append(score)
#             print(f"Run {i+1}: Best score: {score}. Number of epochs: {self.epochs}. Time: {self.time}")
#         return scores

# class HybridAlgorithm(GeneticAlgorithm):

#     def __init__(self, generations=10000, population_size=10, mutation_rate=0.7, keep_parents=0, time_limit=100, T=10):
#         super().__init__(generations, population_size, mutation_rate, keep_parents, time_limit)
#         self.T = T                                  # Nhiệt độ ban đầu

    def simulated_annealing(self, chromosome, new_chromosome):
        f1 = self.fitness(chromosome)               # Điểm số của cá thể hiện tại
        f2 = self.fitness(new_chromosome)           # Điểm số của cá thể mới
        delta_f = f2 - f1                           
        if delta_f < 0:
            return True                             # Chấp nhận cá thể mới
        else:
            P = math.exp(-delta_f/self.T)
            r = random.random()                     # Sinh số ngẫu nhiên trong khoảng [0, 1]
            if r<=P:
                return True                         # Chấp nhận cá thể mới
            else:
                return False                        # Không chấp nhận cá thể mới

    def generate_new_population(self):
        self.population.sort(key=lambda x: self.fitness(x))         # Sắp xếp cá thể theo điểm số tăng dần
        selected, selection_p = self.selection(self.population)     # Chọn lọc cá thể
        total_p = sum(selection_p)
        normalized_p = [selection_p[i]/total_p for i in range (len(selection_p))]     # Chuẩn hóa tỉ lệ chọn lọc
        sorted_selected = sorted(zip(selected, normalized_p), key=lambda x: x[1], reverse=True)       # Sắp xếp cá thể theo tỉ lệ chọn lọc giảm dần
        selected, normalized_p = map(list, zip(*sorted_selected))
        # p = self.selection(self.population)            
        # new_population = []                                         # Danh sách chứa cá thể mới
        if self.keep_parents > 0:                                   # Giữ lại một số cá thể tốt nhất
            new_population = selected[:min(self.keep_parents, len(selected))]
        # if self.keep_parents > 0:                                   # Giữ lại một số cá thể tốt nhất
        #     new_population = sorted(self.population[:min(self.keep_parents, len(self.population))], key=lambda x: self.fitness(x), reverse=True)
        while len(new_population) < len(self.population):
                                                                                # Lai ghép ra các cá thể mới
            parents_idx = self.choice(len(selected), size=2,                      # Chọn lọc hai cá thể cha mẹ
                                    replace=False, p=normalized_p)
            parent1, parent2 = [selected[parents_idx[0]], selected[parents_idx[1]]]
            child1, child2 = self.crossover([parent1, parent2])
            mutation_rate = random.random()
            if mutation_rate > self.mutation_rate:
                child1 = self.mutation(child1)
            mutation_rate = random.random()
            if mutation_rate > self.mutation_rate:
                child2 = self.mutation(child2)
            # parents_idx = self.choice(len(self.population), size=2,                      # Chọn lọc hai cá thể cha mẹ
            #                         replace=False, p=p)
            # parent1, parent2 = [self.population[parents_idx[0]], self.population[parents_idx[1]]]
            # child1, child2 = self.crossover([parent1, parent2])                        
            if self.simulated_annealing(parent1, child1):
                new_population += [child1]
            if self.simulated_annealing(parent2, child2):
                new_population += [child2]
        # for i in range (len(new_population)):                           # Đột biến các cá thể mới
        #     mutation_rate = random.random()
        #     print(len(new_population[i]))
        #     if mutation_rate > self.mutation_rate:
        #         new_population[i] = self.mutation(new_population[i])
        return new_population[:self.population_size]

    def run(self, verbose=0):
        self.mean_history = []                           # Lịch sử giá trị trung bình của mỗi thế hệ
        self.best_history = []
        self.all_best_history = []
        self.best_solution = []                     # Lời giải tốt nhất
        self.best_score = float('inf')              # Điểm tốt nhất
        self.not_increase_epochs = 0                # Số lượng thế hệ không cải thiện điểm số
        self.epochs = 0                             # Số lượng thế hệ hiện tại
        self.best_generation = 0                    # Thế hệ chứa lời giải tốt nhất
        for i in range (self.generations):
            self.T = max(3*2**(-self.epochs/self.generations), 0.3)
            # self.T = 0.99*self.T
            start = time.time()
            self.population = self.generate_new_population()
            fitnesses = []
            score = float('inf')
            best_chromosome = []
            for chromosome in self.population:
                new_score = self.fitness(chromosome)
                fitnesses.append(new_score)
                if new_score < score:
                    score = new_score
                    best_chromosome = chromosome.copy()
            # self.history.append(score)
            self.mean_history.append(sum(fitnesses)/len(fitnesses))
            if score < self.best_score:
                self.best_score = score
                self.best_solution = best_chromosome.copy()
                self.not_increase_epochs = 0
                self.best_generation = self.epochs
            else:
                self.not_increase_epochs += 1
                if self.not_increase_epochs == 5000:
                    break
            self.best_history.append(score)
            self.all_best_history.append(self.best_score)
            self.epochs += 1
            end = time.time()
            self.time += end-start
            if self.time > self.time_limit:
                break
            if verbose==1:
                print('-------------------------------------------------')
                print(f"Generation {self.epochs}: .Current best score: {score}. All time best score: {self.best_score}")
                print(f"Elapsed time: {self.time}")
                print(f"Temperature: {self.T}")
                print(f"Mean score: {sum(fitnesses)/len(fitnesses)}")
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
    saga = GeneticAlgorithm(generations=20000, population_size=50, mutation_rate=0.5, keep_parents=10, time_limit=30, T=10000)
    saga.load_input_file('inputs/input7.txt')
    # saga.load_input_line()
    best_score, best_solution, epochs, best_generation = saga.run(verbose=1)
    # scores = ga.evaluate(10)
    # scores = np.array(scores)
    # print(f"Mean: {np.mean(scores)}")
    # print(f"Std: {np.std(scores)}")
    # print(best_score)
    # print(best_solution)
    # print(epochs)
    # print(f"Best score: {best_score}\nBest solution is {best_solution}\nFound at generation {best_generation}")
    zero_idx = [i for i in range (len(best_solution)) if best_solution[i] == 0]
    # print(saga.best_score)
    # print(best_solution)
    print(saga.k)
    for i in range (len(zero_idx)-1):
        print(zero_idx[i+1]-zero_idx[i])
        toPrint = best_solution[zero_idx[i]:zero_idx[i+1]]
        for j in range (len(toPrint)):
            print(toPrint[j], end = " ")
        print()
    plt.plot(saga.mean_history)
    plt.show()
    plt.plot(saga.all_best_history, color='blue')
    plt.plot(saga.best_history, color='red')
    plt.show()