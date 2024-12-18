import random
import time
from collections import deque
from multiprocessing import Process, Queue

def initialize_solution(N, K, distance_matrix):

    routes = [[i] for i in range(1, N + 1)]  

    while len(routes) > K:
        min_cost_idx = min(range(len(routes)), key=lambda idx: calculate_route_distance(routes[idx], distance_matrix))
        R1 = routes.pop(min_cost_idx) 

        for node in R1:
            best_increase = float('inf')
            best_route_idx = None
            best_position = None

            for i, route in enumerate(routes):
                candidate_positions = [0, len(route) // 2, len(route)]
                for j in candidate_positions:
                    new_route = route[:j] + [node] + route[j:]
                    increase = calculate_route_distance(new_route, distance_matrix) - calculate_route_distance(route, distance_matrix)
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = i
                        best_position = j

            if best_route_idx is not None and best_position is not None:
                routes[best_route_idx].insert(best_position, node)

    return routes



def calculate_route_distance(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

def evaluate_solution(routes, distance_matrix):
    return max(calculate_route_distance([0] + route, distance_matrix) for route in routes)

def generate_neighbors_reinsertion(routes, distance_matrix):
    neighbors = []
    for i in range(len(routes)):
        for j in range(len(routes)):
            if i != j and routes[i]:
                for node in routes[i]:
                    new_routes = [list(r) for r in routes]
                    new_routes[i].remove(node)
                    new_routes[j].append(node)
                    neighbors.append(new_routes)
    return neighbors

def generate_neighbors_2opt(routes, distance_matrix):
    neighbors = []
    for idx, route in enumerate(routes):
        if len(route) < 2:
            continue
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                if i == 0 and j == len(route) - 1:
                    continue
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_routes = [list(r) for r in routes]
                new_routes[idx] = new_route
                neighbors.append(new_routes)
    return neighbors

def generate_neighbors_exchange(routes, distance_matrix):
    neighbors = []

    for i in range(len(routes)):
        for j in range(i, len(routes)):  
            if not routes[i] or not routes[j]:
                continue  

            if i == j and len(routes[i]) > 1:
                for idx1 in range(len(routes[i])):
                    for idx2 in range(idx1 + 1, len(routes[i])):
                        new_route = routes[i][:]
                        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
                        new_routes = [list(r) for r in routes]
                        new_routes[i] = new_route
                        neighbors.append(new_routes)

            if i != j:
                for idx1 in range(len(routes[i])):
                    for idx2 in range(len(routes[j])):
                        new_routes = [list(r) for r in routes]
                        # Swap nodes
                        new_routes[i][idx1], new_routes[j][idx2] = new_routes[j][idx2], new_routes[i][idx1]
                        neighbors.append(new_routes)

    return neighbors

def generate_neighbors_perturbation_combined(routes, distance_matrix):
    neighbors = []

    for _ in range(5): 
        new_routes = [list(route) for route in routes]
        non_empty_routes = [r for r in new_routes if r]
        if len(non_empty_routes) < 2:
            continue  

        route_a, route_b = random.sample(non_empty_routes, 2)
        node_idx = random.randint(0, len(route_a) - 1)
        node = route_a.pop(node_idx)
        insert_idx = random.randint(0, len(route_b))
        route_b.insert(insert_idx, node)
        neighbors.append(new_routes)

    reinsertion_neighbors = generate_neighbors_reinsertion(routes, distance_matrix)
    neighbors.extend(reinsertion_neighbors)

    two_opt_neighbors = generate_neighbors_2opt(routes, distance_matrix)
    neighbors.extend(two_opt_neighbors)

    return neighbors

def generate_neighbors_combined(routes, distance_matrix):
    neighbors = generate_neighbors_reinsertion(routes, distance_matrix)
    neighbors += generate_neighbors_2opt(routes, distance_matrix)
    return neighbors

def tabu_search_process(
    N, K, distance_matrix, max_iter, time_limit,
    neighborhood_func, tabu_tenure, neighborhood_prob,
    solution_queue, solution_pool, segment_duration
):
    start_time = time.time()
    last_segment_time = time.time()

    if neighborhood_func == generate_neighbors_reinsertion:
        tenure = int(0.05 * N)
    elif neighborhood_func == generate_neighbors_2opt:
        tenure = int(0.1 * N)
    else:
        tenure = int(0.075 * N)


    current_solution = initialize_solution(N, K, distance_matrix)
    best_solution = current_solution
    best_cost = evaluate_solution(best_solution, distance_matrix)

    tabu_list = deque(maxlen=tenure)
    tabu_set = set([tuple(tuple(route) for route in current_solution)])

    iteration = 0
    while time.time() - start_time < time_limit and iteration < max_iter:
        iteration += 1

        if time.time() - last_segment_time > segment_duration:
            solution_pool.put((best_solution, best_cost))
            while not solution_pool.empty():
                candidate_solution, candidate_cost = solution_pool.get()
                if candidate_cost < best_cost:
                    best_solution = candidate_solution
                    best_cost = candidate_cost
                    current_solution = candidate_solution
            last_segment_time = time.time()

        neighbors = neighborhood_func(current_solution, distance_matrix)
        neighbors = [n for n in neighbors if tuple(tuple(route) for route in n) not in tabu_set]
        if not neighbors:
            continue

        neighbor_costs = [evaluate_solution(n, distance_matrix) for n in neighbors]
        best_neighbor_idx = neighbor_costs.index(min(neighbor_costs))
        best_neighbor = neighbors[best_neighbor_idx]
        best_neighbor_cost = neighbor_costs[best_neighbor_idx]

        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost

        current_solution = best_neighbor
        tabu_set.add(tuple(tuple(route) for route in current_solution))
        tabu_list.append(current_solution)

    solution_queue.put((best_solution, best_cost))


def parallel_tabu_search(
    N, K, distance_matrix,
    max_iter=1000, time_limit=30, segment_duration=5
):

    solution_queue = Queue()
    solution_pool = Queue()

    neighborhood_configs = [
        (generate_neighbors_reinsertion, 0.4),        # Thread 1: Reinsertion + 2-opt
        (generate_neighbors_2opt, 0.2),            # Thread 2: 2-opt
        (generate_neighbors_exchange, 0.2),        # Thread 3: Exchange
        (generate_neighbors_perturbation_combined, 0.2)  # Thread 4: Perturbation + mixed
    ]


    processes = []
    for neighborhood_func, prob in neighborhood_configs:
        p = Process(
            target=tabu_search_process,
            args=(N, K, distance_matrix, max_iter, time_limit,
                  neighborhood_func, N, prob, solution_queue,
                  solution_pool, segment_duration)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    best_solution = None
    best_cost = float('inf')
    while not solution_queue.empty():
        solution, cost = solution_queue.get()
        if cost < best_cost:
            best_solution = solution
            best_cost = cost

    return best_solution, best_cost


def main():
    
    N, K = map(int, input().split())
    distance_matrix = [list(map(int, input().split())) for _ in range(N+1)]

    best_solution, best_cost = parallel_tabu_search(
        N, K, distance_matrix,
        max_iter=10000, time_limit=240
    )

    print(K)
    for route in best_solution:
        print(len(route) + 1)
        print(" ".join(map(str, [0] + route))) 


if __name__ == "__main__":
    main()
