import random
import math
from typing import List, Tuple, Optional

class VRPSolver:
    def __init__(self, distance_matrix: List[List[int]], num_vehicles: int, 
                 max_iterations: int = 10000, 
                 initial_temperature: float = 10000, 
                 cooling_rate: float = 0.995,
                 max_route_distance: Optional[int] = None,
                 vehicle_capacity: Optional[int] = None):
        """
        Initialize VRP Solver with Simulated Annealing
        
        Args:
        - distance_matrix: 2D matrix of distances between locations
        - num_vehicles: Number of vehicles available
        - max_iterations: Maximum number of iterations
        - initial_temperature: Starting temperature for SA
        - cooling_rate: Rate of temperature reduction
        - max_route_distance: Maximum allowed distance for a single route
        - vehicle_capacity: Maximum capacity for each vehicle
        """
        self.distance_matrix = distance_matrix
        self.num_vehicles = num_vehicles
        self.num_customers = len(distance_matrix) - 1  # Excluding depot
        
        # SA parameters
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        # Constraint parameters
        self.max_route_distance = max_route_distance or float('inf')
        self.vehicle_capacity = vehicle_capacity or float('inf')
        
        # Tracking
        self.best_solution = None
        self.best_cost = float('inf')
        self.iteration_costs = []

    def initialize_solution(self) -> List[List[int]]:
        """
        Generate initial solution by random customer distribution
        
        Returns:
        List of routes, each route is a list of customer numbers
        """
        # Create list of customers
        customers = list(range(1, self.num_customers + 1))
        random.shuffle(customers)
        
        # Distribute customers to vehicles
        vehicles = [[] for _ in range(self.num_vehicles)]
        for i, customer in enumerate(customers):
            vehicles[i % self.num_vehicles].append(customer)
        
        return vehicles

    def calculate_cost(self, vehicles: List[List[int]]) -> Tuple[float, List[float]]:
        """
        Calculate total cost of the solution
        
        Returns:
        - Maximum route cost
        - List of individual route costs
        """
        route_costs = []
        for route in vehicles:
            if not route:
                route_costs.append(0)
                continue
            
            # Calculate route cost including depot to first/last customer
            route_cost = self.distance_matrix[0][route[0]]
            for i in range(len(route) - 1):
                route_cost += self.distance_matrix[route[i]][route[i+1]]            
            route_costs.append(route_cost)
        
        return max(route_costs), route_costs

    def generate_neighbor(self, vehicles: List[List[int]]) -> List[List[int]]:
        """
        Generate a neighboring solution using multiple strategies
        """
        new_vehicles = [route[:] for route in vehicles]
        move_type = random.choice([
            'swap_inter_route', 
            'swap_intra_route', 
            'relocate', 
            'route_shuffle', 
            '2-opt'
        ])
        
        if move_type == 'swap_inter_route':
            # Swap customers between different routes
            route1, route2 = random.sample(range(self.num_vehicles), 2)
            if new_vehicles[route1] and new_vehicles[route2]:
                idx1 = random.randint(0, len(new_vehicles[route1]) - 1)
                idx2 = random.randint(0, len(new_vehicles[route2]) - 1)
                new_vehicles[route1][idx1], new_vehicles[route2][idx2] = \
                    new_vehicles[route2][idx2], new_vehicles[route1][idx1]
        
        elif move_type == 'swap_intra_route':
            # Swap customers within the same route
            route = random.randint(0, self.num_vehicles - 1)
            if len(new_vehicles[route]) > 1:
                idx1, idx2 = random.sample(range(len(new_vehicles[route])), 2)
                new_vehicles[route][idx1], new_vehicles[route][idx2] = \
                    new_vehicles[route][idx2], new_vehicles[route][idx1]
        
        elif move_type == 'relocate':
            # Move a customer from one route to another
            route1, route2 = random.sample(range(self.num_vehicles), 2)
            if new_vehicles[route1]:
                customer = random.choice(new_vehicles[route1])
                new_vehicles[route1].remove(customer)
                new_vehicles[route2].append(customer)
        
        elif move_type == 'route_shuffle':
            # Shuffle an entire route
            route = random.randint(0, self.num_vehicles - 1)
            if len(new_vehicles[route]) > 1:
                random.shuffle(new_vehicles[route])
        
        elif move_type == '2-opt':
            # 2-opt optimization for a route
            route = random.randint(0, self.num_vehicles - 1)
            if len(new_vehicles[route]) > 2:
                i, j = sorted(random.sample(range(len(new_vehicles[route])), 2))
                new_vehicles[route][i:j+1] = reversed(new_vehicles[route][i:j+1])
        
        return new_vehicles

    def is_solution_feasible(self, vehicles: List[List[int]]) -> bool:
        """
        Check if the solution meets constraints
        """
        for route in vehicles:
            # Route distance constraint
            route_distance = self.distance_matrix[0][route[0]]
            for i in range(len(route) - 1):
                route_distance += self.distance_matrix[route[i]][route[i+1]]
            route_distance += self.distance_matrix[route[-1]][0]
            
            if route_distance > self.max_route_distance:
                return False
        
        return True

    def simulated_annealing(self) -> List[List[int]]:
        """
        Main Simulated Annealing algorithm
        """
        # Initialize solution
        current_solution = self.initialize_solution()
        current_cost, _ = self.calculate_cost(current_solution)
        
        # Best solution tracking
        best_solution = current_solution
        best_cost = current_cost
        
        # Temperature and iteration tracking
        temperature = self.initial_temperature
        no_improvement_counter = 0
        
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor_solution = self.generate_neighbor(current_solution)
            neighbor_cost, _ = self.calculate_cost(neighbor_solution)
            
            # Calculate cost difference
            delta = neighbor_cost - current_cost
            
            # Acceptance probability
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                # Update best solution
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
            
            # Track iteration costs
            self.iteration_costs.append(current_cost)
            
            # Adaptive cooling
            temperature *= self.cooling_rate
            
            # Early stopping condition
            if no_improvement_counter > self.max_iterations // 10:
                break
        
        # Store best solution
        self.best_solution = best_solution
        self.best_cost = best_cost
        
        return best_solution

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Perform multiple runs of Simulated Annealing
        """
        num_runs = 5  # Number of independent runs
        best_overall_solution = None
        best_overall_cost = float('inf')
        
        for _ in range(num_runs):
            solution = self.simulated_annealing()
            cost, _ = self.calculate_cost(solution)
            
            if cost < best_overall_cost:
                best_overall_solution = solution
                best_overall_cost = cost
        
        return best_overall_solution, best_overall_cost

def main():
    # Example usage
    def input_data():
        N, K = [int(x) for x in input().split()]
        distance_matrix = []
        for _ in range(N + 1):
            distance_matrix.append([int(x) for x in input().split()])
        return N, K, distance_matrix

    # Input data
    N, K, distance_matrix = input_data()
    
    # Create solver
    solver = VRPSolver(
        distance_matrix=distance_matrix, 
        num_vehicles=K,
        max_route_distance=50,  # Optional: Add route distance constraint
        initial_temperature=10000,
        cooling_rate=0.995
    )
    
    # Solve the problem
    best_solution, best_cost = solver.solve()
    
    # Print results
    # print("Best Solution:")
    # for i, route in enumerate(best_solution):
    #     print(f"Vehicle {i+1} route: {route}")
    # print(f"\nBest Cost: {best_cost}")
    print(K)
    for vehicle in best_solution:
        print(len(vehicle) + 1)
        print(0, end=" ")
        print(*vehicle)
main()