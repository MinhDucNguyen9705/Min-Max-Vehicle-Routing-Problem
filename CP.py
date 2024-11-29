from ortools.sat.python import cp_model

from itertools import permutations

def Input():
    filename = 'input.txt'
    with open(filename, 'r') as f:
        N, K = list(map(int, f.readline().split()))
        d = []
        temp = []
        # extend = []
        for i in range (N+1):
            temp = list(map(int, f.readline().split()))
            for j in range (K):
                temp.append(temp[0])
            # extend.append(temp[-1])
            d.append(temp[1:])
        # extend = extend[1:]
        extend = d[0]
        # for k in range (K):
        #     extend.append(0)
        for k in range (K):
            d.append(extend)
    return N, K, d[1:]

N, K, d = Input()

print(N, K)

print(d)

def get_all_subsets(input_set):
    # Convert the input set to a list for easier indexing
    elements = list(input_set)
    # The total number of subsets is 2^n (where n is the size of the set)
    subsets = []
    num_subsets = 2 ** len(elements)
    
    for i in range(num_subsets):
        subset = {elements[j] for j in range(len(elements)) if (i & (1 << j)) != 0}
        subsets.append(subset)
    
    return subsets

def get_all_subsets_of_size(input_set, size):
    all_subsets = get_all_subsets(input_set)
    return [subset for subset in all_subsets if len(subset) >= size]

model = cp_model.CpModel()

x = [[[model.NewIntVar(0, 1, f'x[{i}][{j}][{k}]') for k in range(K)] for j in range(N+K)] for i in range(N+K)]
y = [[model.NewIntVar(0, 1, f'y[{i}][{k}]') for k in range(K)] for i in range (N+K)]
z = [model.NewIntVar(0, 10000, f'z[{i}]') for i in range(K)]
obj = model.NewIntVar(0, 10000, 'obj')

for k in range (K):
    model.Add(y[N+k][k] == 1)

for k in range (K):
    for i in range(N+K):
        model.Add(x[i][i][k]==0)

for i in range (N+K):
    model.Add(sum(y[i][k] for k in range(K)) == 1)

for i in range (N+K):
    for k in range (K):
        model.Add(sum(x[i][j][k] for j in range (N+K)) == y[i][k])

for j in range (N+K):
    for k in range (K):
        model.Add(sum(x[i][j][k] for i in range (N+K)) == y[j][k])

for k in range (K):
    for i in range (N+K):
        for j in range (N+K):
            b = model.NewBoolVar(f'b[{i}][{j}][{k}]')
            model.Add(x[i][j][k] == 1).OnlyEnforceIf(b)
            model.Add(x[i][j][k] == 0).OnlyEnforceIf(b.Not())
            model.Add(y[j][k] == 1).OnlyEnforceIf(b)
            model.Add(x[j][i][k] == 0).OnlyEnforceIf(b)

# for k in range (K):
#     for i in range (N+K):
#         model.Add(x[i][N+k][k] == 0)

all_nodes = set(range(N+K))
print(all_nodes)

for subset in get_all_subsets_of_size(all_nodes, 2):
    combs = list(permutations(subset, 2))
#     # print(combs)
#     # for (i, j) in combs:
#     #     print(i, j)
    for k in range (K):
        model.add(sum(x[i][j][k] for (i, j) in combs if i<N and j<N) <= len(subset)-1)

for k in range (K):
    model.Add(z[k] == sum(d[i][j]*x[i][j][k] for i in range(N+K) for j in range(N+K)))
    model.Add(obj >= z[k])

model.Minimize(obj)

solve = cp_model.CpSolver()

status = solve.Solve(model)

print(status)
if status == cp_model.OPTIMAL:
    print(solve.ObjectiveValue())
    for k in range (K):
        print(k)
        for i in range (N+K):
            for j in range (N+K):
                if solve.Value(x[i][j][k]) == 1:
                    print(i, j)