from ortools.sat.python import cp_model

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

model = cp_model.CpModel()

x = [[[model.NewIntVar(0, 1, f'x[{i}][{j}][{k}]') for k in range(K)] for j in range(N+K)] for i in range(N+K)]
z = [model.NewIntVar(0, K-1, f'z[{i}]') for i in range(N+K)]
y = [model.NewIntVar(0, 10000, f'y[{i}]') for i in range(K)]
obj = model.NewIntVar(0, 10000, 'obj')

for k in range (K):
    model.add(z[N+k] == k)

for i in range(N+K):
    model.Add(sum(x[i][j][k] for j in range(N+K) for k in range(K)) == 1)
    model.Add(sum(x[j][i][k] for j in range(N+K) for k in range(K)) == 1)

# for k in range (K):
#     for i in range (N+K):
#         model.Add(sum(x[i][j][k] for j in range(N+K)) == sum(x[j][i][k] for j in range(N+K)))

for k in range (K):
    for i in range (N+K):
        for j in range (N+K):
            b = model.NewBoolVar(f'b[{i}][{j}][{k}]')
            model.Add(x[i][j][k] == 1).OnlyEnforceIf(b)
            model.Add(x[i][j][k] == 0).OnlyEnforceIf(b.Not())
            model.Add(z[i]==k).OnlyEnforceIf(b)
            model.Add(z[j]==k).OnlyEnforceIf(b)
            model.Add(x[j][i][k] == 0).OnlyEnforceIf(b)

for k in range (K):
    for i in range (N+K):
        model.Add(x[i][i][k] == 0)

# for k in range (K):
#     for i in range (N+K):
#         model.Add(x[i][N+k][k] == 0)

for k in range (K):
    model.Add(y[k] == sum(d[i][j]*x[i][j][k] for i in range(N+K) for j in range(N+K)))
    model.Add(obj >= y[k])

model.Minimize(obj)
# obj = model.NewIntVar(0, 1000000, 'obj')

# model.add(obj == sum(d[i][j]*x[i][j][k] for i in range(N+K) for j in range(N+K) for k in range(K)))

solver = cp_model.CpSolver()
status = solver.Solve(model)

print(status)
print(solver.ObjectiveValue())

def print_path(begin, path):
    ans = f"0"
    curr = begin
    added = set()
    while len(added)!=len(path):
        for i in range(len(path)):
            if path[i][0] == curr and i not in added:
                added.add(i)
                if path[i][1] < N:
                    ans += " " + str(path[i][1]+1)
                else:
                    ans += " " 
                curr = path[i][1]
    return ans

if status == cp_model.OPTIMAL:
    print('Objective value =', solver.ObjectiveValue())
    for k in range(K):
        total = 0
        path = []
        for i in range(N+K):
            for j in range(N+K):
                if solver.Value(x[i][j][k]) == 1:
                    # print(f'x[{i}][{j}][{k}] = 1')
                    total += d[i][j]
                    path.append((i, j))
        # print(f'y[{k}] = {total}')
        print(len(path)-1)
        print(print_path(N+k, path))
    # for i in range(N+K):
    #     print(f'z[{i}] = {solver.Value(z[i])}')
