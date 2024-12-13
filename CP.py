from ortools.sat.python import cp_model
def Input():
    filename = 'input_1.txt'
    with open(filename, 'r') as f:
        N, K = list(map(int, f.readline().split()))
        d = []
        temp = []
        for i in range (N+1):
            temp = list(map(int, f.readline().split()))
            for j in range (K):
                temp.append(temp[0])
            d.append(temp[1:])
#tạo depot ảo cho mỗi xe
        extend = d[0]
        for k in range (K):
            d.append(extend)
    return N, K, d[1:]

N, K, d = Input()
print(K)

model = cp_model.CpModel()
# x = 1 nếu xe k đi từ i đến j, 0 nếu không
# y = 1 nếu xe k đi từ i, 0 nếu không
# z là tổng khoảng cách mà xe k đi
# obj là hàm objective
x = [[[model.NewIntVar(0, 1, f'x[{i}][{j}][{k}]') for k in range(K)] for j in range(N+K)] for i in range(N+K)]
y = [[model.NewIntVar(0, 1, f'y[{i}][{k}]') for k in range(K)] for i in range (N+K)]
z = [model.NewIntVar(0, 10000, f'z[{i}]') for i in range(K)]
obj = model.NewIntVar(0, 10000, 'obj')

#xe k phải đến được depot ảo của nó
for k in range (K):
    model.Add(y[N+k][k] == 1)

#không thể tự quay lại chính node đó
for k in range (K):
    for i in range(N+K):
        model.Add(x[i][i][k]==0)

#mỗi node được phục vụ bởi 1 xe
for i in range (N+K):
    model.Add(sum(y[i][k] for k in range(K)) == 1)

#tổng vào phải bằng tổng ra (= 0 or 1)
for i in range (N+K):
    for k in range (K):
        model.Add(sum(x[i][j][k] for j in range (N+K)) == y[i][k])

for j in range (N+K):
    for k in range (K):
        model.Add(sum(x[i][j][k] for i in range (N+K)) == y[j][k])

# chắc chắn là đi từ i đến j thì không được quay lại từ j về i
for k in range (K):
    for i in range (N+K):
        for j in range (N+K):
            b = model.NewBoolVar(f'b[{i}][{j}][{k}]')
            model.Add(x[i][j][k] == 1).OnlyEnforceIf(b)
            model.Add(x[i][j][k] == 0).OnlyEnforceIf(b.Not())
            model.Add(y[j][k] == 1).OnlyEnforceIf(b)
            model.Add(x[j][i][k] == 0).OnlyEnforceIf(b)

#loại bỏ subtour
u = [[model.NewIntVar(0, N - 1, f'u[{i}][{k}]') for k in range(K)] for i in range(N+K)]

for k in range(K):
    for i in range(N + K):
        for j in range(N + K):
            if i != j:
                model.Add(u[j][k] - u[i][k] <= (N - 1) * (1 - x[i][j][k]))
                
#thiết lập hàm mục tiêu
for k in range (K):
    model.Add(z[k] == sum(d[i][j]*x[i][j][k] for i in range(N+K) for j in range(N+K)))
    model.Add(obj >= z[k])

model.Minimize(obj)
solve = cp_model.CpSolver()
status = solve.Solve(model)
# print(status)

# if status == cp_model.OPTIMAL:
#     print(solve.ObjectiveValue())
#     for k in range (K):
#         print(k)
#         res = []
#         for i in range (N+K):
#             for j in range (N+K):
#                 if solve.Value(x[i][j][k]) == 1:
#                     res.append(i)
#                     res.append(j)

#         res = set(res)
#         for i in res:
#             print(i, end=" ")
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(K)  # In ra số lượng xe
    for k in range(K):
        route = []
        for i in range(N + K):
            for j in range(N + K):
                if solve.Value(x[i][j][k]) == 1:
                    route.append((i, j))
        
        # Xây dựng chu trình đầy đủ từ các cạnh
        visited = set()
        current = N + k  # Bắt đầu từ depot ảo
        full_route = []
        while current not in visited:
            visited.add(current)
            full_route.append(current)
            next_node = None
            for i, j in route:
                if i == current and j not in visited:
                    next_node = j
                    break
            current = next_node
        
        # Loại bỏ depot ảo khỏi danh sách lộ trình
        full_route = []
        for node in route:
            if node <N:
                full_route.append(node)
            else:
                full_route.append(node - N)

        # In số lượng điểm và lộ trình
        print(len(full_route))
        print(" ".join(map(str, full_route)))
else:
    print("No feasible solution found.")

