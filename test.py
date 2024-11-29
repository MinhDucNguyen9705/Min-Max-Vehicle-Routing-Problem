from itertools import permutations

n = 20 
k = 6
d = [[0 for i in range (n+1)] for j in range (n+1)]
coordinates = []
with open('test.txt', 'r') as f:
    for i in range (n+1):
        temp = f.readline().split()
        node = int(temp[0])
        coordinates.append(list(map(int, temp[1][1:-1].split(","))))
        # print(temp)

for i in range (len(coordinates)):
    for j in range (len(coordinates)):
        d[i][j] = (abs(coordinates[i][0] - coordinates[j][0])**2 + abs(coordinates[i][1] - coordinates[j][1])**2)**0.5

# for i in range (len(d)):
#     for j in range (len(d)):
#         print(d[i][j], end = " ")
#     print()

with open("input.txt", "w") as f:
    f.write(f'{n} {k}\n')
    for i in range (n+1):
        for j in range (n+1):
            f.write(f'{d[i][j]} ')
        f.write('\n')