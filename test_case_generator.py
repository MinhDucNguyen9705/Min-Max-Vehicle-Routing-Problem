import random 

filename = 'input.txt'

n = 1000
k = random.randint(1, n//2)

d = []
for i in range (n+1):
    temp = []
    for j in range (n+1):
        if i==j:
            temp.append(0)
        else:
            temp.append(random.randint(1, 1000))
    d.append(temp)

with open (filename, 'w') as f:
    f.write(f'{n} {k}\n')
    for i in range (n+1):
        for j in range (n+1):
            f.write(f'{d[i][j]} ')
        f.write('\n')