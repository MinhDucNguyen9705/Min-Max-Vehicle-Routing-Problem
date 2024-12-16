filename = "test.txt"
with open(filename, 'r') as f:
    n, k = map(int, f.readline().split())
    nodes = []
    for i in range (n):
        # print(f.readline().split())
        id, x, y, name = f.readline().split()
        x = float(x)
        y = float(y)
        nodes.append([x, y])
        # print(d)

from math import radians, cos, sin, asin, sqrt
def distance(lat1, lat2, lon1, lon2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return round(c * r)

d = []

for i in range (len(nodes)):
    row = []
    for j in range (len(nodes)):
        if i==j:
            row.append(0)
        else:
            row.append(distance(nodes[i][0], nodes[j][0], nodes[i][1], nodes[j][1]))
    d.append(row)

print(n, k)
print(d[0])

with open("inputs/belgium-n50-k10.txt", 'w') as f:
    f.write(f"{n} {k}\n")
    for row in d:
        # Convert each list to a comma-separated string and write to file
        f.write(" ".join(map(str, row)) + "\n")