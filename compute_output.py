class computeOutput():

    def load_input_file(self, filename):
        with open (filename, 'r') as f:
            n, k = list(map(int, f.readline().split()))
            d = []
            for i in range (n+1):
                d.append(list(map(float, f.readline().split())))
        self.n = n          # Số lượng node
        self.k = k          # Số lượng bưu tá
        self.d = d          # Ma trận khoảng cách   

    def load_input_line(self):
        n, k = list(map(int, input().split()))
        d = []
        for i in range (n+1):
            d.append(list(map(float, input().split())))
        self.n = n          # Số lượng node
        self.k = k          # Số lượng bưu tá
        self.d = d          # Ma trận khoảng cách   

    def parse_output(self):
        ans = 0
        k = int(input())
        for i in range (k):
            total = 0
            nodes = int(input())
            path = list(map(int, input().split()))
            for j in range (1,len(path)):
                total += self.d[path[j-1]][path[j]]
            # total += self.d[path[-1]][path[0]]
            ans = max(total, ans)
            print(total)
        print(ans)

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

if __name__ == "__main__":
    compute = computeOutput()
    compute.load_input_file('inputs/input7.txt')
    # compute.load_input_line()
    compute.parse_output()