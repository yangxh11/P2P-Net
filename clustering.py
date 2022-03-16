import numpy as np
import itertools

class PartsResort():
    def __init__(self, num_center, feature_dim):
        super(PartsResort, self).__init__()
        self.num_center = num_center
        self.feature_dim = feature_dim

        self.centers = np.zeros([num_center, feature_dim])
        self.count = 0   

        self.permutations = list(itertools.permutations(range(num_center)))

    def update(self, points, order):
        batch = points.shape[0]

        # [batch, topN, feature_dim]
        resorted_points = np.zeros_like(points)
        for i in range(batch):
            resorted_points[i] = points[i][order[i], :]
        
        # [topN, feature_dim]
        resorted_points = np.mean(resorted_points, axis=0)
        for i in range(self.num_center):
            self.centers[i] = (self.centers[i]*self.count*0.9 + resorted_points[i]*batch) / (self.count*0.9 + batch)
        self.count += batch

    def classify(self, points, is_train):  
        # input: points [batch, topN, feature_dim]
        # output: [batch, topN]
        batch, topN, _ = points.shape
        if np.sum(self.count) == 0:
            order = np.stack([list(range(topN))]*batch, axis=0)
            # self.update(points, order)
        else:
            order = np.zeros([batch, topN], dtype=np.int)
            for i in range(points.shape[0]):
                topn_points = points[i]
                order[i] = self.graph_assign(topn_points)
        if is_train:
            self.update(points, order)
        return order

    def graph_assign(self, topn_points):
        adj_matrix_center = np.dot(self.centers, self.centers.transpose())
        adj_matrix = np.dot(topn_points, topn_points.transpose())
        adj_matrix_center = adj_matrix_center / adj_matrix_center.max()
        adj_matrix = adj_matrix / adj_matrix.max()
        
        max_similarity = 0
        order = list(range(self.num_center))
        for perm in self.permutations:
            adj_matrix = adj_matrix[:, perm][perm, :]
            prod = np.sum(adj_matrix_center * adj_matrix)
            if prod > max_similarity:
                max_similarity = prod
                order = list(perm)
            # print(max_similarity, prod, order)
        return order



if __name__ == "__main__":
    PC = PartsResort(6, 105)

    points = np.random.randint(0, 10, size=[2, 6, 105])
    PC.classify(points)
    points = np.random.randint(0, 10, size=[2, 6, 105])
    PC.classify(points)
