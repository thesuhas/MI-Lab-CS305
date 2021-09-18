import numpy as np
import math


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """

        dist = []

        for i in range(len(x)):
            # Get point
            pt = x[i]
            # Get array for each iteration
            d = list()
            for j in range(len(self.data)):
                # Difference of two points
                diff = abs(np.subtract(self.data[j], pt))
                # Minowski Distance Factor
                diff = np.power(diff, self.p)
                # Distance
                minowski = np.sum(diff)
                d.append(math.pow(minowski, (1/self.p)))
            dist.append(d)
        
        return dist
        
        

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        distances = self.find_distance(x)
        neigh_dists = []
        neigh_idx = []

        for i in range(len(distances)):
            mapp = sorted(zip(distances[i], [j for j in range(len(distances[i]))]))
            neigh_dists.append([mapp[j][0] for j in range(len(mapp[:self.k_neigh]))])
            neigh_idx.append([mapp[j][1] for j in range(len(mapp[:self.k_neigh]))])

        return [neigh_dists, neigh_idx]
            
        

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        [dist, idx] = self.k_neighbours(x)
        epsilon = math.pow(10, -7)
        pred = []

        # If weighted KNN
        if self.weighted:
            for i in range(len(dist)):
                classes = dict()

                for j in range(len(idx[i])):
                    # If that class exists, add additional weight to that class
                    if self.target[idx[i][j]] in classes:
                        classes[self.target[idx[i][j]]] += 1/(dist[i][j] + epsilon)
                    # If that class does not exist, add it to the dict
                    else:
                        classes[self.target[idx[i][j]]] = 1/(dist[i][j] + epsilon)
                # Finding max key in dictionary and adding to pred
                pred.append(max(zip(classes.values(), classes.keys()))[1])
        # If vanilla KNN
        else:
            for i in range(len(idx)):
                pred.append(self.target[max(set(idx[i]), key=idx[i].count)])
            
        return pred
   

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        
        pred = self.predict(x)

        count = 0

        for i in range(len(pred)):
            if pred[i] == y[i]:
                count += 1
        
        return (count / len(pred)) * 100
