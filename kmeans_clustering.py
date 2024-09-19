import numpy as np
import argparse
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        # Initialize data, clusters and iterations are adjusted according to previewing data and adjusting accordingly
        self.model = None
        self.data = None
        self.n_clusters = 3
        self.max_iter = 999 #idk, pretty high I guess

        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        # Load the dataset
        mat = scipy.io.loadmat(self.dataset_file)
        # print(f"MAT KEYS: {mat.keys()}")
        self.data = mat['X']
        
    def model_fit(self):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        # Initialize K-means model
        self.model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=50)
        
        # Fit the model to yadadadadada
        self.model.fit(self.data)

        # cluster_centers = np.array([[0,0]])
        cluster_centers = self.model.cluster_centers_
        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()

    classifier = MykmeansClustering(args.dataset_file)

    clusters_centers = classifier.model_fit()

    print(clusters_centers)

    '''
    Plot data to screen using matplotlib, X's indicate the center (centroid)
    of the given cluster
    '''
    # plt.scatter(classifier.data[:, 0], classifier.data[:, 1], c=classifier.model.labels_, cmap='autumn') # pretty!
    # plt.scatter(clusters_centers[:, 0], clusters_centers[:, 1], s=100, c='black', marker='o')
    # plt.title('K-means Clustering')
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.style.use('seaborn')
    # plt.show()
    