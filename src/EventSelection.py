'''
    Author: radofana@gmail.com
    Date: 2023-10-05 14:30:00
'''
import numpy as np
import matplotlib.pyplot as plt

class TopologyClassification():
    def __init__(self, data):
        self.data = data

    def __NhitsMoreThan(self, Nhits=500):
        moreHits = True
        if len(self.data['Q'].flatten()) < Nhits:
            moreHits = False
        return moreHits
    
    # Assuming the data has sufficient hits
    def PCA(self):
        x = self.data['x'].flatten()
        y = self.data['y'].flatten()
        z = self.data['z'].flatten()
        # covariance matrix
        data_matrix = np.vstack((x, y, z)).T
        covariance_matrix = np.cov(data_matrix, rowvar=False)
        print("Covariance matrix:\n", covariance_matrix)
        # eignvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        print("Eigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)

        # Calculate the mean of the data points
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)
        mean = np.array([mean_x, mean_y, mean_z])
        
        # Find the principal component
        principal_component_index = np.argmax(eigenvalues)
        principal_component = eigenvectors[:, principal_component_index]
        print("Principal component (eigenvector with the largest eigenvalue):\n", principal_component)
        
        # Project the data onto the principal component
        projected_data = np.dot(data_matrix - mean, principal_component)

        result = np.array([(mean, principal_component, projected_data)], 
                  dtype=[('mean', '3float64'), ('principal_component', '3float64'), ('projected_data', 'float64', projected_data.shape)])
        return result

    # Run the classification
    def classify(self):
        if not self.__NhitsMoreThan(Nhits=100):
            print("Not enough hits for classification.")
            return -1
        else:
            self.PCA()
