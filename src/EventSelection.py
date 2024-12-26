'''
    Author: radofana@gmail.com
    Date: 2023-10-05 14:30:00
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils import drawEvent, _PCA, DBSCAN_clustering

import numpy.lib.recfunctions

class TopologyClassification():
    def __init__(self, data, evtID=0, cuts={'track_length': 10,
                                    'Nhits': 100,
                                    'threshold': 3.3}):
        self.data = data
        self.dtype = self.data.dtype
        self.cuts = cuts
        self.evtID = evtID

    def __NhitsMoreThan(self):
        moreHits = True
        if len(self.data['Q'].flatten()) < self.cuts['Nhits']:
            moreHits = False
        return moreHits
    
    def clustering(self, data, cluster_labels):
        # clustering
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            selected_indices = np.where(cluster_labels == label)[0]
            mask = np.zeros(len(data['x']), dtype=bool)
            mask[selected_indices] = True
            filtered_data = np.zeros(len(data['x'][mask]), dtype=self.dtype)
            filtered_data[list(self.dtype.names)] = data[list(self.dtype.names)][mask]
            # Plot the filtered data in 3 different canvases
            fig_xy, ax_xy = plt.subplots()
            ax_xy.scatter(data['x'], data['y'], c='blue', label='all data', s=5)
            ax_xy.scatter(filtered_data['x'], filtered_data['y'], c='red', label='Selected', s=5)
            ax_xy.set_xlabel('X')
            ax_xy.set_ylabel('Y')
            ax_xy.legend()
            ax_xy.set_title('XY View')
            ax_xy.set_xlim([-70, 70])
            ax_xy.set_ylim([-70, 70])
            output_dir = "../output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(os.path.join(output_dir, "filtered_data_xy_view_evt{}_Cluster{}.png".format(self.evtID, label)))

            fig_xz, ax_xz = plt.subplots()
            ax_xz.scatter(filtered_data['x'], filtered_data['z'], c='red', label='Selected', s=5)
            ax_xz.set_xlabel('X')
            ax_xz.set_ylabel('Z')
            ax_xz.legend()
            ax_xz.set_title('XZ View')
            ax_xz.set_xlim([-70, 70])
            ax_xz.set_ylim([-70, 70])
            plt.savefig(os.path.join(output_dir, "filtered_data_xz_view_evt{}_Cluster{}.png".format(self.evtID, label)))

            fig_yz, ax_yz = plt.subplots()
            ax_yz.scatter(filtered_data['y'], filtered_data['z'], c='red', label='Selected', s=5)
            ax_yz.set_xlabel('Y')
            ax_yz.set_ylabel('Z')
            ax_yz.legend()
            ax_yz.set_title('YZ View')
            ax_yz.set_xlim([-70, 70])
            ax_yz.set_ylim([-70, 70])
            plt.savefig(os.path.join(output_dir, "filtered_data_yz_view_evt{}_Cluster{}.png".format(self.evtID, label)))


    def selectMainTrack(self, evtData, resultPCA):
        if not self.__NhitsMoreThan():
            print("Not enough hits for classification.")
            return -1, -1
        mean = resultPCA['mean'][0]
        principal_component = resultPCA['principal_component'][0]
        print(resultPCA['mean'])
        print(principal_component)
        x = evtData['x']
        y = evtData['y']
        z = evtData['z']
        # covariance matrix
        original_data = np.vstack((x, y, z)).T
        # Calculate the projection of each point onto the extrapolated principal component
        projections = np.dot(original_data - mean, principal_component)[:, np.newaxis] * principal_component + mean

        # Calculate the distance of each point to the principal component
        distances = np.linalg.norm(original_data - projections, axis=1)

        # Select points within the threshold distance to the principal component
        selected_indices = np.where(distances <= self.cuts['threshold'])[0]

        # Create a boolean mask for the selected points
        mask = np.zeros(len(evtData['x']), dtype=bool)
        mask[selected_indices] = True

        # Create a new structured array with the filtered length
        filtered_data = np.zeros(len(evtData['x'][mask]), dtype=self.dtype)
        
        # Assign the filtered values to the new structured array
        filtered_data[list(self.dtype.names)] = evtData[list(self.dtype.names)][mask]
        #
        # Mask for the unfiltered data
        mask_data = np.ones(len(evtData['x']), dtype=bool)
        mask_data[selected_indices] = False
        data = np.array(evtData[mask_data], dtype=self.dtype)

        # Plot the original data and the filtered data in 2D views
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        output_dir = "../output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # XY view
        # axs[0].scatter(evtData['x'], evtData['y'], c='gray', label='Unselected', alpha=1, s=5)
        # axs[0].scatter(filtered_data['x'], filtered_data['y'], c='red', label='Selected', s=5)
        # Draw the principal component vector
        axs[0].quiver(mean[0], mean[1], principal_component[0], principal_component[1], 
              angles='xy', scale_units='xy', scale=10, color='blue', width=0.005)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].legend()
        axs[0].set_title('XY View')
        axs[0].set_xlim([-70, 70])
        axs[0].set_ylim([-70, 70])

        # XZ view
        axs[1].scatter(evtData['x'], evtData['z'], c='gray', label='Unselected', alpha=1, s=5)
        axs[1].scatter(filtered_data['x'], filtered_data['z'], c='red', label='Selected', s=5)
        # Draw the principal component vector
        axs[1].quiver(mean[0], mean[2], principal_component[0], principal_component[2], 
              angles='xy', scale_units='xy', scale=10, color='blue', width=0.005)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')
        axs[1].legend()
        axs[1].set_title('XZ View')
        axs[1].set_xlim([-70, 70])
        axs[1].set_ylim([-70, 70])

        # YZ view
        axs[2].scatter(evtData['z'], evtData['y'], c='gray', label='Unselected', alpha=1, s=5)
        axs[2].scatter(filtered_data['z'], filtered_data['y'], c='red', label='Selected', s=5)
        axs[2].set_xlabel('Z')
        axs[2].set_ylabel('Y')
        axs[2].legend()
        axs[2].set_title('YZ View')
        axs[2].set_xlim([-70, 70])
        axs[2].set_ylim([-70, 70])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "filtered_data_2d_views_evt{}.png".format(self.evtID)))
        return data, filtered_data
    
    def event2tracks(self):
        data = self.data.copy() # copy the data
        dtype_list = self.dtype.descr # get the dtype list
        dtype_list.append(('trackID', 'int32')) # add a new field 'trackID' to the dtype list
        tracks = np.array([], dtype=np.dtype(dtype_list)) # create an empty array for the tracks
        print(tracks.dtype)
        track_ID_counter = 0
        for i in range(2):
            resultPCA = PCA(data=data)
            data, filtered_data = self.selectMainTrack(evtData=data, resultPCA=resultPCA)
            if type(data) != int:
                track_ids = [track_ID_counter for _ in range(len(filtered_data['x']))]
                filtered_data = np.lib.recfunctions.append_fields(filtered_data, 'trackID', track_ids, usemask=False)
                tracks = np.append(tracks, filtered_data)
                track_ID_counter += 1
            else:
                break
        print(len(np.unique(tracks['trackID'])))

    def plot_views(self, data):
        output_dir = "../output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # XY view
        plt.figure()
        plt.scatter(data['x'], data['y'], c=data['Q'], cmap='viridis', label='selected', alpha=1, s=5)
        plt.colorbar(label='Charge Q')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('XY View')
        plt.xlim([-70, 70])
        plt.ylim([-70, 70])
        plt.savefig(os.path.join(output_dir, "data_xy_view_evt{}.png".format(self.evtID)))

        # XZ view
        plt.figure()
        plt.scatter(data['x'], data['z'], c=data['Q'], cmap='viridis', label='selected', alpha=1, s=5)
        plt.colorbar(label='Charge Q')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend()
        plt.title('XZ View')
        plt.xlim([-70, 70])
        plt.ylim([-70, 70])
        plt.savefig(os.path.join(output_dir, "data_xz_view_evt{}.png".format(self.evtID)))

        # YZ view
        plt.figure()
        plt.scatter(data['y'], data['z'], c=data['Q'], cmap='viridis', label='selected', alpha=1, s=5)
        plt.colorbar(label='Charge Q')
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.legend()
        plt.title('YZ View')
        plt.xlim([-70, 70])
        plt.ylim([-70, 70])
        plt.savefig(os.path.join(output_dir, "data_yz_view_evt{}.png".format(self.evtID)))

    # Run the classification
    def classify(self):
        # resultPCA = PCA(data=self.data)
        print(self.dtype)
        result_pca = _PCA(data=self.data)
        print(result_pca)
        dbscan = DBSCAN_clustering(data=self.data, eps=3, min_samples=5)
        # print(dbscan)
        self.clustering(data=self.data, cluster_labels=dbscan)
        # if type(result_pca) == int:
        #     return -1
        # data, filtered_data = self.selectMainTrack(evtData=self.data, resultPCA=result_pca)
        # if type(data) == int:
        #     return -1
        drawEvent(self.data, self.evtID)
        # # print(data)
        # self.plot_views(filtered_data)
        # self.event2tracks()
