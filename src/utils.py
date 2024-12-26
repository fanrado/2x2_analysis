'''
Utilities for the analysis.
'''
import sys, os
import matplotlib.pyplot as plt
import h5flow
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def list_hdf5_contents(file_path):
    with h5py.File(file_path, 'r') as file:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        file.visititems(print_attrs)

def Read_Flow(root_path: str, filename: str):
  h5flow_data = h5flow.data.H5FlowDataManager('/'.join([root_path, filename]), 'r')
  return h5flow_data

def getNEvents(h5flow_data):
    return len(h5flow_data['charge/events', 'charge/calib_prompt_hits'])

def get_Eventdata(h5flow_data, evid: int):
    NEvents = getNEvents(h5flow_data=h5flow_data)
    calib_data = h5flow_data['charge/events', 'charge/calib_prompt_hits', evid][0]
    calib_data = np.array(calib_data, dtype=calib_data.dtype)
    return calib_data, NEvents

def Eventdata2Array(evtdata, evtid: int):
    x = evtdata.data['x'].flatten()
    y = evtdata.data['y'].flatten()
    z = evtdata.data['z'].flatten()
    Nhits = len(x)
    charge = evtdata.data['Q'].flatten()
    dtype = [('evtID', 'i4'), ('x', 'f4', (Nhits,)), ('y', 'f4', (Nhits,)), ('z', 'f4', (Nhits,)), ('Q', 'f4', (Nhits,))]
    data = [(evtid, x, y, z, charge)]
    arr = np.array(data, dtype=dtype)
    return arr

def _PCA(data):
    x = data['x']
    y = data['y']
    z = data['z']
    # covariance matrix
    data_matrix = np.vstack((x, y, z)).T
    pca = PCA(n_components=1)
    pca.fit(data_matrix)
    mean = pca.mean_
    principal_component = pca.components_[0]
    result = np.array([(mean, principal_component)], 
            dtype=[('mean', '3float64'), ('principal_component', '3float64')])
    
    return result

def DBSCAN_clustering(data, eps=1.0, min_samples=5):
    x = data['x']
    y = data['y']
    z = data['z']
    data_matrix = np.vstack((x, y, z)).T
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_matrix)

    return clustering.labels_

def SpatialClustering(data, eps=1.0, min_samples=5):
    dtype = data.dtype
    labels = DBSCAN_clustering(data, eps=eps, min_samples=min_samples)
    # clustering
    unique_labels = np.unique(labels)
    print(unique_labels)
    # tracks = []
    tracks = np.empty(len(unique_labels), dtype='O')
    for ilabel, label in enumerate(unique_labels):
        # if label !=-1:
        # print(label)
        selected_indices = np.where(labels == label)[0]
        mask = np.zeros(len(data['x']), dtype=bool)
        mask[selected_indices] = True
        filtered_data = np.zeros(len(data['x'][mask]), dtype=dtype)
        filtered_data[list(dtype.names)] = np.array(data[list(dtype.names)][mask], dtype=dtype)
        # tracks.append(filtered_data)
        tracks[ilabel] = filtered_data

    return tracks


########################################################################################
def drawEvent(arr_evtData, evtID, filename='event'):
    import matplotlib.pyplot as plt
    plt.ioff()
    x = arr_evtData['x'].flatten()
    y = arr_evtData['y'].flatten()
    z = arr_evtData['z'].flatten()
    charge = arr_evtData['Q'].flatten()

    fig, ax = plt.subplots(1,3,figsize=(5*3,5))
    # xy projection
    sc0 = ax[0].scatter(x, y, c=charge, s=1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    plt.colorbar(sc0, ax=ax[0], label='Charge (Q)')
    # zy projection
    sc1 = ax[1].scatter(z, y, c=charge, s=1)
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('y')
    plt.colorbar(sc1, ax=ax[1], label='Charge (Q)')
    # xz projection
    sc2 = ax[2].scatter(x, z, c=charge, s=1)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('z')
    plt.colorbar(sc2, ax=ax[2], label='Charge (Q)')
    # ax.set_aspect('equal', 'box')
    plt.savefig('../output/{}{}.png'.format(filename, evtID))
    plt.close()

# Learning how PCA works
def custom_PCA(data):
    x = data['x']
    y = data['y']
    z = data['z']
    # covariance matrix
    data_matrix = np.vstack((x, y, z)).T
    covariance_matrix = np.cov(data_matrix, rowvar=False)
    print("Covariance matrix:\n", covariance_matrix)
    if np.isnan(covariance_matrix).any():
        print("Covariance matrix contains NaN values.")
        return -1
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

    result = np.array([(mean, principal_component)], 
            dtype=[('mean', '3float64'), ('principal_component', '3float64')])
    return result
########################################################################################