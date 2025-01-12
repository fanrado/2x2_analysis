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

def getNEvents(h5flow_data):
    return h5flow_data['charge/events/data'].shape[0]

def Read_Flow(root_path: str, filename: str):
  h5flow_data = h5flow.data.H5FlowDataManager('/'.join([root_path, filename]), 'r')
  return h5flow_data, getNEvents(h5flow_data)

def get_Eventdata(h5flow_data, evid: int):
    # NEvents = getNEvents(h5flow_data=h5flow_data)

    oneEventdata = h5flow_data['charge/events', 'charge/calib_prompt_hits', evid][0]
    try:
        oneEventdata = h5flow_data['charge/events', 'charge/calib_final_hits', evid][0]
    except:
        pass
    oneEventdata = np.array(oneEventdata, dtype=oneEventdata.dtype)
    return oneEventdata#, NEvents

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
    # data_matrix = np.vstack((x, y, z)).T
    data_matrix = np.vstack((x,y,z)).T
    success = True
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_matrix)
    except Exception as e:
        success = False
    
    if success:
        return clustering.labels_
    else:
        return np.array([])

def SpatialClustering(data, eps=1.0, min_samples=5):
    dtype = data.dtype
    labels = DBSCAN_clustering(data, eps=eps, min_samples=min_samples)
    success_clustering = True
    if len(labels)==0:
        success_clustering = False
    if success_clustering:
        # clustering
        unique_labels = np.unique(labels)
        print(unique_labels)
        # tracks = []
        tracks = np.empty(len(unique_labels), dtype='O')
        for ilabel, label in enumerate(unique_labels):
            # print(label)
            selected_indices = np.where(labels == label)[0]
            mask = np.zeros(len(data['x']), dtype=bool)
            mask[selected_indices] = True
            filtered_data = np.zeros(len(data['x'][mask]), dtype=dtype)
            filtered_data[list(dtype.names)] = np.array(data[list(dtype.names)][mask], dtype=dtype)
            # tracks.append(filtered_data)
            tracks[ilabel] = filtered_data
        return tracks
    else:
        return np.array([])

##################################################################################################################
###________Hough transform_____________
# def HoughTransform2D(trackData, itrack):
#     # We will use the yz plane only
#     # x = trackData['x']
#     rs = []
#     thetas = []
#     x = trackData['x']
#     y = trackData['y']
#     z = trackData['z']

import numpy as np

def hough_transform_from_points(points, theta_res=1, rho_res=1, x_range=None, y_range=None):
    """
    Implements the Hough Transform algorithm for line detection from a set of points.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of points with shape (N, 2) where each row is [x, y]
    theta_res : float
        Resolution of theta in degrees
    rho_res : float
        Resolution of rho in pixels
    x_range : tuple, optional
        (min_x, max_x) range of x coordinates. If None, calculated from points
    y_range : tuple, optional
        (min_y, max_y) range of y coordinates. If None, calculated from points
    
    Returns:
    --------
    accumulator : numpy.ndarray
        The Hough accumulator array
    thetas : numpy.ndarray
        Array of theta values used
    rhos : numpy.ndarray
        Array of rho values used
    """
    # Convert points to numpy array if not already
    points = np.asarray(points)
    
    # Calculate coordinate ranges if not provided
    if x_range is None:
        x_range = (np.min(points[:, 0]), np.max(points[:, 0]))
    if y_range is None:
        y_range = (np.min(points[:, 1]), np.max(points[:, 1]))
    
    # Calculate diagonal length for rho
    diag_len = int(np.ceil(np.sqrt(
        (x_range[1] - x_range[0])**2 + 
        (y_range[1] - y_range[0])**2
    )))
    
    # Create theta and rho arrays
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    
    # Initialize the accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Calculate sine and cosine for all theta values
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # For each point
    for x, y in points:
        # Calculate rho for all theta values
        rho_vals = x * cos_thetas + y * sin_thetas
        
        # Find the closest rho bin for each calculated rho value
        rho_idxs = np.round((rho_vals + diag_len) / rho_res).astype(int)
        
        # Increment accumulator cells
        for theta_idx in range(len(thetas)):
            rho_idx = rho_idxs[theta_idx]
            if 0 <= rho_idx < len(rhos):
                accumulator[rho_idx, theta_idx] += 1
    
    return accumulator, thetas, rhos

def find_peaks(accumulator, threshold=None, min_distance=10):
    """
    Find peaks in the Hough accumulator array.
    
    Parameters:
    -----------
    accumulator : numpy.ndarray
        The Hough accumulator array
    threshold : float, optional
        Minimum value for a peak to be considered
    min_distance : int
        Minimum distance between peaks
        
    Returns:
    --------
    peaks : numpy.ndarray
        Indices of peaks in the accumulator array (rho_idx, theta_idx)
    """
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)
    
    # Find local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(accumulator, size=min_distance) == accumulator
    
    # Apply threshold
    peaks = np.where((accumulator > threshold) & local_max)
    return np.column_stack(peaks)

def get_line_equations(peaks, thetas, rhos):
    """
    Convert peaks to line equations in the form (rho, theta).
    
    Parameters:
    -----------
    peaks : numpy.ndarray
        Peak coordinates from the accumulator (rho_idx, theta_idx)
    thetas : numpy.ndarray
        Array of theta values
    rhos : numpy.ndarray
        Array of rho values
    
    Returns:
    --------
    lines : numpy.ndarray
        Array of (rho, theta) pairs representing detected lines
    """
    lines = []
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]]
        lines.append((rho, theta))
    return np.array(lines)

def get_line_points(rho, theta, x_range=(-1000, 1000)):
    """
    Get two points on a line given rho and theta.
    
    Parameters:
    -----------
    rho : float
        Distance from origin to the line
    theta : float
        Angle in radians
    x_range : tuple
        Range of x coordinates to consider
        
    Returns:
    --------
    tuple
        Two points on the line ((x1, y1), (x2, y2))
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    
    x1 = int(x0 + x_range[0] * (-b))
    y1 = int(y0 + x_range[0] * (a))
    x2 = int(x0 + x_range[1] * (-b))
    y2 = int(y0 + x_range[1] * (a))
    
    return (x1, y1), (x2, y2)

def find_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            
            # Convert polar coordinates to Cartesian (ax + by = c)
            a1, b1 = np.cos(theta1), np.sin(theta1)
            a2, b2 = np.cos(theta2), np.sin(theta2)
            c1, c2 = rho1, rho2
            
            # Solve the system of linear equations
            A = np.array([[a1, b1], [a2, b2]])
            C = np.array([c1, c2])
            
            if np.linalg.det(A) != 0:  # Check if lines are not parallel
                intersection = np.linalg.solve(A, C)
                intersections.append(intersection)
    
    return intersections
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

if __name__=='__main__':
    # Example usage with point coordinates
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],  # Points forming a line
        [10, 0],
        [10, 10]  # Points forming a vertical line
    ])
    # Create a 2D plot of the points
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=100)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Points')
    plt.axis('equal')
    plt.savefig('original_points.png')
    plt.close()
    # Apply Hough Transform
    accumulator, thetas, rhos = hough_transform_from_points(
        points,
        theta_res=1,
        rho_res=1,
        x_range=(-20, 20),
        y_range=(-20, 20)
    )
    print('rho', rhos)
    # Find peaks in the accumulator
    peaks = find_peaks(accumulator, min_distance=5, threshold=4)

    # Get line equations
    lines = get_line_equations(peaks, thetas, rhos)

    # For each detected line, you can get points on the line
    x = []
    y = []
    for rho, theta in lines:
        point1, point2 = get_line_points(rho, theta)
        print(f"Line points: {point1} to {point2}")
        x.append(point1)
        y.append(point2)

    #
    intersections = find_intersections(lines=lines)

    # Plot original points and detected lines
    plt.figure(figsize=(10, 10))
    # plt.scatter(points[:, 0], points[:, 1], color='blue', s=100)
    for (x1, y1), (x2, y2) in zip(x, y):
        plt.plot([x1, x2], [y1, y2], 'r-', marker='.')
    for intersection in intersections:
        plt.scatter(*intersection, color='green', s=100)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with Detected Lines')
    plt.axis('equal')
    plt.savefig('transformed_points.png')
    plt.close()
    print(points[:,0], points[:,1])
    print(intersections)