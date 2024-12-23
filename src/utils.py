'''
Utilities for the analysis.
'''
import h5flow
import h5py
import numpy as np

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

def get_Eventdata(h5flow_data, evid: int):
    calib_data = h5flow_data['charge/events', 'charge/calib_prompt_hits', evid]
    return calib_data

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

def drawEvent(arr_evtData):
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
    plt.savefig('event.png')
    plt.close()