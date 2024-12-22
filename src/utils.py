'''
Utilities for the analysis.
'''
import h5flow
import h5py

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

def get_data(h5flow_data, evid: int):
    calib_data = h5flow_data['charge/events', 'charge/calib_prompt_hits', evid]
    return calib_data

def drawEvent(evtData):
    import matplotlib.pyplot as plt
    plt.ioff()
    x = evtData.data['x'].flatten()
    y = evtData.data['y'].flatten()
    z = evtData.data['z'].flatten()
    charge = evtData.data['Q'].flatten()

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