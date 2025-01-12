"""
testSelection.py

Author: radofana@gmail.com
Date: December 24, 2024
Time: 22:42

---------------------------------------------------------------------------------
Test of the source codes in ../src
"""

import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import get_Eventdata, Read_Flow
from Event import Event

if __name__=='__main__':
    root_path = '../../flowdata'
    # filename = "packet-0050015-2024_07_08_13_37_49_CDT.FLOW.hdf5"
    # filename = "packet-0050017-2024_07_09_01_04_39_CDT.FLOW.hdf5"
    # filename = "packet-0050018-2024_07_11_19_49_51_CDT.FLOW.hdf5"
    filename = "packet-0050018-2024_07_11_19_59_52_CDT.FLOW.hdf5"
    # filename = "packet-0050017-2024_07_09_01_04_39_CDT.FLOW.hdf5"
    # filename = "packet-0050015-2024_07_08_13_37_49_CDT.FLOW.hdf5"
    # filename = 'packet-0050018-2024_07_11_19_59_52_CDT.FLOW.hdf5'
    # filename = 'packet-0050016-2024_07_08_13_40_57_CDT.FLOW.hdf5'
    # filename = 'packet-0050017-2024_07_08_13_43_25_CDT.FLOW.hdf5'

    output_dir = filename.split('.')[0]
    os.makedirs(os.path.join('../output', output_dir), exist_ok=True)
    output_dir = '/'.join(['../output', output_dir])

    flow_data, NEvents = Read_Flow(root_path=root_path, filename=filename)

    # evtID = 481
    # for evtID in range(420, 450):
    # for evtID in range(1265, 1300):
    for evtID in range(0, NEvents):
        evt_object = Event(flowdata=flow_data, eventID=evtID, eps=3, min_samples=10)
        if evt_object is None:
            continue
        else:
            if evt_object.Ntracks < 2:
                continue
        import matplotlib.pyplot as plt

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot XZ view
        for i in range(evt_object.Ntracks):
            track0 = evt_object.eventData[evt_object.eventData['trackID']==i]
            ax1.scatter(track0['x'], track0['z'], s=0.5, label='track{}'.format(i))

            # Plot YZ view
            ax2.scatter(track0['y'], track0['z'], s=0.5, label='track{}'.format(i))

            # Plot XY view
            ax3.scatter(track0['x'], track0['y'], s=0.5, label='track{}'.format(i))
        
        ax1.set_title('XZ View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')

        ax2.set_title('YZ View')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')

        ax3.set_title('XY View')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # Adjust layout and save
        plt.legend()
        plt.tight_layout()
        os.makedirs('../output', exist_ok=True)
        plt.savefig('{}/Event{}.png'.format(output_dir, evtID))
        plt.close()
        # evt_object.data2EventStructure()