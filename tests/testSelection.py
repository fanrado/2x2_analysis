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
    filename = "packet-0050015-2024_07_08_13_37_49_CDT.FLOW.hdf5"
    flow_data = Read_Flow(root_path=root_path, filename=filename)
    evtID = 481
    # for evtID in range(420, 450):
    for evtID in range(481, 482):
        # event_data = get_Eventdata(flow_data, evtID)

        evt_object = Event(flowdata=flow_data, eventID=evtID, eps=3.3, min_samples=5)
        # evt_object.data2EventStructure()