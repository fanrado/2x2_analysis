"""
Author: radofana@gmail.com
Date: 2023-10-06
Time: 14:30

This script is for file reading tests in the WireCell-NDLAr project.
"""

import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the Read_Flow function from utils.py
from utils import Read_Flow
from utils import list_hdf5_contents
from utils import get_Eventdata
from utils import drawEvent
from utils import Eventdata2Array

if __name__=='__main__':
    root_path = '../../flowdata'
    filename = "packet-0050015-2024_07_08_13_37_49_CDT.FLOW.hdf5"
    
    # Use the Read_Flow function to read the charge data
    flow_data = Read_Flow(root_path=root_path, filename=filename)
    list_hdf5_contents('/'.join([root_path, filename]))
    # evtID = 481
    # calib_data = get_Eventdata(flow_data, evtID)
    # arr_calib = Eventdata2Array(evtdata=calib_data, evtid=evtID)
    # drawEvent(arr_calib)
    # print(arr_calib[['x', 'y', 'z']])