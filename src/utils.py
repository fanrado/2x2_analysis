'''
Utilities for the analysis.
'''
import h5flow

def Read_Flow(root_path: str, filename: str):
  h5flow_data = h5flow.data.DataManager('/'.join([root_path, filename]), 'r')
  return h5flow_data['charge']
