
'''
    Author : radofana@gmail.com
    Date: December 25, 2024
'''
import os, sys, numpy as np
import numpy.lib.recfunctions as rfn
from utils import get_Eventdata, SpatialClustering

class Event():
    def __init__(self, flowdata, eventID, eps, min_samples):
        block_data, self.NEvents = get_Eventdata(h5flow_data=flowdata, evid=eventID)
        self.data = SpatialClustering(data=block_data, eps=eps, min_samples=min_samples)
        self.eventID = eventID
        self.trk_event_dtype = self.__eventDtype() # dtype ==> [('eventID', 'int32'), track.dtype]
        self.eventData = self.data2EventStructure() 

    def __eventDtype(self):
        trackDtype = self.data[0].dtype
        trk_dtype_list = trackDtype.descr
        trk_dtype_list.append(('eventID', 'int32'))
        trk_dtype_list.append(('trackID', 'int32'))
        return np.dtype(trk_dtype_list)
    
    def data2EventStructure(self):
        eventData = np.array([], dtype=self.trk_event_dtype)
        for trackID, track in enumerate(self.data):
            array_evtID = np.array([self.eventID for _ in range(len(track['x']))])
            array_trkID = np.array([trackID for _ in range(len(track['x']))])
            track = rfn.append_fields(track, 'eventID', array_evtID, usemask=False)
            track = rfn.append_fields(track, 'trackID', array_trkID, usemask=False)
            if len(eventData)==0:
                eventData = track
            else:
                eventData = np.concatenate((eventData, track))
        return eventData
    
    def getTrack(self, trackID):
        return self.eventData[self.eventData['trackID']==trackID]
    
    def getEvent(self):
        return self.eventID, self.eventData
    
##----------- Vertex Finding algorithm -------------##
class VertexFinding():
    def __init__(self, eventData):
        self.eventData = eventData

    def cutsOnTracks(self, cuts_dict: dict):
        pass

    def PrimaryVertexFinding(self):
        pass
