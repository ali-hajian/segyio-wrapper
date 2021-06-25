"""

Created by Ali Hajian

"""

import segyio
from time import time
import numpy as np

# class to create SGY reader objects
class SGY_Reader():
    # need to pass the file name for initialization
    def __init__(self, file_name):
        self.file_name = file_name
        with segyio.open(self.file_name) as src:
            self.ilines = list(src.ilines)
            self.xlines = list(src.xlines)
            self.trace_len = src.trace[0].size
            self.depths = range(self.trace_len)

    # function to select iline slices and load them from the sgy file into a numpy array
    def get_iline_slices(self, ilines):
        t = time()
        # make sure ilines are iterables
        try:
            n_ilines = len(ilines)
        except:
            ilines = [ilines]
            n_ilines = len(ilines)
        
        # make sure all ilines are valid
        for i in range(n_ilines):
            if ilines[i] not in self.ilines:
                print("iline %i is not valid" %ilines[i])
                return None
        
        # open and read the iline slices from the SGY file
        with segyio.open(self.file_name) as src:
            if n_ilines > 1:
                slice_list = np.empty(
                    (n_ilines, len(self.xlines), self.trace_len),
                    dtype=np.float32
                )
            for i in range(n_ilines):
                if n_ilines > 1:
                    slice_list[i] = src.iline[ilines[i]]
                else:
                    slice_list = src.iline[ilines[i]]
                    
        print('--> %i iline slices from %s loaded in %.2f sec' %(n_ilines, 
                                                                 self.file_name, 
                                                                 time()-t))

        return slice_list
    
    # function to select xline slices and load them from the sgy file into a numpy array
    def get_xline_slices(self, xlines):
        t = time()
        # make sure xlines are iterables
        try:
            n_xlines = len(xlines)
        except:
            xlines = [xlines]
            n_xlines = len(xlines)
        
        # make sure all xlines are valid
        for i in range(n_xlines):
            if xlines[i] not in self.xlines:
                print("xline %i is not valid" %xlines[i])
                return None
        
        # open and read the xline slices from the SGY file
        with segyio.open(self.file_name) as src:
            if n_xlines > 1:
                slice_list = np.empty(
                    (n_xlines, len(self.ilines), self.trace_len),
                    dtype=np.float32
                )
            for i in range(n_xlines):
                if n_xlines > 1:
                    slice_list[i] = src.xline[xlines[i]]
                else:
                    slice_list = src.xline[xlines[i]]
                    
        print('--> %i xline slices from %s loaded in %.2f sec' %(n_xlines, 
                                                                 self.file_name, 
                                                                 time()-t))

        return slice_list
    
    # function to select depth slices and load them from the sgy file into a numpy array
    def get_depth_slices(self, depths):
        t = time()
        # make sure depths are iterables
        try:
            n_depths = len(depths)
        except:
            depths = [depths]
            n_depths = len(depths)
        
        # make sure all depths are valid
        for i in range(n_depths):
            if depths[i] not in self.depths:
                print("depth %i is not valid" %depths[i])
                return None
        
        # open and read the depth slices from the SGY file
        with segyio.open(self.file_name) as src:
            if n_depths > 1:
                slice_list = np.empty(
                    (n_depths, len(self.ilines), len(self.xlines)),
                    dtype=np.float32
                )
            for i in range(n_depths):
                if n_depths > 1:
                    slice_list[i] = src.depth_slice[depths[i]]
                else:
                    slice_list = src.depth_slice[depths[i]]
                    
        print('--> %i depth slices from %s loaded in %.2f sec' %(n_depths, 
                                                                 self.file_name, 
                                                                 time()-t))

        return slice_list
            
    # function to select traces and load them from the sgy file into a numpy array
    def get_traces(self, ilines, xlines):
        t = time()
        # make sure ilines and xlines are iterables
        try:
            n_ilines = len(ilines)
        except:
            ilines = [ilines]
            n_ilines = len(ilines)
        try:
            n_xlines = len(xlines)
        except:
            xlines = [xlines]
            n_xlines = len(xlines)
            
        # make sure ilines and xlines have the same size
        if n_ilines == n_xlines:
            n_traces = n_ilines
        else:
            print("Number of ilines and xlines need to be the same.")
            return None
        
        # make sure all ilines and xlines are valid
        for i in range(n_traces):
            if ilines[i] not in self.ilines:
                print("iline %i is not valid" %ilines[i])
                return None
            if xlines[i] not in self.xlines:
                print("xline %i is not valid" %xlines[i])
                return None
        
        # open and read the traces from the SGY file
        with segyio.open(self.file_name) as src:
            if n_traces > 1:
                trace_list = np.empty((n_traces, self.trace_len), dtype=np.float32)
            for i in range(n_traces):
                if n_traces > 1:
                    trace_list[i] = src.gather[ilines[i], xlines[i]]
                else:
                    trace_list = src.gather[ilines[i], xlines[i]][0]
                    
        print('--> %i traces from %s loaded in %.2f sec' %(n_traces, self.file_name, time()-t))

        return trace_list
    
    # function to select slices or traces and load them from the sgy file into a numpy array
    def get_data(self, ilines=None, xlines=None, depths=None):
        if depths is not None:
            if ilines or xlines:
                print("Too many arguments. Can not use depths " \
                       "and ilines xlines at the same time.")
                return None
            else:
                # get depth slices
                data = self.get_depth_slices(depths)
        elif ilines is not None:
            if xlines is not None:
                # get the traces at the intersection of inlines and xlines
                data = self.get_traces(ilines, xlines)
            else:
                # get iline slices
                data = self.get_iline_slices(ilines)
        elif xlines is not None:
            # get xline slices
            data = self.get_xline_slices(xlines)
        else:
            print("Need to pass at least one of ilines or xlines or depths.")
            return None
        
        return data