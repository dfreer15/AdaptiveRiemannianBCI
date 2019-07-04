from pylsl import StreamInlet, resolve_stream
from numpy import *
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

### START QtApp #####
app = QtGui.QApplication([])            # you MUST do this once (initialize things)
####################

win = pg.GraphicsWindow(title="Signal from serial port") # creates a window
p = win.addPlot(title="Realtime plot")  # creates empty space for the plot in the window
curve = p.plot()                        # create an empty "plot" (a curve to plot)

windowWidth = 500                       # width of the window displaying the curve
Xm = linspace(0,0,windowWidth)          # create array that will contain the relevant time series     
ptr = -windowWidth                      # set first x position

# Realtime data plot. Each time this function is called, the data display is updated
def update(x):
    global curve, ptr, Xm    
    Xm[:-1] = Xm[1:]                      # shift data in the temporal mean 1 sample left
    value = x                # read line (single value) from the serial port
    Xm[-1] = float(value)                 # vector containing the instantaneous values      
    ptr += 1                              # update x position for displaying the curve
    curve.setData(Xm)                     # set the curve with this data
    curve.setPos(ptr,0)                   # set x position in the graph to 0
    QtGui.QApplication.processEvents()    # you MUST process the plot now
    return

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'signal')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
try:
    while True:
        sample, timestamp = inlet.pull_sample()
        if sample:
            update(sample[4])
        # print(timestamp, sample)
except KeyboardInterrupt:
    pass

### END QtApp ####
pg.QtGui.QApplication.exec_() # you MUST put this at the end