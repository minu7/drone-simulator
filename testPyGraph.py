from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import threading
import time

app = QtGui.QApplication([])

## Create window with GraphicsView widget
w = pg.GraphicsView()
w.show()
w.resize(800,800)
w.setWindowTitle('test')

view = pg.ViewBox()
w.setCentralItem(view)

## lock the aspect ratio
view.setAspectLocked(True)

## Create image item
data = np.full((200,200, 3), 200)
img = pg.ImageItem(data)
view.addItem(img)


def thread_function():
    time.sleep(2)
    print("ciao")
    data[0:40, 40:200, 0:3] = 0
    img.updateImage()

x = threading.Thread(target=thread_function)
x.start()
## Start Qt event loop unless running in interactive mode or using pyside.
QtGui.QApplication.instance().exec_()
