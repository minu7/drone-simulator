import os
import sys
import numpy as np
from PyQt4.QtGui import *
import threading
import time

def thread_function():
    time.sleep(5)
    print("ciao")
    im[True] = rgb(0, 255, 0)
    w.update()

def rgb(r, g, b):
    return (qRgb(r, g, b) & 0xffffff) - 0x1000000
# Create window
app = QApplication(sys.argv)
w = QWidget()
w.setWindowTitle("Test ")

# Create widget
label = QLabel(w)
im = np.zeros((500, 500))
# qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
qimage = QImage(im.data, im.shape[0], im.shape[1], QImage.Format_RGB32)

pixmap = QPixmap.fromImage(qimage)
label.setPixmap(pixmap)
w.resize(pixmap.width(),pixmap.height())

im[True] = rgb(255, 0, 0)
# Draw window
w.show()
x = threading.Thread(target=thread_function)
x.start()
app.exec_()