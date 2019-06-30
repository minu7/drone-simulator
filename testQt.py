import os
import sys
import numpy as np
from PyQt4.QtGui import *
import threading
import time

def thread_function():
    time.sleep(5)
    print("ciao")
    im[0:40, 40:200] = rgb(0, 0, 255)
    w.update()

def rgb(r, g, b):
    return (qRgb(r, g, b) & 0xffffff) - 0x1000000
# Create window
app = QApplication(sys.argv)
w = QWidget()
w.setWindowTitle("Test ")

# Create widget
label = QLabel(w)
im = np.full((200, 200), qRgb(0, 0, 0) + qRgb(0, 0, 0))
# qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
qimage = QImage(im.data, im.shape[0], im.shape[1], QImage.Format_RGB32)

pixmap = QPixmap.fromImage(qimage)
label.setPixmap(pixmap)
w.resize(pixmap.width(),pixmap.height())

# Draw window
w.show()
x = threading.Thread(target=thread_function)
x.start()
app.exec_()
