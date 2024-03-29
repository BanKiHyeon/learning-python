import sys
from PyQt5 import QtWidgets
from PyQt5 import uic


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("ui.ui", self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    w.show()

    sys.exit(app.exec())
