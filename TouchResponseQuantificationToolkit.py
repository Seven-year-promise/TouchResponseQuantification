from PyQt5.QtWidgets import (QApplication)
from PyQt5.QtWidgets import QMainWindow


from ToolkitMainWindow import Ui_mainWindow
"""
QDialog {
    /*background: #fafafa;*/
    background: blue;
    border:1px solid #cccccc;
    border-radius:5px;
}
QWidget#main {
    background:#fafafa;
    border:1px solid #cccccc;
    border-bottom-left-radius:5px;
    border-bottom-right-radius:5px;
}
QFrame#header {
    background:#eeeeee;
    border:1px solid #cccccc;
    border-top-left-radius:5px;
    border-top-right-radius:5px;
    text-align:left;
    padding-top: 5px;
    padding-bottom:5px;
    padding-left: 15px;
    padding-right: 15px;
}
QLabel#title {
    font-size:16px;
    color:#333333;
    font-weight: 500;
}
QLabel#close_btn {
    font-size: 16px;
    color: #ee4040;
}
QWidget#main Qlabel{
    font-size:14px;
    color:#000000;
}
"""



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()

    ui = Ui_mainWindow()

    #ui.setupUi(mainWindow)
    #mainWindow.show()
    sys.exit(app.exec_())