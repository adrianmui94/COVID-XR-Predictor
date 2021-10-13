#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyQt5 import QtCore, QtGui, QtWidgets
from pynput import mouse
import pyautogui
import os
import tensorflow.keras as k
from tensorflow.keras.models import Model
import cv2
import numpy as np
import shutil

class MyModel():
    """
    Ideally we'll have our preprocessing steps in our Model class instead of stuffing it inside our 
    """
    def __init__(self, model_name):
        self.estimator = k.models.load_model("./models/" + model_name)
        self.INPUT_DIMS = self.estimator.layers[0].input_shape
        self.PREVIEW_DIMS = (299, 299)
        
    
    def preview_candidate(self, path):
        """
        Resize the candidate into a 299x299 square.
        """
        img = cv2.imread(path)
        resize = cv2.resize(img, self.PREVIEW_DIMS)
        cv2.imwrite("last_candidate.png", resize)
    
    def transform_candidate(self):
        """
        Preprocess a single image into a readable array
        """
        img = cv2.imread("last_screenshot.png")
        resize = cv2.resize(img, (self.INPUT_DIMS[1], self.INPUT_DIMS[2]))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        arr = gray/255
        return np.expand_dims(arr, 2)
    
    def predict(self):
        inputs = np.expand_dims(self.transform_candidate(), 0)
        preds = self.estimator(inputs)
        return preds
        
class PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):   
        super().__init__()
        # Initial Calibration
        self.num_clicks = 0
        self.MAX_CLICKS = 2
        
        if "config.txt" in os.listdir():
            with open("config.txt", "r") as f:
                self.X1, self.Y1, self.X2, self.Y2 = tuple(map(int, f.read().splitlines()))
        else:
            self.X1 = 0
            self.Y1 = 0
            self.X2 = 1
            self.Y2 = 1
        
        # For selection of model
        self.model_list = [model for model in os.listdir("./models/") if "model" in model]
        if self.model_list:
            self.model_name = self.model_list[0]
        else:
            self.model_name = None
        self.model = MyModel(self.model_name)
        
        # For saving/loading/deleting model
        self.saved_studies = [file for file in os.listdir("./saved_studies/")]
        self.prediction = ""
        self.probability = 0.0
        
        self.setupUi()
        
        # Connect to calibration
        self.SelectSnapArea.clicked.connect(self.select_snap_area)
        self.SnapArea.clicked.connect(self.snap_candidate)
        self.SelectModel.currentIndexChanged.connect(self.model_selection_change)
        self.Evaluate.clicked.connect(self.evaluate_func)
        self.SaveStudy.clicked.connect(self.save_study)
        self.LoadStudy.currentIndexChanged.connect(self.load_study)
        
    def setupUi(self):
        
        # Main Window
        self.setObjectName("MainWindow")
        self.resize(627, 700)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        # Candidate Preview Window
        self.Candidate = QtWidgets.QLabel(self.centralwidget)
        self.Candidate.setGeometry(QtCore.QRect(20, 20, 299, 299))
        self.Candidate.setFrameShape(QtWidgets.QFrame.Panel)
        self.Candidate.setText("")
        if "last_candidate.png" in os.listdir():
            self.Candidate.setPixmap(QtGui.QPixmap("last_candidate.png"))
        else:
            self.Candidate.setPixmap(QtGui.QPixmap("placeholder.png"))
        self.Candidate.setObjectName("Candidate")
        
        # Candidate Heatmap Window
        self.CandidateHeatmap = QtWidgets.QLabel(self.centralwidget)
        self.CandidateHeatmap.setGeometry(QtCore.QRect(20, 350, 299, 299))
        self.CandidateHeatmap.setFrameShape(QtWidgets.QFrame.Panel)
        self.CandidateHeatmap.setText("")
        self.CandidateHeatmap.setPixmap(QtGui.QPixmap("placeholder.png"))
        self.CandidateHeatmap.setObjectName("CandidateHeatmap")
        
        # Select Snap Area Button
        self.SelectSnapArea = QtWidgets.QPushButton(self.centralwidget)
        self.SelectSnapArea.setGeometry(QtCore.QRect(350, 20, 240, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SelectSnapArea.setFont(font)
        self.SelectSnapArea.setObjectName("SelectSnapArea")
        
        # Select Model ComboBox
        self.SelectModel = QtWidgets.QComboBox(self.centralwidget)
        self.SelectModel.setGeometry(QtCore.QRect(350, 140, 240, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SelectModel.setFont(font)
        self.SelectModel.setObjectName("SelectModel")
        self.SelectModel.addItems(self.model_list)
        
        
        # Snap Area Button
        self.SnapArea = QtWidgets.QPushButton(self.centralwidget)
        self.SnapArea.setGeometry(QtCore.QRect(350, 220, 240, 99))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SnapArea.setFont(font)
        self.SnapArea.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.SnapArea.setObjectName("SnapArea")
        
        # Save Study button
        self.SaveStudy = QtWidgets.QPushButton(self.centralwidget)
        self.SaveStudy.setGeometry(QtCore.QRect(350, 410, 240, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SaveStudy.setFont(font)
        self.SaveStudy.setObjectName("SaveStudy")
        
        # Display Model Name Label
        self.SaveStudyConfirmation = QtWidgets.QLabel(self.centralwidget)
        self.SaveStudyConfirmation.setGeometry(QtCore.QRect(350, 450, 240, 20))
        self.SaveStudyConfirmation.setObjectName("SaveStudyConfirmation")
        
        # Load Study Button
        self.LoadStudy = QtWidgets.QComboBox(self.centralwidget)
        self.LoadStudy.setGeometry(QtCore.QRect(350, 490, 240, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.LoadStudy.setFont(font)
        self.LoadStudy.setObjectName("LoadStudy")
        self.LoadStudy.addItems(self.saved_studies)
        
        # Evaluate Button
        self.SelectSnapArea.setObjectName("SelectSnapArea")
        self.Evaluate = QtWidgets.QPushButton(self.centralwidget)
        self.Evaluate.setGeometry(QtCore.QRect(350, 350, 240, 50))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Evaluate.setFont(font)
        self.Evaluate.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Evaluate.setObjectName("Evaluate")
        
        # Display Model Name Label
        self.DisplayModelName = QtWidgets.QLabel(self.centralwidget)
        self.DisplayModelName.setGeometry(QtCore.QRect(350, 190, 241, 16))
        self.DisplayModelName.setObjectName("DisplayModelName")
        
        # Probability Output Label
        self.ProbabilityOutput = QtWidgets.QLabel(self.centralwidget)
        self.ProbabilityOutput.setGeometry(QtCore.QRect(350, 540, 240, 50))
        self.ProbabilityOutput.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ProbabilityOutput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.ProbabilityOutput.setObjectName("ProbabilityOutput")
        
        # Config Output Label
        self.ConfigOutput = QtWidgets.QLabel(self.centralwidget)
        self.ConfigOutput.setGeometry(QtCore.QRect(350, 80, 241, 41))
        self.ConfigOutput.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ConfigOutput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.ConfigOutput.setObjectName("ConfigOutput")
        self.setCentralWidget(self.centralwidget)
        
        # MenuBar
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 627, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(self)
        self.actionAbout.setObjectName("actionAbout")
        self.menu.addAction(self.actionAbout)
        self.menubar.addAction(self.menu.menuAction())
        
        self.retranslateUi(self)
        
        QtCore.QMetaObject.connectSlotsByName(self)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SelectSnapArea.setText(_translate("MainWindow", "Configure Snap Area"))
        self.SnapArea.setText(_translate("MainWindow", "Snap Area"))
        self.Evaluate.setText(_translate("MainWindow", "Evaluate"))
        self.SaveStudy.setText(_translate("MainWindow", "Save Study"))
        self.SaveStudyConfirmation.setText(_translate("MainWindow", ""))
        self.DisplayModelName.setText(_translate("MainWindow", self.adjust_model_label()))
        self.ProbabilityOutput.setText(_translate("MainWindow", self.adjust_probability_label()))
        self.ConfigOutput.setText(_translate("MainWindow", self.adjust_config_label()))
        self.menu.setTitle(_translate("MainWindow", "..."))
        self.actionAbout.setText(_translate("MainWindow", "About"))
    
    """
    ************************************************************************************
    THESE ARE FOR ADJUSTING DIALOGUE BOXES
    ************************************************************************************
    """
    
    def adjust_config_label(self, override=None):
        BASE_TEXT = "X1,Y1: ({},{})\nX2,Y2: ({},{})"
        if override:
            return override
        return BASE_TEXT.format(self.X1, self.Y1, self.X2, self.Y2)
    
    def adjust_probability_label(self, override=None):
        BASE_TEXT = "Prediction: {}\nProbability: {}"
        if override:
            return override
        return BASE_TEXT.format(self.prediction, self.probability)
    
    def adjust_model_label(self, override=None):
        if override:
            return override
        BASE_TEXT = "Current Model: {}"
        return BASE_TEXT.format(self.model_name)    
    def save_study_confirmation(self, filename, override=None):
        if override:
            return overrride
        BASE_TEXT = "Saved study: "
        return BASE_TEXT + filename
    
    """
    ************************************************************************************
    THESE ARE FOR CONFIGURING THE READER
    ************************************************************************************
    """
    def get_coords(self):
        '''
        Use to get coordinates upon mouse click and mouse release.
        For use in "Select Snap Area"
        '''
        def on_click(x, y, button, pressed, self=self):
            if self.num_clicks == 0:
                self.X1 = x
                self.Y1 = y
                self.num_clicks += 1
            else:
                self.X2 = x
                self.Y2 = y
                self.num_clicks = 0
                return False
        
        with mouse.Listener(on_click=on_click) as listener:
            try:
                listener.join()
            except:
                print(self.X1, self.Y1, self.X2, self.Y2)
                listener.stop
                
    def select_snap_area(self):
        self.ConfigOutput.setText(self.adjust_config_label(override="****CALIBRATING****"))
        self.get_coords()
        self.ConfigOutput.setText(self.adjust_config_label())
        with open("config.txt", "w") as f:
            f.write(f"{self.X1}\n{self.Y1}\n{self.X2}\n{self.Y2}")
        
        
    """
    ************************************************************************************
    THESE ARE FOR CHANGING THE MODEL
    ************************************************************************************
    """
    
    def model_selection_change(self):
        self.model_name = self.SelectModel.currentText()
        self.model = MyModel(self.model_name)
        self.DisplayModelName.setText(self.adjust_model_label())
        
    """
    ************************************************************************************
    THESE ARE FOR SNAPPING SHOTS
    ************************************************************************************
    """
    
    def snap_candidate(self):
        width = abs(self.X2 - self.X1)
        height = abs(self.Y2 - self.Y1)
        screenshot = pyautogui.screenshot(region=(self.X1, self.Y1, width, height))
        screenshot.save("last_screenshot.png")
        self.model.preview_candidate("last_screenshot.png")
        self.Candidate.setPixmap(QtGui.QPixmap("last_candidate.png"))
        
    """
    ************************************************************************************
    THESE ARE FOR EVALUATING
    ************************************************************************************
    """
    
    def evaluate_func(self):
        mapping = {0: "COVID NEGATIVE", 1: "COVID POSITIVE"}
        model_output = self.model.predict()[0]
        self.prediction = mapping[np.argmax(model_output)]
        self.probability = np.max(model_output)
        self.ProbabilityOutput.setText(self.adjust_probability_label())
    
    def gradcam(self):
        pass
    """
    ************************************************************************************
    THESE ARE FOR SAVING, LOADING, AND DELETING STUDIES
    ************************************************************************************
    """
    
    def save_study(self):
        custom_studies = [file for file in self.saved_studies if "study_" in file]
        if custom_studies:
            last_file = self.saved_studies[-1]
            last_int = last_file.split('_')[-1].replace('.png', '').strip('0')
            last_int = int(last_int) + 1
        else:
            last_int = 1
        filename = f"study_{last_int:05}.png"
        shutil.copy("last_screenshot.png", "./saved_studies/" + filename)
        self.saved_studies = os.listdir("./saved_studies/")
        self.LoadStudy.addItem(filename)
        self.SaveStudyConfirmation.setText(self.save_study_confirmation(filename))
        
        
    def load_study(self):
        filename = self.LoadStudy.currentText()
        shutil.copy(f"./saved_studies/{filename}", "last_screenshot.png")
        self.model.preview_candidate("last_screenshot.png")
        self.Candidate.setPixmap(QtGui.QPixmap("last_candidate.png"))
        self.evaluate_func()
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = PredictionApp()
    ui.show()
    sys.exit(app.exec_())


# In[ ]:




