import sys 
from PyQt6 import QtCore, QtGui, QtWidgets
from ui_mainwindow import Ui_Form
import cv2
import numpy as np
import os.path


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.ui = Ui_Form()
		self.ui.setupUi(self)
		self.imgPath = '.'
		self.img01 = cv2.imread('01.jpg', 0) # Default img01
		self.img02 = cv2.imread('02.jpg', 0) # Default img02
		self.tempImg = '01.jpg'
		self.computedImg = self.img01
		self.ui.img_btn01.clicked.connect(self.updateImg1)
		self.ui.img_btn02.clicked.connect(self.updateImg2)
		self.ui.inputFileButton.clicked.connect(self.onInputFileButtonClicked)
		self.ui.computeMatch.clicked.connect(self.compute_orb)
        
	def onInputFileButtonClicked(self):
		filename, filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', self.imgPath, 'All Files (*.*)')
		if filename:
			self.ui.inputFileLineEdit.setText(filename)
			self.tempImg = filename
			self.imgPath = os.path.dirname(filename)
	        
	def updateImg1(self):
		self.img01 = cv2.imread(self.tempImg, 0)

	def updateImg2(self):
		self.img02 = cv2.imread(self.tempImg, 0)

	def compute_orb(self):    
		img1 = self.img01
		img2 = self.img02
		orb = cv2.ORB_create()    	
		keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
		keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(descriptors1, descriptors2)
		matches = sorted(matches, key = lambda x:x.distance)
		img3 = self.draw_matches(img1, keypoints1, img2, keypoints2, matches[:30])
		cv2.imwrite('computed.png', img3)
		self.setImage(img3)
    
	def draw_matches(self, img1, keypoints1, img2, keypoints2, matches):
		rows1, cols1 = img1.shape[:2]
		rows2, cols2 = img2.shape[:2]
		output_img = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
		output_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
		output_img[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2,img2])
		for match in matches:
			img1_idx = match.queryIdx
			img2_idx = match.trainIdx
			(x1, y1) = keypoints1[img1_idx].pt
			(x2, y2) = keypoints2[img2_idx].pt
			radius = 4
			colour = (0,255,0)
			thickness = 1
			cv2.circle(output_img, (int(x1),int(y1)), radius, colour, thickness)
			cv2.circle(output_img, (int(x2)+cols1,int(y2)), radius, colour, thickness)
			cv2.line(output_img, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), colour, thickness)
		return output_img
    
	def setImage(self, img):
		img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format.Format_RGB888).rgbSwapped()
		img_pix = QtGui.QPixmap.fromImage(img)
		if img_pix.height() < img_pix.width():
			self.ui.img_frame.setPixmap(img_pix.scaledToHeight(self.ui.img_frame.width(), QtCore.Qt.TransformationMode.SmoothTransformation))
		elif img_pix.height() > img_pix.width():
			self.ui.img_frame.setPixmap(img_pix.scaledToWidth(self.ui.img_frame.width(), QtCore.Qt.TransformationMode.SmoothTransformation))
		else:
			self.ui.img_frame.setPixmap(img_pix.scaled(self.ui.img_frame.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))	
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
