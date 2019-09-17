import cv2
import numpy as np
import h5py
from keras.models import load_model,model_from_json


model = model_from_json(open('E:\\image_python\\data\\Minst\\my_model.json').read())
model.load_weights('E:\\image_python\\data\\Minst\\model.h5')
#model = load_model('D:\\image_python\\data\\Minst\\model.h5')



testData = np.empty((6, 1, 28, 28), dtype="float32")

imgfile = r'E:\image_python\data\Minst\test\test\1.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image1")
cv2.imshow("Image1", imgData)
testData[0, :, :, :] = arr

imgfile = r'E:\image_python\data\Minst\test\test\8.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image2")
cv2.imshow("Image2", imgData)
testData[1, :, :, :] = arr

imgfile = r'E:\image_python\data\Minst\test\test\15.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image3")
cv2.imshow("Image3", imgData)
testData[2, :, :, :] = arr

imgfile = r'E:\image_python\data\Minst\test\test\22.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image4")
cv2.imshow("Image4", imgData)
testData[3, :, :, :] = arr

imgfile = r'E:\image_python\data\Minst\test\test\28.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image5")
cv2.imshow("Image5", imgData)
testData[4, :, :, :] = arr

imgfile = r'E:\image_python\data\Minst\test\test\40.jpg'
imgData = cv2.imread(imgfile, 0)  # 数据
imgData = cv2.resize(imgData,(28,28))
arr = np.asarray(imgData, dtype="float32")
cv2.namedWindow("Image6")
cv2.imshow("Image6", imgData)
testData[5, :, :, :] = arr

testData = testData.reshape(testData.shape[0], 28, 28, 1)

print(model.predict_classes(testData, batch_size=1, verbose=0))

cv2.waitKey(0)
cv2.destroyAllWindows()











