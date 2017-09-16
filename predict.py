from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
model = load_model('model.h5')

img = cv2.imread('img.jpg')
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.reshape(img, [1, 224, 224, 3])

img = (img - np.mean(img))/np.std(img)


results = model.predict(img)
print(results)

print(np.argmax(results))
