#to train and test the model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle
from scipy import signal
import itertools

def uniform(img,kernel_size = 3):
  kernel = (1/kernel_size**2)*np.ones([kernel_size,kernel_size])
  # 'boundary' tells the function to implement the convolution in a symmetrical 
  # way. 'same' produces an image with the same size as the input image
  filtered = signal.convolve2d(img, kernel, boundary='symm', mode='same')
  return filtered

def mid_filter(img, size):
  m, n = img.shape
  filtered = np.zeros([m,n])
  # Let's pad the image with extra zeros
  gray_image_v2 = np.zeros([m+2, n+2])
  gray_image_v2[1:m+1,1:n+1] = img
  step_size = int((size-1)/2)
  for i in range(m):
    for j in range(n):
      filtered[i,j] = 1/2.0*(np.max(gray_image_v2[i-step_size +1 :i+step_size +1, j-step_size +1 : j+step_size +1])+ np.min(gray_image_v2[i-step_size +1 :i+step_size +1,
                                                  j-step_size +1 : j+step_size +1]))
  return filtered

def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes, rotation=45)
    plt.yticks(tick_marks,classes)

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix without normalization")
    
    print(cm)
    thresh=cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",
                              color="white" if cm[i,j]>thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

os.chdir('data/train')
dir=os.listdir(os.curdir)
print(dir)
x_train=[]
y_train=[]
x_faces=[]
for i in dir:
    for j in os.listdir(f'{i}'):
        print(f'{i}/{j}')
        img=cv2.imread(f'{i}/{j}')
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray, 1.1,4)
        if len(face):
            x,y,w,h=face[0]
            gray2=gray[y:y+h,x:x+w]
            gray3=cv2.resize(gray2,(128,128))
            gray4=uniform(gray3)
            gray5=mid_filter(gray4,3)
            x_faces.append(gray5)
            pca=PCA(n_components=128)
            pca_img=pca.fit_transform(gray5)
            x_train.append(pca_img.reshape((-1)))
            y_train.append(i)
os.chdir('../..')
# plot_image_grid(x_faces[:8])
x_train=x_train+x_train
y_train=y_train+y_train
x_train=np.array(x_train)

classifier=SVC()
clf=CalibratedClassifierCV(classifier).fit(x_train,y_train)
filename='trained_model.sav'

pickle.dump(clf,open(filename,'wb'))

clf=pickle.load(open(filename,'rb'))
os.chdir('data/test')

x_test=[]
y_test=[]
dir=os.listdir(os.curdir)
for i in dir:
    for j in os.listdir(f'{i}'):
        print(f'{i}/{j}')
        img=cv2.imread(f'{i}/{j}')
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray, 1.1,4)
        if len(face):
            x,y,w,h=face[0]
            gray2=gray[y:y+h,x:x+w]
            gray3=cv2.resize(gray2,(128,128))
            gray4=uniform(gray3)
            gray5=mid_filter(gray4,3)
            pca=PCA(n_components=128)
            pca_img=pca.fit_transform(gray5)
            x_test.append(pca_img.reshape((-1)))
            y_test.append(i)
x_test=np.array(x_test)
predictions=clf.predict(x_test)

print(classification_report(y_test, predictions))

cm=confusion_matrix(y_true=y_test,y_pred=predictions)
cm_labels=['Alexander','Diego','Paula','Rafael','Roberto','Rodrigo']
plot_confusion_matrix(cm=cm,classes=cm_labels,title='Confusion Matrix')

