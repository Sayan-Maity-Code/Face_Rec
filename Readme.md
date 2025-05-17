# Face Recognition System Documentation

This document describes the implementation of a face recognition system in Jupyter Notebook using the `keras-facenet` wrapper for FaceNet embeddings and OpenCV for face detection.

---

## Prerequisites

```bash
pip install opencv-python mtcnn numpy scikit-learn keras-facenet
```

* **Python 3.7+**
* **Jupyter Notebook** or **JupyterLab**

---

## 1. Imports and Setup

```python
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
```

* **os** : File and directory operations.
* **cv2** : Image I/O, webcam capture, color conversion, resizing.
* **numpy** : Numerical arrays for pixel data and embeddings.
* **mtcnn.MTCNN** : Face detection network.
* **keras-facenet.FaceNet** : Auto-download & load of FaceNet model.
* **Normalizer** : L2-normalization of 512-d embeddings.
* **LabelEncoder** : Encode person names as integer labels.
* **SVC / KNeighborsClassifier** : Classifiers for recognition.
* **pickle** : Save/load model and encoders.

---

## 2. Initialize FaceNet Embedder

```python
embedder = FaceNet()
facenet_model = embedder.model
print(f"FaceNet input shape: {facenet_model.input_shape}")
```

* Automatically downloads `facenet_keras.h5` and loads the Keras model.
* Dynamic spatial dimensions `(None, None, None, 3)` — we will resize to `(160,160,3)`.

---

## 3. Face Detection & Extraction

```python
detector = MTCNN()

def extract_face(frame, size=(160,160)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    if not results:
        return None
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = rgb[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    return face
```

* **MTCNN.detect_faces** yields bounding boxes.
* Crop, handle negative coords, and resize to 160×160.

---

## 4. Embedding & Dataset Utilities

```python
def get_embedding(face_pixels):
    emb = embedder.embeddings([face_pixels])
    return emb[0]


def load_dataset_embeddings(path='dataset/'):
    X, y = [], []
    for person in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, person)):
            img = cv2.imread(os.path.join(path, person, img_name))
            face = extract_face(img)
            if face is None: continue
            X.append(get_embedding(face))
            y.append(person)
    return np.array(X), np.array(y)
```

* Walk `dataset/<person>/` folders.
* Extract face, compute 512‑dim embedding, and collect labels.

---

## 5. Train & Persist Classifier

```python
def train_classifier(emb, names):
    if len(names)==0:
        raise ValueError("No data to train")
    in_enc = Normalizer('l2')
    X_norm = in_enc.transform(emb)
    out_enc = LabelEncoder()
    y = out_enc.fit_transform(names)
    if len(out_enc.classes_)==1:
        model = KNeighborsClassifier(n_neighbors=1)
    else:
        model = SVC(kernel='linear', probability=True)
    model.fit(X_norm, y)
    pickle.dump(model, open('classifier.pkl','wb'))
    pickle.dump(in_enc, open('in_enc.pkl','wb'))
    pickle.dump(out_enc, open('out_enc.pkl','wb'))
    return model, in_enc, out_enc

# Initial load or train
if os.path.exists('classifier.pkl'):
    model = pickle.load(open('classifier.pkl','rb'))
    in_enc = pickle.load(open('in_enc.pkl','rb'))
    out_enc = pickle.load(open('out_enc.pkl','rb'))
else:
    emb, names = load_dataset_embeddings()
    model, in_enc, out_enc = train_classifier(emb, names)
```

* **Normalizer** : scales embeddings to unit vectors.
* **LabelEncoder** : maps names ↔ integers.
* **Classifier** : 1‑NN for single class, SVM for ≥2.
* Persist with `pickle`.

---

## 6. Evaluate on Training Set (Optional)

```python
X, y_names = load_dataset_embeddings()
X_norm = in_enc.transform(X)
y_true = out_enc.transform(y_names)
y_pred = model.predict(X_norm)
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true,y_pred,target_names=out_enc.classes_))
```

* Sanity-check performance on the existing dataset.

---

## 7. Interactive Add or Recognize

```python
choice = input("New person? (yes/no):").lower()
if choice=='yes':
    name = input("Name:")
    dir = os.path.join('dataset',name)
    os.makedirs(dir,exist_ok=True)
    print("Add 3 images named 1.jpg,2.jpg,3.jpg in",dir)
    emb,names = load_dataset_embeddings()
    model,in_enc,out_enc = train_classifier(emb,names)

# Recognition
try:
    # script mode: webcam capture
    import __main__
    if getattr(__main__,'__file__',None):
        cam=cv2.VideoCapture(0)
        print("Press SPACE to capture")
        while True:
            ret,frame=cam.read()
            cv2.imshow('recog',frame)
            if cv2.waitKey(1)&0xFF==32:
                face=extract_face(frame)
                break
        cam.release();cv2.destroyAllWindows()
    else:
        path=input("Image path:")
        face=extract_face(cv2.imread(path))
except:
    path=input("Image path:")
    face=extract_face(cv2.imread(path))

emb_vec=get_embedding(face)
emb_norm=in_enc.transform([emb_vec])
pred=model.predict(emb_norm)
print("Recognized:",out_enc.inverse_transform(pred)[0])
```

---

## Approach Overview

1. **Detection** : MTCNN → crop & resize.
2. **Embedding** : FaceNet → 512‑D descriptor.
3. **Normalization** : L2 unit-length scaling.
4. **Classification** : KNN/SVM on embeddings.
5. **Persistence** : Save model & encoders for reuse.
