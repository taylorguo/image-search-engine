#  !/anaconda3/envs/dlib python3.6
# coding: utf-8
############################
import os, pickle, cv2, imutils
import numpy as np
from face_recognition import face_distance
from imutils.paths import list_images

project_path = os.path.dirname(os.path.realpath(__file__))
# indexed_pickle = "index.cpickle"
index_cpickle = "trump.cpickle"

dataset_folder = "dataset"
dataset = os.path.join(project_path, dataset_folder)
trump_folder = os.path.join(dataset, "trump")

class Histogram3D:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image):
        hist = cv2.calcHist([image], [0,1,2], None, self.bins, [0,256,0,256,0,256])
        if imutils.is_cv2(): hist = cv2.normalize(hist)
        else: hist= cv2.normalize(hist, hist)
        return hist.flatten()

class SearchEngine:
    def __init__(self, index):
        self.index = index
    def chi2distance(self, hA, hB, eps = 1e-10):
        d = 0.5 * np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(hA,hB)])
        return d
    def search(self, nfeatures):
        results = {}
        for (k, features) in self.index.items():
            results[k] = self.chi2distance(features, nfeatures)
        results = sorted([(v, k) for (k, v) in results.items()])
        return results

def index_dataset(dataset_path, indexed_pickle):
    index = {}
    descriptor = Histogram3D([8,8,8])
    for image_path in list_images(dataset_path):
        image = image_path[image_path.rfind("/")+1:]
        index[image] = descriptor.describe(cv2.imread(image_path))
    with open(indexed_pickle, "wb") as pickle_file:
        pickle_file.write(pickle.dumps(index))
    print("[INFO] done...indexed {} images".format(len(index)))

def search_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
    # print(face_distance(known_face_encodings, face_encoding_to_check))
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)



if __name__ == "__main__":

    index_dataset(trump_folder, index_cpickle)