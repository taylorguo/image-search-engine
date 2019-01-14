#  !/anaconda3/envs/dlib python3.6
# coding: utf-8
############################
import os, pickle,cv2, sys, shutil
import numpy as np
from base_search import Histogram3D, SearchEngine, index_cpickle, search_faces, dataset
import face_recognition
from imutils import build_montages

current_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_path = os.path.join(project_path, "dataset")
stars_folder = os.path.join(dataset_path, "stars")
nobody_folder = os.path.join(dataset_path, "nobody")
montage_folder = os.path.join(project_path, "montage")

detection_method = ["hog", "cnn"]
trained_pickle = "stars.pickle"
pickle_file = os.path.join(project_path, "face-recognition", trained_pickle)

def search_indexed(indexed_cpickle, dataset_path):
    index = pickle.loads(open(indexed_cpickle, "rb").read())
    searcher = SearchEngine(index)
    for (query, nFeatures) in index.items():
        results = searcher.search(nFeatures)
        path = os.path.join(dataset_path, query)
        query_image = cv2.imread(path)
        print("Query: {}".format(query))

        montageA = np.zeros((166*5, 400, 3), dtype="uint8")
        montageB = np.zeros((166*5, 400, 3), dtype="uint8")

        for i in range(0,10):
            (score, image_name) = results[i]
            path = os.path.join(dataset_path, image_name)
            result = cv2.imread(path)
            print("\t{}. {} : {:.3f}".format(i+1, image_name, score))

            if i < 5: montageA[i*166:(i+1)*166, :] = result
            else: montageB[(i-5)*166:((i-5)+1)*166, :] = result

        cv2.imshow("Results 1-5", montageA)
        cv2.imshow("Results 6-10", montageB)
        cv2.waitKey(0)

def search_unindexed_image(indexed_cpickle, dataset_path, testimage):
    test_image = cv2.imread(testimage)
    # cv2.imshow("Searching", test_image)

    descriptor = Histogram3D([8,8,8])
    nfeatures = descriptor.describe(test_image)

    index = pickle.loads(open(indexed_cpickle, "rb").read())
    # print(index)
    searcher = SearchEngine(index)
    results = searcher.search(nfeatures)

    # # 删除展示图片代码
    # montageA = np.zeros((166 * 5, 166, 3), dtype="uint8")
    # montageB = np.zeros((166 * 5, 166, 3), dtype="uint8")
    scores = []
    for i in range(0, 10):
        (score, image_name) = results[i]
        # # 删除展示图片代码
        # path = os.path.join(dataset_path, image_name)
        # result = cv2.imread(path)
        # result = cv2.resize(result, (166, 166))
        print("\t{}. {} : {:.3f}".format(i + 1, image_name, score))
        scores.append(score)
    print(scores)

    # # 删除展示图片的代码
    #     if i < 5:
    #         montageA[i * 166:(i + 1) * 166, :] = result
    #     else:
    #         montageB[(i - 5) * 166:((i - 5) + 1) * 166, :] = result
    #
    #
    # cv2.imshow("Results 1-5", montageA)
    # cv2.imshow("Results 6-10", montageB)
    # cv2.waitKey(0)
    return scores

def find_faces_in_db(trained_pickle, detection_method, image_path):
    print("[INFO] loading encodings...")
    data = pickle.loads(open(trained_pickle, "rb").read())

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=detection_method[1])
    encodings = face_recognition.face_encodings(rgb, boxes)

    # print(encodings, "\n", data["encodings"][50])


    # names = []
    image_db = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.44)
        # matches = search_faces(data["encodings"], encoding, 0.4)

        # name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            # counts = {}
            for i in matchedIdxs:
                # name = data["names"][i]
                img_db = data["images"][i]
                # counts[name] = counts.get(name, 0) + 1
                # print(counts[name])
                image_db.append(img_db)
            # name = max(counts, key=counts.get)
        # names.append(name)
        # image_db.append(img_db)
    m_image_db = []
    # print(img_db)
    for i in image_db:
        img = cv2.imread(i)
        simg = cv2.resize(img, (226,226))
        m_image_db.append(simg)

    montage = build_montages(m_image_db, (226,226), (4,4))[0]
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    image = cv2.resize(image, (226,226))
    cv2.imshow("Resize Image", image)
    cv2.waitKey(0)
    cv2.imshow("Found", montage)
    cv2.waitKey(0)



if __name__ == "__main__":
    # # search_indexed(indexed_pickle, dataset_path)
    # # test_image = "./queries/shire-query.png"
    # # search_unindexed_image(indexed_pickle, dataset_path, test_image)
    # target_img = os.path.join(nobody_folder, "zsq512", "00028.jpg")
    # # target_img = sys.argv[1]
    # find_faces_in_db(pickle_file, detection_method, target_img)

    from base_search import trump_folder
    to_test = os.path.join(current_path, "trump-v")
    if not os.path.exists("not_face"):
        os.mkdir("not_face")

    from imutils.paths import list_images
    for i in list_images(to_test):
        print(i)
        # im2test = cv2.imread(i)
        # cv2.imshow("original",im2test)
        # cv2.waitKey(0)

        # print(i)
        if i.endswith(".png"):
            ss = search_unindexed_image(index_cpickle, trump_folder, i)
            if min(ss) >= float(1):
                shutil.move(i, os.path.join(current_path, "not_face"))

