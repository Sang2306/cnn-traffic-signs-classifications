import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
# Shuffle data
from sklearn.utils import shuffle

# Normalize the data
X_train = ((X_train - 127.5) / 127.5)
X_valid = ((X_valid - 127.5) / 127.5)
X_test = ((X_test - 127.5) / 127.5)
X_train, y_train = shuffle(X_train, y_train)
BATCH_SIZE = 695


def constrast_limit(image):
    """
        Can bang hist cua do sang anh trong khong gian mau YCrCb
    :param image:
    :return: img_hist_equalized
    """
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def laplacian_of_gaussian(image):
    """
        Su dung bo loc nhieu Gaussian va Laplacian
    :param image:
    :return: | LoG_image |
    """
    LoG_image = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image


def binarization(image):
    """
        Phan nguong anh
    :param image:
    :return: thresh
    """
    thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)[1]
    return thresh


def preprocess_image(image):
    """
    :param image:
    :return: image
    """
    image = constrast_limit(image)
    image = laplacian_of_gaussian(image)
    image = binarization(image)
    return image


def remove_small_components(image, threshold):
    """
        Loai bo cac thanh phan nho
    :param image:
    :param threshold:
    :return:
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros(output.shape, dtype=np.uint8)
    # chi giu lai cac thanh phan co kich thuoc > threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def contour_is_sign(contour, centroid, threshold):
    """
    :param contour:
    :param centroid:
    :param threshold:
    :return: (Bool, value)
    """
    distance_list = []
    for element in contour:
        first_coor = element[0]
        distance = np.sqrt((first_coor[0] - centroid[0]) ** 2 + (first_coor[1] - centroid[1]) ** 2)
        distance_list.append(distance)
    max_distance = max(distance_list)
    signature = [float(dist) / max_distance for dist in distance_list]
    # kiem tra co phai la bien bao hay khong.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:  # sign
        return True, max_distance + 2
    else:  # not sign
        return False, max_distance + 2


def crop_sign(image, coordinate):
    """
    :param image:
    :param coordinate:
    :return: cropped sign
    """
    width = image.shape[1]
    height = image.shape[0]
    left = max([int(coordinate[0][0]), 0])
    top = max([int(coordinate[0][1]), 0])
    right = min([int(coordinate[1][0]), width - 1])
    bottom = min([int(coordinate[1][1]), height - 1])
    return image[top - 8:bottom + 8, left - 8:right + 8]


def find_lagest_sign(image, contour, threshold, distance_theshold):
    """
    :param image:
    :param contour:
    :param threshold:
    :param distance_theshold:
    :return: (sign, coordinate)
    """
    max_distance = 0
    coordinate = None
    sign = None
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return
    c_x = int(M["m10"] / M["m00"])
    c_y = int(M["m01"] / M["m00"])
    is_sign, distance = contour_is_sign(contour, [c_x, c_y], 1 - threshold)
    if is_sign and distance > max_distance and distance > distance_theshold:
        coordinate = np.reshape(contour, [-1, 2])
        left, top = np.amin(coordinate, axis=0)
        right, bottom = np.amax(coordinate, axis=0)
        coordinate = [(left, top), (right, bottom)]
        sign = crop_sign(image, coordinate)
    return sign, coordinate


import_graph = tf.train.import_meta_graph('./lenet.meta')


def main(args):
    with tf.Session() as sess:
        import_graph.restore(sess, tf.train.latest_checkpoint('.'))

        fc2 = tf.get_default_graph().get_tensor_by_name('fc2:0')
        fc3_W = tf.get_default_graph().get_tensor_by_name('fc3_W:0')
        fc3_b = tf.get_default_graph().get_tensor_by_name('fc3_b:0')
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        one_hot_y = tf.get_default_graph().get_tensor_by_name('one_hot_y:0')
        prediction = tf.get_default_graph().get_tensor_by_name('prediction:0')
        correct_prediction = tf.get_default_graph().get_tensor_by_name('correct_prediction:0')
        accuracy_operation = tf.get_default_graph().get_tensor_by_name('accuracy_operation:0')

        # tf.summary.FileWriter('./graph', tf.get_default_graph())

        def evaluate(X_data, y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

        # import  imageio
        # import matplotlib.pyplot as plt
        # stop = imageio.imread('./extra_signs/stop.jpg')
        # # stop = cv2.cvtColor(stop, cv2.COLOR_BGR2RGB)
        # plt.imshow(stop)
        # sign_class = sess.run(prediction, feed_dict={x: np.array([stop])})
        # print('Sign class: ', sign_class)                                                                                                                                  # print("Test Accuracy = {:.3f}".format(test_accuracy))
        # return
        ###########################################################
        """
            Doc thong tin bien bao tu file csv
        """
        csv_file = pd.read_csv(args.csv)
        data = csv_file.to_numpy()
        signs_name = list(data[:, 1])
        for i, name in enumerate(signs_name):
            print(i, '\t', name)
        """
            Phan nhan phat hien bien bao
        """
        camera = cv2.VideoCapture(args.video)
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            frame_copy = frame.copy()
            # xu ly hinh anh
            binary_image = preprocess_image(frame_copy)
            binary_image = remove_small_components(binary_image, threshold=300)
            # Giu lai cac mau blue, red, white, black
            blur = cv2.GaussianBlur(frame_copy, (3, 3), 0)
            # BGR to HSV
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # Tim red color
            r_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            r_mask2 = cv2.inRange(hsv, np.array([100, 10, 170]), np.array([180, 230, 255]))
            r_mask3 = cv2.inRange(hsv, np.array([157, 60, 11]), np.array([180, 255, 255]))
            red_mask = r_mask1 | r_mask2 | r_mask3
            # Tim blue color
            blue_mask = cv2.inRange(hsv, np.array([100, 70, 40]), np.array([130, 255, 255]))
            # Tim white color
            white_mask = cv2.inRange(hsv, np.array([0, 0, 255]), np.array([255, 255, 255]))
            # Tim black mask
            black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([170, 150, 50]))
            # Ket hop 4 mask
            blue_red_mask = cv2.bitwise_or(blue_mask, red_mask)
            white_black_mask = cv2.bitwise_or(white_mask, black_mask)
            mask = cv2.bitwise_or(blue_red_mask, white_black_mask)
            # Ap dung mat na vao anh nhi phan de loai bo
            binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
            # Tim tat ca cac contours(bien) trong binary_image
            cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.imshow("BINARY_IMAGE", binary_image)
            if len(cnts) > 0:
                for cnt in cnts:
                    # Chi lay ROI cua bien bao gan camera nhat de predict
                    try:
                        sign, coordinate = find_lagest_sign(image=frame, contour=cnt, threshold=0.7,
                                                            distance_theshold=15)
                        if coordinate is not None:
                            cv2.rectangle(frame_copy, coordinate[0], coordinate[1], (255, 255, 255), 1)
                        sign = cv2.resize(sign, (32, 32), 1, 1)
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Sign', sign)
                        sign_class = sess.run(prediction, feed_dict={x: np.array([sign])})
                        if sign_class[0]:
                            sign_name = signs_name[sign_class[0]]
                            cv2.rectangle(frame_copy, coordinate[0], coordinate[1], (0, 255, 0), 2)
                            # Vi tri ve text box
                            text_box = [(coordinate[0][0], coordinate[0][1] - 15),
                                        (coordinate[1][0] + (len(sign_name) * 2), coordinate[0][1])]
                            cv2.rectangle(frame_copy, text_box[0], text_box[1], (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame_copy, sign_name, (coordinate[0][0], coordinate[0][1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (0, 0, 0), thickness=2)
                    except Exception as e:
                        pass
            cv2.imshow('VIDEO', frame_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Ap dung CNN vao viec recognition bien bao duong bo"
    )
    parser.add_argument(
        'video',
        metavar='video_file_name',
        help='ten file video - mac dinh su dung webcam',
        default=0
    ).required = False

    parser.add_argument(
        'csv',
        metavar='cvs_file_name',
        help='file csv chua ten bien bao',
        default='./signnames.csv',
    ).required = False

    args = parser.parse_args()
    main(args)
