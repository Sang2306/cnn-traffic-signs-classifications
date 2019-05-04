# Load pickled data
# import pickle
#
# training_file = './traffic-signs-data/train.p'
# validation_file = './traffic-signs-data/valid.p'
# testing_file = './traffic-signs-data/test.p'
#
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
#
# X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']
# Shuffle data
# from sklearn.utils import shuffle

# Normalize the data
# X_train = ((X_train - 127.5) / 127.5)
# X_valid = ((X_valid - 127.5) / 127.5)
# X_test = ((X_test - 127.5) / 127.5)
#
# X_train, y_train = shuffle(X_train, y_train)

# define model
import tensorflow as tf

# BATCH_SIZE = 200

# Load trained model
import_graph = tf.train.import_meta_graph('./lenet.meta')

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

    # def evaluate(X_data, y_data):
    #     num_examples = len(X_data)
    #     total_accuracy = 0
    #     sess = tf.get_default_session()
    #     for offset in range(0, num_examples, BATCH_SIZE):
    #         batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
    #         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
    #         total_accuracy += (accuracy * len(batch_x))
    #     return total_accuracy / num_examples
    #
    # test_accuracy = evaluate(X_test, y_test)
    # print("Test Accuracy = {:.3f}".format(test_accuracy))
    ###########################################################
    """
        Doc thong tin bien bao tu file csv
    """
    import pandas as pd
    csv_file = pd.read_csv('./signnames.csv')
    data = csv_file.to_numpy()
    signs_name = list(data[:, 1])

    import cv2
    import numpy as np
    cap = cv2.VideoCapture('./MVI_106.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        blur = cv2.medianBlur(frame.copy(), 5)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([100, 70, 40])
        upper_blue = np.array([130, 255, 255])

        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        r_mask1 = cv2.inRange(hsv, lower_red, upper_red)
        r_mask2 = cv2.inRange(hsv, np.array([100, 10, 170]), np.array([180, 230, 255]))
        r_mask3 = cv2.inRange(hsv, np.array([157, 60, 11]), np.array([180, 255, 255]))
        red_mask = r_mask1 | r_mask2 | r_mask3

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = blue_mask + red_mask

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        mask = cv2.erode(mask.copy(), None, iterations=2)
        mask2 = cv2.dilate(mask.copy(), None, iterations=2)

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        framedraw = frame.copy()
        if len(cnts) > 0:
            for cnt in cnts:
                _x, _y, w, h = cv2.boundingRect(cnt)
                roi = frame[_y:_y + h, _x:_x + w]
                # extract image and feed to cnn
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (32, 32))
                sign_class = sess.run(prediction, feed_dict={x: np.array([roi])})
                if int(sign_class) == 14:
                    sign_name = signs_name[int(sign_class)]
                    cv2.rectangle(framedraw, (_x, _y), (_x + w, _y + h), (0, 255, 0), 2)
                    cv2.putText(framedraw, sign_name, (_x, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('frame', framedraw)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
