# based on https://github.com/experiencor/keras-yolo3
from service import yolo3_lib
from service import constants
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
from service import constants
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# model expects input to be a color img with size of 416x416


def initialize_model():
    model = yolo3_lib.make_yolov3_model()
    weight_reader = yolo3_lib.WeightReader(constants.WEIGHTS_FILE_NAME)
    weight_reader.load_weights(model)
    model.save(constants.MODEL_FILE_NAME)


def predict_img(img_path):
    # path = os.path.join(constants.DATA_DIR, constants.EXAMPLE_IMG_NAME)
    image = load_img(img_path)
    og_width, og_height = image.size

    image = load_img(img_path, target_size=(constants.IMG_HEIGHT, constants.IMG_WIDTH))

    img_arr = img_to_array(image)
    img_arr.astype('float32')
    img_arr /= 255.0

    img_arr = expand_dims(img_arr, 0)

    model = load_model(constants.MODEL_FILE_NAME)

    v_boxes, v_labels, v_scores = predict(model, img_arr, og_height, og_width)

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(img_path, v_boxes, v_labels, v_scores)


def predict(model, img_arr, og_height, og_width):

    y_hat = model.predict(img_arr)
    # print('prediction: ', [a.shape for a in y_hat])

    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

    boxes = list()

    for i in range(len(y_hat)):
        boxes += yolo3_lib.decode_netout(y_hat[i][0], anchors[i],
                                         constants.CLASS_THRESHOLD, constants.IMG_HEIGHT, constants.IMG_WIDTH)

    yolo3_lib.correct_yolo_boxes(boxes, og_height, og_width, constants.IMG_HEIGHT, constants.IMG_WIDTH)

    yolo3_lib.do_nms(boxes, 0.5)

    v_boxes, v_labels, v_scores = get_boxes(boxes, constants.CLASS_LABELS, constants.CLASS_THRESHOLD)

    return v_boxes, v_labels, v_scores


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

# draw all results


def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()
