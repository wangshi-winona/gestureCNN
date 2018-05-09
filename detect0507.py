import cv2
import mymodel as myNN
import json

import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D



minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
fx = 10
fy = 355
fh = 18

isGuessing = False

weightfile = "weight_0508.hdf5"

def binaryMask(frame, x0, y0, width, height):
    global isGuessing, mod

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)

    # if saveImg == True:
    #     saveROIImg(res)
    if isGuessing == True:
        retgesture = guessGesture(mod, res)
        cv2.putText(frame, myNN.output[int(retgesture)], (fx, fy + fh), font, size*3, (0, 255, 0), 1, 1)
        # if lastgesture != retgesture:
        #     lastgesture = retgesture
        #     print lastgesture

            #
            #     ## Checking for only PUNCH gesture here
            #     ## Run this app in Prediction Mode and keep Chrome browser on focus with Internet Off
            #     ## And have fun :) with Dino
            #     if lastgesture == 3:
            #         jump = ''' osascript -e 'tell application "System Events" to key code 49' '''
            #         #jump = ''' osascript -e 'tell application "System Events" to key down (49)' '''
            #         os.system(jump)
            #         print myNN.output[lastgesture] + "= Dino JUMP!"
            #
        time.sleep(0.1)
            #     #guessGesture = False
    # elif visualize == True:
    #     layer = int(raw_input("Enter which layer to visualize "))
    #     cv2.waitKey(1)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False

    return res

# def loadModel():
#     global get_output
#     model = Sequential()
#
#     model.add(Conv2D(myNN.nb_filters, (myNN.nb_conv, myNN.nb_conv),
#                      padding='valid', data_format = 'channels_first',
#                      input_shape=(myNN.img_channels, myNN.img_rows, myNN.img_cols)))
#     convout1 = Activation('relu')
#     model.add(convout1)
#     model.add(Conv2D(myNN.nb_filters, (myNN.nb_conv, myNN.nb_conv)))
#     convout2 = Activation('relu')
#     model.add(convout2)
#     model.add(MaxPooling2D(pool_size=(myNN.nb_pool, myNN.nb_pool)))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(myNN.nb_classes))
#     model.add(Activation('softmax'))
#
#
#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
#     # Model summary
#     model.summary()
#     # Model conig details
#     model.get_config()
#
#
#     print "loading ", weightfile
#     model.load_weights(weightfile)
#
#     layer = model.layers[11]
#     get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
#
#     return model


# This function does the guessing work based on input images
def guessGesture(model, img):
    global get_output


    # Load image and flatten it
    image = np.array(img).flatten()

    # reshape it
    image = image.reshape(myNN.img_channels, myNN.img_rows, myNN.img_cols)

    # float32
    image = image.astype('float32')

    # normalize it
    image = image / 255

    # reshape for NN
    rimage = image.reshape(1, myNN.img_channels, myNN.img_rows, myNN.img_cols)

    # Now feed it to the NN, to fetch the predictions
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([rimage, 0])[0]

    # print prob_array

    d = {}
    i = 0
    for items in myNN.output:
        d[items] = (int)(prob_array[0][i] * 100)
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.iteritems(), key=operator.itemgetter(1))[0]
    prob = d[guess]
    # print d

    if prob > 80.0:
        print guess + "  Probability: ", prob
        # print myNN.output.index(guess)
        # Enable this to save the predictions in a json file,
        # Which can be read by plotter app to plot bar graph
        # dump to the JSON contents to the file

        # with open('gesturejson.txt', 'w') as outfile:
        #     json.dump(d, outfile)

        return myNN.output.index(guess)

    else:
        return 1

def Main():
    global get_output, isGuessing, visualize, mod, x0, y0, width, height

    mod, get_output = myNN.loadCNN(0, weightfile)

    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)

    # set rt size as 640x480
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    while (True):
        ret, frame = cap.read()
        max_area = 0

        frame = cv2.flip(frame, 3)

        if ret == True:
            # if binaryMode == True:
            roi = binaryMask(frame, x0, y0, width, height)
            # else:
            #     roi = skinMask(frame, x0, y0, width, height)

        # cv2.putText(frame, 'Options:', (fx, fy), font, 0.7, (0, 255, 0), 2, 1)
        # cv2.putText(frame, 'b - Toggle Binary/SkinMask', (fx, fy + fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'g - Toggle Prediction Mode', (fx, fy + 2 * fh), font, size, (0, 255, 0), 1, 1)
        # cv2.putText(frame, 'q - Toggle Quiet Mode', (fx, fy + 3 * fh), font, size, (0, 255, 0), 1, 1)
        # cv2.putText(frame, 'n - To enter name of new gesture folder', (fx, fy + 4 * fh), font, size, (0, 255, 0), 1, 1)
        # cv2.putText(frame, 's - To start capturing new gestures for training', (fx, fy + 5 * fh), font, size,
        #             (0, 255, 0), 1, 1)
        cv2.putText(frame, 'ESC - Exit', (fx, fy + 3 * fh), font, size, (0, 255, 0), 1, 1)

        cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        ## Use Esc key to close the program
        if key == 27:
            break

        ## Use g key to start gesture predictions via CNN
        elif key == ord('g'):
            isGuessing = not isGuessing
            print "Prediction Mode - {}".format(isGuessing)


        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5


    # Realse & destroy
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()