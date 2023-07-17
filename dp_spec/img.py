import cv2


def preprocess_img(img):
    """crop the useless part (CarRacing-v2-specific cropping) and convert from rgb to gray"""

    return cv2.cvtColor(img[:84, 6:90], cv2.COLOR_RGB2GRAY) / 255
