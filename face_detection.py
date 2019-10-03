import sys
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Color:
    BOLD = '\033[1m'
    UNDER = '\033[4m'
    END = '\033[0m'
    my_cv2 = 50, 30, 255


def face_detection(live_or_file: int):
    """
    Perform face_detection on a picture using MTCNN implementation.
    """
    detector = MTCNN()
    if live_or_file == 0:
        camera_feed = cv2.VideoCapture(0)
    while True:
        if live_or_file == 0:
            _, image = camera_feed.read()
        else:
            image = cv2.imread(live_or_file, cv2.IMREAD_COLOR)
        filename = 'Captured_image.png'
        cv2.imwrite(filename, image)
        results = detector.detect_faces(image)
        nbr_of_faces = len(results)
        # Filter : keep faces with confidence over 0.90
        j = 0
        while j < nbr_of_faces:
            if results[j]['confidence'] < 0.90:
                results.pop(j)
                nbr_of_faces = len(results)
                j -= 1
            j += 1
        # Retrieve a bounding box for each person in results json object.
        # Display the confidence and the bounding box on the picture.
        for person in results:
            bb = person['box']
            font = cv2.FONT_HERSHEY_PLAIN
            confidence_value = round(person['confidence'] * 100, 2)
            confidence_percent = str(confidence_value) + "%"
            cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), Color.my_cv2, 3)
            cv2.putText(image, confidence_percent, (bb[0], bb[1] - 3), font, 1, Color.my_cv2, 1, cv2.LINE_AA)
        cv2.imwrite(filename, image)
        cv2.imshow('Camera Capture', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if live_or_file == 0:
        camera_feed.release()
    cv2.destroyAllWindows()


def main():
    """
    Detect faces on a live camera feed or on a picture or video file.

    For a detection on a live camera feed, use no argument. Else your argument should be the path of your file.
    Usage case 1 : python face_detection.py
    Usage case 2 : python face_detection.py FILE_PATH
    """
    if len(sys.argv) == 1:
        face_detection(0)
    elif len(sys.argv) == 2:
        face_detection(sys.argv[1])
    else:
        print(Color.UNDER + 'Usage' + Color.END + ': ' + Color.BOLD
              + 'python face_detection.py ' + Color.END + '['
              + Color.BOLD + 'FILE_PATH' + Color.END + '] \n\n\tUse ' + Color.BOLD
              + 'no arguments' + Color.END + ' to perform face detection on a camera feed.\n\tUse ' + Color.BOLD
              + 'FILE_PATH' + Color.END + ' to perform face detection on a image or a video.')
    return


if __name__ == "__main__":
    main()
