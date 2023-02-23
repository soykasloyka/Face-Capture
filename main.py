import cv2


def face_capture():
    cascade_path = ''

    clf = cv2.CascadeClassifier(cascade_path)
    camera = cv2.VideoCapture('')

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


def main():
    face_capture()

if __name__ == '__main__':
    main()
