import cv2
import numpy as np
import matplotlib.pyplot as plt


def conversion():
    src = cv2.imread("img1.jpg")
    src = cv2.resize(src, (int(src.shape[1] / 2), int(src.shape[0] / 2)))

    '''
    GRAY : 회색조 이미지
    Lab : CIE Lab으로 변환 : L(밝기), A: RED-GREEN 색상 정도, B : YELLOW-BLUE 색상 정도
    YCrCb : Y(휘도 : 밝기), Cb/Cr(색채, 크로마 : 색상 성분)
    HSV : Hue(색상), Saturation(채도), Value(밝기)으로 변환
    '''

    GRAY = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    Lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    YCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    RGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    cv2.imshow("src", src)
    cv2.imshow("GRAY", GRAY)
    cv2.imshow("Lab", Lab)
    cv2.imshow("YCrCb", YCrCb)
    cv2.imshow("HSV", HSV)
    cv2.imshow("RGB", RGB)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale():
    src = cv2.imread('img1.jpg')

    height, width = src.shape[:2]
    res1 = cv2.resize(src, (int(1.2 * width), int(1.2 * height)), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(src, (int(0.6 * width), int(0.6 * height)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("src", src)
    cv2.imshow("res1", res1)
    cv2.imshow("res2", res2)
    print("src.shape=", src.shape)
    print("res1.shape=", res1.shape)
    print("res2.shape=", res2.shape)
    cv2.waitKey()
    cv2.destroyAllWindows()


def translation():
    import numpy as np
    import cv2

    img = cv2.imread('cat1.jpg')
    rows, cols, ch = img.shape

    # x축, y축 , 이동할 거리?
    M = np.float32([[1, 0, 300], [0, 1, 50]])

    # 이미지 변환
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('img1', img)
    cv2.imshow('src', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def revolve():
    img = cv2.imread('cat1.jpg')
    rows, cols, ch = img.shape
    # 중심축 좌표
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)

    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective():
    img = cv2.imread('cat1.jpg')
    rows, cols = img.shape[:2]
    print(rows, cols)

    #  원근 변환은 이미지의 원근을 변경하여 다른 각도나 시각에서 찍은 것처럼 보이도록 하는 작업
    pts1 = np.float32([[150, 50], [400, 50], [60, 450], [310, 450]])
    pts2 = np.float32([[50, 50], [rows - 50, 50], [50, cols - 50], [rows - 50, cols - 50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))

    cv2.imshow("img", img)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def filtering():
    img = cv2.imread('noise.jpg')

    blur1 = cv2.blur(img, (5, 5))
    blur2 = cv2.GaussianBlur(img, (5, 5), 1)
    blur3 = cv2.medianBlur(img, 5)

    # image display
    plt.figure(figsize=(10, 5), dpi=100)
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(141), plt.imshow(img), plt.title("Original")
    plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(blur1), plt.title("Mean Filtering")
    plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(blur2), plt.title("Gauss Filtering")
    plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blur3), plt.title("Median Filtering")
    plt.xticks([]), plt.yticks([])
    plt.show()


def video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('capture', frame)
        key = cv2.waitKey(1)
        if key & 0x0ff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # conversion()
    # scale()
    # translation()
    # revolve()
    # perspective()
    # filtering()
    video()
    pass
