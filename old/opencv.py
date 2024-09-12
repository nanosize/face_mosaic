import cv2

# モザイクをかける関数
def mosaic_area(src, x, y, w, h, mosaic_rate=0.1):
    # 顔部分を切り出す
    face = src[y:y+h, x:x+w]
    
    # モザイクをかける
    face = cv2.resize(face, (int(w * mosaic_rate), int(h * mosaic_rate)))
    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 元の画像にモザイクを貼り付ける
    src[y:y+h, x:x+w] = face
    return src

# 画像の読み込み
src = cv2.imread('/Users/nao/programming/face_mosaic/adpDSC_5491--1885830139.jpg')


# 顔検出用のカスケードファイルのパス
face_cascade_path = '/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
# グレースケールに変換
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 顔を検出
faces = face_cascade.detectMultiScale(src_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 顔にモザイクをかける
for (x, y, w, h) in faces:
    src = mosaic_area(src, x, y, w, h)

# モザイクをかけた画像を保存
cv2.imwrite('data/dst/opencv_mosaic_face.jpg', src)

print('モザイク処理が完了しました。')
