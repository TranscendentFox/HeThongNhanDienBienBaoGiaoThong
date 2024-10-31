import os

import cv2
import keras
import numpy as np
import pandas as pd
from imageio.v2 import imread
from keras.src.layers import BatchNormalization
from keras.src.layers import Conv2D, MaxPooling2D
from keras.src.layers import Dense, Dropout, Flatten, Input
from keras.src.models import Sequential
from keras.src.optimizers import Adam

from sklearn.metrics import classification_report


def load_data(input_size=(64, 64), data_path='GTSRB/Final_Training/Images'):
    pixels = []
    labels = []
    # Loop qua các thư mục trong thư mục Images
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue

        # Đọc file csv để lấy thông tin về ảnh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')

        # Lăp trong file
        for row in info_file.iterrows():
            # Đọc ảnh
            pixel = imread(os.path.join(class_dir, row[1].Filename))
            # Trích phần ROI theo thông tin trong file csv
            pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
            # Resize về kích cỡ chuẩn
            img = cv2.resize(pixel, input_size)

            # Thêm vào list dữ liệu
            pixels.append(img)

            # Thêm nhãn cho ảnh
            labels.append(row[1].ClassId)

    return pixels, labels


# Đường dẫn ảnh
data_path = 'GTSRB/Final_Training/Images'
pixels, labels = load_data(data_path=data_path)


def split_train_val_test_data(pixels, labels):
    # Chuẩn hoá dữ liệu pixels và labels
    pixels = np.array(pixels)
    labels = keras.utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)
    X = pixels[randomize]
    print("X=", X.shape)
    y = labels[randomize]

    # Chia dữ liệu theo tỷ lệ 60% train và 40% còn lại cho val và test
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    val_size = int(X_val.shape[0] * 0.5)  # 50% của phần 40% bên trên
    X_val, X_test = X_val[:val_size], X_val[val_size:]
    y_val, y_test = y_val[:val_size], y_val[val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(pixels, labels)

df = pd.read_csv('signnames.csv')

class_names = []
for i in df['SignName']:
    class_names.append(i)

model = keras.models.load_model('traffic_sign_model_resnet.keras')
model.predict(X_test)

# Kiểm tra model với dữ liệu mới
print(model.evaluate(X_test, y_test))

# Dự đoán các nhãn cho tập kiểm tra
y_pred = model.predict(X_test)

# Chuyển các dự đoán từ one-hot encoding sang các nhãn
y_pred_classes = np.argmax(y_pred, axis=1)

# Chuyển nhãn thật từ one-hot encoding sang các nhãn
y_true = np.argmax(y_test, axis=1)

# Tạo bảng classification_report
report = classification_report(y_true, y_pred_classes, target_names=class_names)

# In ra bảng classification_report
print(report)