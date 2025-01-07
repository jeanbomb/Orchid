#蘭花5品種影像分析專案-CNN提升準確度

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# 圖像數據集目錄
image_dir = '.\orchid-genus\inet'

# 讀取所有圖片並標註品種
X = []  # 圖像特徵
y = []  # 標籤（品種名稱）

# 遍歷每個文件夾（每個品種的資料）
for folder_name in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                # 讀取圖片
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))  # 調整為224x224

                # 將圖片加入特徵列表，並將對應的品種名稱加入標籤列表
                X.append(image)  # 使用彩色圖像
                y.append(folder_name)  # 標籤是文件夾名稱

# 將 X 和 y 轉換為 numpy 陣列
X = np.array(X)
y = np.array(y)

# 使用 LabelEncoder 將品種名稱轉換為數字標籤
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 使用 one-hot 編碼將標籤轉換為獨熱向量
y_one_hot = to_categorical(y_encoded)

# 將圖像數據進行歸一化處理，將像素值轉換為[0, 1]之間
X = X.astype('float32') / 255.0

# 拆分訓練集和測試集，80% 用於訓練，20% 用於測試
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# 搭建 CNN 模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # 使用品種數量作為輸出節點數
])

# 編譯模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# 讀取新圖片並進行預處理
new_image_path = '.\orchid-genus\orchido1.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224))
new_image = new_image.astype('float32') / 255.0  # 歸一化處理

# 增加一個維度來模擬批量輸入
new_image = np.expand_dims(new_image, axis=0)

# 預測結果
predicted_class = model.predict(new_image)
predicted_class_label = label_encoder.inverse_transform([np.argmax(predicted_class)])

# 在測試集上評估模型的準確度
test_loss, test_accuracy = model.evaluate(X_test, y_test)
# 顯示測試集上的準確度
print(f'模型在測試集上的準確度: {test_accuracy * 100:.2f}%')

#字典
cattleya={
    "花語":"敬愛、傾慕、女性的魅力",
    "花期":"花期不定，一年開一～二次"
}
dendrobium={
    "花語":"歡迎、喜悅",
    "花期":"10月~12月"
}
oncidium={
    "花語":"隱藏的愛，無憂的快樂",
    "花期":"秋~春季"
}
phalaenopsis={
    "花語":"隱藏的愛，無憂的快樂",
    "花期":"11月~4月"
}
Vanda={
    "花語":"隱藏的愛，無憂的快樂",
    "花期":"晚夏~早秋"
}

#輸出結果
if predicted_class_label[0] == 'cattleya':
    print(f'預測的蘭花品種是: {predicted_class_label[0]}嘉德麗雅蘭')
    print('花語:',cattleya.get("花語"))
    print('花期:',cattleya.get("花期"))

elif predicted_class_label[0] == 'dendrobium':
    print(f'預測的蘭花品種是: {predicted_class_label[0]}石斛蘭')
    print('花語:',dendrobium.get("花語"))
    print('花期:',dendrobium.get("花期"))

elif predicted_class_label[0] == 'oncidium':
    print(f'預測的蘭花品種是: {predicted_class_label[0]}文心蘭')
    print('花語:',oncidium.get("花語"))
    print('花期:',oncidium.get("花期"))

elif predicted_class_label[0] == 'phalaenopsis':
    print(f'預測的蘭花品種是: {predicted_class_label[0]}蝴蝶蘭')
    print('花語:',phalaenopsis.get("花語"))
    print('花期:',phalaenopsis.get("花期"))

elif predicted_class_label[0] == 'Vanda':
    print(f'預測的蘭花品種是: {predicted_class_label[0]}萬代蘭')
    print('花語:',Vanda.get("花語"))
    print('花期:',Vanda.get("花期"))
