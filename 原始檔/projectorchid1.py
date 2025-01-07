#蘭花5品種影像分析專案-簡單分析
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#初步圖像處理
# 讀取圖片
image = cv2.imread('.\orchid-genus\inet\cattleya\C1.jpg')
resized_image = cv2.resize(image, (224, 224))  # 調整大小為224x224像素

# OpenCV 預設是 BGR 格式，要轉為 RGB 顯示
image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)    #轉換為灰階圖片

edges = cv2.Canny(gray_image, 100, 200) #Canny 邊緣檢測

# 顯示圖片
#plt.imshow(edges, cmap='gray')
#plt.axis('off')  # 不顯示軸
#plt.show()
#---------------------------------------------------------------------------
#訓練機器學習模型-支持向量機（SVM）
# 資料集的圖片目錄
image_dir = '.\orchid-genus\inet'

# 讀取所有圖片並標註品種（圖片按品種分類放在不同文件夾）
X = []  # 圖像特徵
y = []  # 標籤（品種名稱）

# 每個文件夾的名稱是品種名稱，圖片都放在相應的文件夾中
for folder_name in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                # 讀取圖片
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))  # 將圖片縮放到相同大小

                # 將圖片轉為灰階並展平為一維向量（簡單的特徵提取）
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                flattened_image = gray_image.flatten()  # 展平為一維向量

                # 將特徵和標籤存儲到 X 和 y
                X.append(flattened_image)
                y.append(folder_name)  # 標籤是文件夾名稱，即蘭花品種

# 將 X 和 y 轉換為 numpy 陣列
X = np.array(X)
y = np.array(y)
#---------------------------------------------------------------------------
#數據處理
# 拆分資料集，80% 用於訓練，20% 用於測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#---------------------------------------------------------------------------
#機器學習模型
# 初始化 SVM 模型
model = SVC(kernel='linear')  # 使用線性核

# 訓練模型
model.fit(X_train, y_train)
#---------------------------------------------------------------------------
#評估模型
# 用測試集來預測
y_pred = model.predict(X_test)

# 計算準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'模型的準確度: {accuracy * 100:.2f}%')
#---------------------------------------------------------------------------
#對新圖片進行預測 模型的準確度: 30~40%
# 讀取新的圖片
new_image_path = '.\orchid-genus\orchido1.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224))

# 轉為灰階並展平
gray_new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
flattened_new_image = gray_new_image.flatten()

# 使用模型預測
predicted_class = model.predict([flattened_new_image])  # 這裡需要傳入二維數組

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
if predicted_class[0] == 'cattleya':
    print(f'預測的蘭花品種是: {predicted_class[0]}嘉德麗雅蘭')
    print('花語:',cattleya.get("花語"))
    print('花期:',cattleya.get("花期"))

elif predicted_class[0] == 'dendrobium':
    print(f'預測的蘭花品種是: {predicted_class[0]}石斛蘭')
    print('花語:',dendrobium.get("花語"))
    print('花期:',dendrobium.get("花期"))

elif predicted_class[0] == 'oncidium':
    print(f'預測的蘭花品種是: {predicted_class[0]}文心蘭')
    print('花語:',oncidium.get("花語"))
    print('花期:',oncidium.get("花期"))

elif predicted_class[0] == 'phalaenopsis':
    print(f'預測的蘭花品種是: {predicted_class[0]}蝴蝶蘭')
    print('花語:',phalaenopsis.get("花語"))
    print('花期:',phalaenopsis.get("花期"))

elif predicted_class[0] == 'Vanda':
    print(f'預測的蘭花品種是: {predicted_class[0]}萬代蘭')
    print('花語:',Vanda.get("花語"))
    print('花期:',Vanda.get("花期"))