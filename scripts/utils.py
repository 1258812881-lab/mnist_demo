from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
import os
#可处理大部分图片格式，注意png透明底需要加白底，否则可能会影响数据
#图片处理工具，读入图片处理成像素值数组，以便于模型预测
def prepro_real_image(filepath):
    try:
        from PIL import Image
        import numpy as np
        img= Image.open(filepath)
        img=img.convert('L') #转黑白
        width,height=img.size
        if width !=height:  #裁剪为正方形
            size = min(width, height)
            left = (width - size) / 2
            top = (height - size) / 2
            right = (width + size) / 2
            bottom = (height + size) / 2
            img=img.crop((left, top, right, bottom))
        img=img.resize((28,28))
        image_array=np.array(img).reshape(1,-1)
        return image_array
    except ImportError:
        print("错误：未找到 Pillow 库。请运行 'pip install Pillow' 安装它以支持图片加载。")
    
#可视化工具
def visualize_prediction(image, prediction, confidence):
    """可视化预测结果"""
    plt.figure(figsize=(4, 4))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {prediction} (confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

