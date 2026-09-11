#需要导入模型预测脚本进行测试
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#测试1：用测试集的单个数据测试script/的预测脚本可否正常运行
"""
from scripts.predict import predict
import numpy as np
# 加载一个测试样本
X =np.loadtxt('data/raw/digits4000/digits4000_digits_vec.txt',delimiter='\t')
test_sample = X[600].reshape(1,-1)
print(f"test_sample 类型: {predict(test_sample)}")
"""
#测试2: 借助script/的工具具脚本处理自己手写图片(存入data/external)，并再次导入预测脚本测试模型效果
from scripts.predict import predict
from scripts.utils import prepro_real_image,visualize_prediction
img=prepro_real_image('data/external/handwritten_digit.jpg')
pred,coe=predict(img)
print(visualize_prediction(img, pred, coe))
print(f"预测: {pred}, 置信度: {coe:.3f}")