# scripts/predict.py
import joblib
import numpy as np

# 加载模型
model = joblib.load('models/svm_mnist_model.pkl')

def predict(image_784):
    """
    预测手写数字
    参数: image_784 - 784个像素值的列表或数组
    返回: 预测的数字
    """
    # 预处理
    
    image_norm = np.array(image_784) / 255.0
    
    # 预测
    prediction = model.predict(image_norm)[0]
    con = model.decision_function(image_norm)[0]#返回样本到决策边界的距离，即置信度
    confidence = np.max(con)  # 取最大值作为置信度
    return prediction, confidence

# 测试示例
if __name__ == '__main__':
    # 加载一个测试样本
    X =np.loadtxt('data/raw/digits4000/digits4000_digits_vec.txt',delimiter='\t')
    test_sample = X[600].reshape(1,-1)
    
    pred, conf = predict(test_sample)
    print(f"pred 类型: {type(pred)}")
    print(f"conf 类型: {type(conf)}")
    print(f"conf 内容: {conf}")
    
    print(f"预测: {pred}, 置信度: {conf:.3f}")