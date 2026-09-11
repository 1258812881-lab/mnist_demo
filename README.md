# 手写字项目handwritten_digit_recognition
---
项目结构树（预期）
```
handwritten_digit_recognition/
│
├── README.md                          # 项目说明文档
├── requirements.txt                   # 依赖包列表
├── .gitignore                        # Git忽略文件
│
├── data/                              # 数据目录
│   ├── raw/digits4000/                # 原始数据
│   │   ├── digits4000_digits_labels.txt
│   │   ├── digits4000_digits_vec.txt
│   │   └── train_indices.txt
│   ├── processed/                    # 处理后的数据
│   │   ├── digits4000_normlized.txt
│   │   └── digits4000_pca.txt
│   └── external/                     # 自己手写的图片
│       └── handwritten_digit.jpg
│
├── notebooks/                # Jupyter Notebook（实验和探索）
│   ├── preprocess.ipynb     # 数据探索和可视化
│   ├── Model_training.ipynb       # 模型训练和调参
│   └── Performance_eva.ipynb      # 模型评估
│
├── scripts/                           # Python 脚本
│   ├── __init__.py
│   ├── predict.py                    # 预测脚本
│   └── utils.py                     # 工具函数
│
├── models/                            # 保存的模型
│   ├── svc_model.pkl           # 最佳 SVC 模型
│   └── model_metadata.json          # 模型元数据
│
├── tests/  
│   ├── test_predict.py                            # 单元测试
│   └── result.png

```
