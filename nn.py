import numpy as np

# 激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.01):
        """
        input_size: 输入层神经元数量
        hidden_size: 隐藏层神经元数量
        output_size: 输出层神经元数量
        activation: 激活函数类型 ('relu' 或 'sigmoid')
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        # 设置激活函数及其导数
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError(f"未知的激活函数 {activation}")
    
    def forward(self, X):
        """
        X: 输入数据 (N, D)，其中 N 是样本数量，D 是特征维度
        retun (N, O)，其中 O 是输出维度
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # 输出层使用sigmoid激活函数
        return self.a2
    
    def backward(self, X, y, output):
        """
        反向传播
        X: 输入数据 (N, D)
        y: 真实标签 (N, O)
        output: 预测输出 (N, O)
        """
        m = X.shape[0]
        
        # 计算损失函数关于z2的梯度
        dz2 = output - y
        
        # 计算损失函数关于W2的梯度
        dW2 = np.dot(self.a1.T, dz2) / m
        
        # 计算损失函数关于b2的梯度
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 计算损失函数关于a1的梯度
        da1 = np.dot(dz2, self.W2.T)
        
        # 计算损失函数关于z1的梯度
        dz1 = da1 * self.activation_derivative(self.z1)
        
        # 计算损失函数关于W1的梯度
        dW1 = np.dot(X.T, dz1) / m
        
        # 计算损失函数关于b1的梯度
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 10 == 0:
                loss = self.compute_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def compute_loss(self, y_true, y_pred):
        # 计算交叉熵损失
        epsilon = 1e-8
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss
    
    def predict(self, X):
        output = self.forward(X)
        predictions = (output >= 0.5).astype(int)
        return predictions.flatten()


# 生成随机二分类数据集
np.random.seed(42)
num_samples = 1000
X = np.random.randn(num_samples, 2)
# 根据和是否大于0来生成标签，方便训练
y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)

input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
activation = 'relu'

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, activation, learning_rate)

# train
nn.train(X, y, epochs=1000)

# predict
predictions = nn.predict(X)
accuracy = np.mean(predictions == y.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")
