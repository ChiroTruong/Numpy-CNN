import numpy as np
import scipy.signal as sci

POOL_SIZE = 2
KERNEL_SIZE = 3
LEARNING_RATE = 0.002
BATCH_SIZE = 8

def load_mnist_images_ubyte(filename):
    with open(filename, 'rb') as f:
        _magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols)
    # AI generated

def load_mnist_labels_ubyte(filename):
    with open(filename, 'rb') as f:
        _magic, num_items = np.frombuffer(f.read(8), dtype='>i4')
        return np.frombuffer(f.read(), dtype=np.uint8)
    # AI generated

X_train = load_mnist_images_ubyte('archive/train-images-idx3-ubyte')
X_train = X_train[:, np.newaxis, :, :].astype(np.float32) / 255.0
X_train = np.pad(X_train, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

y_train = load_mnist_labels_ubyte('archive/train-labels-idx1-ubyte')
y_train = np.eye(10, dtype=np.float32)[y_train]

def ReLU(Z):
    return np.maximum(0, Z)

def d_ReLU(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z)

def CrossEntropy(Y_hat, Y):
    return -np.sum(Y * np.log(Y_hat + 1e-12))

def max_pooling_with_mask(Z_conv, pool_size):
    C, H, W = Z_conv.shape
    z_reshaped = Z_conv.reshape(C, H // pool_size, pool_size, W // pool_size, pool_size)
    Z_pool = z_reshaped.max(axis=(2, 4))
    z_repeated = Z_pool.repeat(pool_size, axis=1).repeat(pool_size, axis=2)
    mask = (Z_conv == z_repeated).astype(float)
    return Z_pool, mask
    # AI generated

def max_pooling_backward(dZ_pool, mask, pool_size=POOL_SIZE):
    dZ_repeated = dZ_pool.repeat(pool_size, axis=1).repeat(pool_size, axis=2)
    return dZ_repeated * mask

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE):
        fan_in = in_channels * kernel_size * kernel_size
        self.K = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)

    def forward(self, X):
        self.X = X
        out_channels, in_channels = self.K.shape[:2]
        in_width, in_height = X.shape[1:3]
        kernel_size = self.K.shape[2]

        Z_conv = np.zeros((
            out_channels,
            in_width - kernel_size + 1,
            in_height - kernel_size + 1
        ))

        for i in range(out_channels):
            for j in range(in_channels):
                Z_conv[i] += sci.correlate2d(self.X[j], self.K[i, j], mode="valid")
            Z_conv[i] += self.b[i]

        Z_pool, self.mask = max_pooling_with_mask(Z_conv, POOL_SIZE)
        self.A = ReLU(Z_pool)

    def backward(self, dA):
        dZ_pool = dA * d_ReLU(self.A)
        dZ_conv = max_pooling_backward(dZ_pool, self.mask)

        out_channels, in_channels = self.K.shape[:2]
        dK = np.zeros_like(self.K)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)

        for i in range(out_channels):
            db[i] = np.sum(dZ_conv[i])
            for j in range(in_channels):
                dK[i, j] = sci.correlate2d(self.X[j], dZ_conv[i], mode='valid')
                dX[j] += sci.convolve2d(dZ_conv[i], self.K[i, j], mode='full')

        self.K -= LEARNING_RATE * dK
        self.b -= LEARNING_RATE * db
        return dX

class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        Z = self.W.T @ self.X + self.b
        self.A = softmax(Z)

    def backward(self, y):
        dZ = self.A - y
        dW = np.outer(self.X, dZ)
        dX = self.W @ dZ

        self.W -= LEARNING_RATE * dW
        self.b -= LEARNING_RATE * dZ
        return dX

epochs = 5
train_samples = len(X_train)

layer_1 = Conv(1, 16)
layer_2 = Conv(16, 32)
mlp = Linear(32 * 6 * 6, 10)

load_weights = False

if load_weights == True:
    data = np.load("cnn_weights.npz")
    layer_1.K, layer_1.b = data["K1"], data["b1"]
    layer_2.K, layer_2.b = data["K2"], data["b2"]
    mlp.W,     mlp.b     = data["W3"], data["b3"]

for ep in range(epochs):
    indices = np.random.permutation(train_samples)
    total_loss = 0.0
    correct = 0

    for step, idx in enumerate(indices, 1):
        layer_1.forward(X_train[idx])
        layer_2.forward(layer_1.A)
        mlp.forward(layer_2.A.reshape(-1))

        total_loss += CrossEntropy(mlp.A, y_train[idx])
        correct += (np.argmax(mlp.A) == np.argmax(y_train[idx]))

        dAlayer2 = mlp.backward(y_train[idx]).reshape(layer_2.A.shape)
        dAlayer1 = layer_2.backward(dAlayer2)
        layer_1.backward(dAlayer1)

        if step % 500 == 0:
            print(f"Epoch {ep + 1} / {epochs} | Step {step} / {train_samples} | "
                  f"Loss: {total_loss / step:.4f} | Acc: {correct / step * 100:.2f}%")

    np.savez(
        "cnn_weights.npz",
        K1 = layer_1.K, b1 = layer_1.b,
        K2 = layer_2.K, b2 = layer_2.b,
        W3 = mlp.W,     b3 = mlp.b
    )
    print(f"==> Epoch Ended {ep + 1} | Loss: {total_loss / train_samples:.4f} | "
          f"Acc: {correct / train_samples * 100:.2f}% -> cnn_weights.npz\n")
