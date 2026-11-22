import numpy as np
import scipy.signal as sci

POOL_SIZE = 2
KERNEL_SIZE = 3
LEARNING_RATE = 0.015
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
    Z_shifted = Z - np.max(Z, axis = 1, keepdims = True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)

def CrossEntropy(Y_hat, Y):
    return -np.sum(Y * np.log(Y_hat + 1e-12))

def max_pooling_with_mask(Z_conv, pool_size = POOL_SIZE):
    B, C, H, W = Z_conv.shape
    z_reshaped = Z_conv.reshape(B, C, H // pool_size, pool_size, W // pool_size, pool_size)
    Z_pool = z_reshaped.max(axis = (3, 5))
    z_repeated = Z_pool.repeat(pool_size, axis = 2).repeat(pool_size, axis = 3)
    mask = (Z_conv == z_repeated).astype(Z_conv.dtype)
    return Z_pool, mask

def max_pooling_backward(dZ_pool, mask, pool_size=POOL_SIZE):
    dZ_repeated = dZ_pool.repeat(pool_size, axis = 2).repeat(pool_size, axis = 3)
    return dZ_repeated * mask

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size = KERNEL_SIZE):
        fan_in = in_channels * kernel_size * kernel_size
        self.K = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)

    def forward(self, X):
        self.X = X
        out_channels, in_channels = self.K.shape[:2]
        in_width, in_height = X.shape[2:4] # X.shape[1:3]
        batch_size = X.shape[0]
        kernel_size = self.K.shape[2]

        Z_conv = np.zeros((
            batch_size,
            out_channels,
            in_width - kernel_size + 1,
            in_height - kernel_size + 1
        ))
        for t in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    Z_conv[t, i] += sci.correlate2d(self.X[t, j], self.K[i, j], mode="valid")
                Z_conv[t, i] += self.b[i]

        Z_pool, self.mask = max_pooling_with_mask(Z_conv)
        self.A = ReLU(Z_pool)

    def backward(self, dA):
        dZ_pool = dA * d_ReLU(self.A)
        dZ_conv = max_pooling_backward(dZ_pool, self.mask)

        batch_size = self.X.shape[0]
        out_channels, in_channels = self.K.shape[:2]
        dK = np.zeros_like(self.K)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)

        for t in range(batch_size): # X.batch_size
            for i in range(out_channels):
                db[i] += np.sum(dZ_conv[t, i])
                for j in range(in_channels):
                    dK[i, j] += sci.correlate2d(self.X[t, j], dZ_conv[t, i], mode='valid')
                    dX[t, j] += sci.convolve2d(dZ_conv[t, i], self.K[i, j], mode='full')

        self.K -= LEARNING_RATE * dK
        self.b -= LEARNING_RATE * db
        return dX

class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        Z = self.X @ self.W + self.b
        self.A = softmax(Z)

    def backward(self, y):

        batch_size = self.X.shape[0]
        dZ = (self.A - y) / batch_size
        dW = self.X.T @ dZ
        db = np.sum(dZ, axis = 0)
        dX = dZ @ self.W.T

        self.W -= LEARNING_RATE * dW
        self.b -= LEARNING_RATE * db
        return dX

epochs = 5
train_samples = len(X_train)

layer_1 = Conv(1, 16)
layer_2 = Conv(16, 32)
mlp = Linear(32 * 6 * 6, 10)

load_weights = True

if load_weights:
    data = np.load("cnn_weights.npz")
    layer_1.K, layer_1.b = data["K1"], data["b1"]
    layer_2.K, layer_2.b = data["K2"], data["b2"]
    mlp.W,     mlp.b     = data["W3"], data["b3"]

for ep in range(epochs):
    indices = np.random.permutation(train_samples)
    total_loss = 0.0
    correct = 0
    samples_seen = 0

    for step, start_idx in enumerate(range(0, train_samples, BATCH_SIZE), 1):
        batch_indices = indices[start_idx : start_idx + BATCH_SIZE]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        batch_size = len(batch_indices)

        layer_1.forward(X_batch)
        layer_2.forward(layer_1.A)
        mlp.forward(layer_2.A.reshape(batch_size, -1))

        total_loss += CrossEntropy(mlp.A, y_batch)
        correct += np.sum(np.argmax(mlp.A, axis = 1) == np.argmax(y_batch, axis = 1))
        samples_seen += batch_size

        dAlayer2 = mlp.backward(y_batch).reshape(layer_2.A.shape)
        dAlayer1 = layer_2.backward(dAlayer2)
        layer_1.backward(dAlayer1)

        if step % 500 == 0:
            print(f"Epoch {ep + 1} / {epochs} | Samples {samples_seen} / {train_samples} | "
                  f"Loss: {total_loss / samples_seen:.4f} | Acc: {correct / samples_seen * 100:.2f}%")

    np.savez(
        "cnn_weights.npz",
        K1 = layer_1.K, b1 = layer_1.b,
        K2 = layer_2.K, b2 = layer_2.b,
        W3 = mlp.W,     b3 = mlp.b
    )
    print(f"==> Epoch Ended {ep + 1} | Loss: {total_loss / train_samples:.4f} | "
          f"Acc: {correct / train_samples * 100:.2f}% -> cnn_weights.npz\n")
