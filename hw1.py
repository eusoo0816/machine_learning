# svm_mlp_trace.py
import numpy as np
import matplotlib.pyplot as plt

# ========= WORK =========
def make_blobs(n_samples=600, centers=2, dim=2, spread=1.2, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-3, 3, size=(centers, dim))
    X, y = [], []
    per = n_samples // centers
    for i in range(centers):
        Xi = means[i] + spread * rng.standard_normal(size=(per, dim))
        X.append(Xi); y.append(np.full(per, i))
    X = np.vstack(X); y = np.concatenate(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def one_hot(y, K):
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), y] = 1.0
    return Y

# ========= SVM（線性 Hinge） =========
class LinearSVMTrace:
    def __init__(self, C=1.0, lr=5e-3, epochs=10, batch_size=64, seed=0,
                 log_every=10, verbose_first_n=5):
        self.C = C; self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.log_every = log_every
        self.verbose_first_n = verbose_first_n
        self.w = None; self.b = 0.0
        # traces
        self.w_norms = []; self.dw_norms = []; self.losses = []
        self.train_acc_hist = []; self.test_acc_hist = []

    @staticmethod
    def hinge_loss_reg(w, b, X, y, C):
        # y in {-1,+1}
        margins = 1 - y * (X @ w + b)
        hinge = np.maximum(0, margins).mean()
        reg = 0.5 * np.dot(w, w)
        return reg + C * hinge

    def fit(self, X, y, Xval=None, yval=None):
        # y -> {-1,+1}
        y2 = np.where(y == 0, -1, y)
        self.w = np.zeros(X.shape[1]); self.b = 0.0

        step = 0
        for ep in range(self.epochs):
            idx = self.rng.permutation(len(X))
            Xs, ys = X[idx], y2[idx]

            for start in range(0, len(X), self.batch_size):
                Xe = Xs[start:start+self.batch_size]
                ye = ys[start:start+self.batch_size]
                if len(Xe) == 0: continue

                for xi, yi in zip(Xe, ye):
                    margin = yi * (np.dot(self.w, xi) + self.b)

                    w_before = self.w.copy()
                    if margin >= 1:
                        # w <- (1 - lr) w； b 不變
                        self.w = (1 - self.lr) * self.w
                        db = 0.0
                    else:
                        # w <- (1 - lr) w + lr*C*yi*xi； b <- b + lr*C*yi
                        self.w = (1 - self.lr) * self.w + self.lr * self.C * yi * xi
                        db = self.lr * self.C * yi
                        self.b += db

                    dw = self.w - w_before

                    # ----  PRINT----
                    if step < self.verbose_first_n:
                        print(f"[SVM step {step}]")
                        print("  w(before)=", w_before)
                        print("  Δw       =", dw)
                        print("  w(after) =", self.w)
                        print("  Δw‖      =", np.linalg.norm(dw), "  ‖w‖=", np.linalg.norm(self.w))

                    if step % self.log_every == 0:
                        L = self.hinge_loss_reg(self.w, self.b, X, y2, self.C)
                        self.losses.append(L)
                        self.w_norms.append(np.linalg.norm(self.w))
                        self.dw_norms.append(np.linalg.norm(dw))
                    step += 1

            # epoch 結尾記錄 acc
            self.train_acc_hist.append(self.score(X, (y2+1)//2))
            if Xval is not None:
                self.test_acc_hist.append(self.score(Xval, yval))

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X, y):  # y in {0,1}
        return np.mean(self.predict(X) == y)

# ========= MLP（1 hidden，ReLU+Softmax） =========
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(z.dtype)
def softmax(Z):  # Z: (B,K)
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z); return e / e.sum(axis=1, keepdims=True)

class MLPTrace:
    def __init__(self, in_dim, hidden_dim, out_dim,
                 lr=5e-3, epochs=60, batch_size=64, seed=0, l2=1e-4,
                 log_every=10, verbose_first_n=5):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/in_dim), size=(hidden_dim, in_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = rng.normal(0, np.sqrt(2/hidden_dim), size=(out_dim, hidden_dim))
        self.b2 = np.zeros((out_dim,))
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size; self.l2 = l2
        self.rng = rng
        self.log_every = log_every; self.verbose_first_n = verbose_first_n
        # traces
        self.losses = []; self.dw1_norms = []; self.dw2_norms = []
        self.train_acc_hist = []; self.test_acc_hist = []

    @staticmethod
    def ce_loss(P, Y, W1, W2, l2):
        # P/Y: (B,K)
        eps = 1e-12
        ce = -np.sum(Y * np.log(P + eps)) / Y.shape[0]
        reg = 0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2))
        return ce + reg

    def fit(self, X, y, Xval=None, yval=None):
        n, d = X.shape; K = int(y.max()) + 1
        Y = one_hot(y, K)
        step = 0
        for ep in range(self.epochs):
            idx = self.rng.permutation(n); Xs, Ys = X[idx], Y[idx]
            for st in range(0, n, self.batch_size):
                Xe = Xs[st:st+self.batch_size]; Ye = Ys[st:st+self.batch_size]
                if len(Xe) == 0: continue

                # forward
                Z1 = Xe @ self.W1.T + self.b1            # (B,H)
                H  = relu(Z1)                             # (B,H)
                Z2 = H  @ self.W2.T + self.b2            # (B,K)
                P  = softmax(Z2)                          # (B,K)

                # grads
                dZ2 = (P - Ye) / Xe.shape[0]              # (B,K)
                dW2 = dZ2.T @ H + self.l2 * self.W2       # (K,H)
                db2 = dZ2.sum(axis=0)

                dH  = dZ2 @ self.W2                       # (B,H)
                dZ1 = dH * relu_grad(Z1)                  # (B,H)
                dW1 = dZ1.T @ Xe + self.l2 * self.W1      # (H,d)
                db1 = dZ1.sum(axis=0)

                # keep copy to量測 ΔW 的範數
                W1_before = self.W1.copy(); W2_before = self.W2.copy()

                # update  (w* = w + Δw, 其中 Δw = -lr * grad)
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

                dW1_step = self.W1 - W1_before
                dW2_step = self.W2 - W2_before

                if step < self.verbose_first_n:
                    print(f"[MLP step {step}]")
                    print("  ‖ΔW1‖=", np.linalg.norm(dW1_step),
                          " ‖ΔW2‖=", np.linalg.norm(dW2_step))

                if step % self.log_every == 0:
                    loss = self.ce_loss(P, Ye, self.W1, self.W2, self.l2)
                    self.losses.append(loss)
                    self.dw1_norms.append(np.linalg.norm(dW1_step))
                    self.dw2_norms.append(np.linalg.norm(dW2_step))
                step += 1

            # epoch end: accuracy
            self.train_acc_hist.append(self.score(X, y))
            if Xval is not None:
                self.test_acc_hist.append(self.score(Xval, yval))

    def predict(self, X):
        Z1 = X @ self.W1.T + self.b1
        H  = relu(Z1)
        Z2 = H @ self.W2.T + self.b2
        P  = softmax(Z2)
        return np.argmax(P, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ========= 主程式 =========
if __name__ == "__main__":
    # 控制輸出粒度
    LOG_EVERY = 10          # 每幾步記錄一次曲線點
    VERBOSE_FIRST_N = 3     # 前 N 步印出 w 與 Δw 的細節

    # ---------- SVM: 二類 ----------
    X2, y2 = make_blobs(n_samples=600, centers=2, dim=2, spread=1.2, seed=42)
    split = int(0.8*len(X2))
    Xtr2, Xte2 = X2[:split], X2[split:]
    ytr2, yte2 = y2[:split], y2[split:]

    svm = LinearSVMTrace(C=1.0, lr=5e-3, epochs=10, batch_size=64, seed=0,
                         log_every=LOG_EVERY, verbose_first_n=VERBOSE_FIRST_N)
    svm.fit(Xtr2, ytr2, Xte2, yte2)

    print("SVM train acc:", svm.score(Xtr2, ytr2))
    print("SVM test  acc:", svm.score(Xte2, yte2))

    # ---------- MLP: 三類 ----------
    X3, y3 = make_blobs(n_samples=900, centers=3, dim=2, spread=1.3, seed=7)
    split = int(0.8*len(X3))
    Xtr3, Xte3 = X3[:split], X3[split:]
    ytr3, yte3 = y3[:split], y3[split:]

    mlp = MLPTrace(in_dim=2, hidden_dim=32, out_dim=3, lr=5e-3, epochs=80,
                   batch_size=64, seed=0, l2=1e-4,
                   log_every=LOG_EVERY, verbose_first_n=VERBOSE_FIRST_N)
    mlp.fit(Xtr3, ytr3, Xte3, yte3)

    print("MLP train acc:", mlp.score(Xtr3, ytr3))
    print("MLP test  acc:", mlp.score(Xte3, yte3))

    # ---------- 繪圖（各用一張圖，避免擠） ----------
    # SVM：loss
    plt.figure(); plt.plot(svm.losses); plt.title("SVM: Regularized Hinge Loss"); plt.xlabel(f"step / {LOG_EVERY}"); plt.ylabel("loss"); plt.grid(True)
    # SVM：‖w‖、‖Δw‖
    plt.figure(); plt.plot(svm.w_norms, label="‖w‖"); plt.plot(svm.dw_norms, label="‖Δw‖"); plt.title("SVM: ‖w‖ and ‖Δw‖ over steps"); plt.xlabel(f"step / {LOG_EVERY}"); plt.legend(); plt.grid(True)
    # SVM：train/test acc（以 epoch 為 x 軸）
    plt.figure(); plt.plot(svm.train_acc_hist, label="train"); 
    if len(svm.test_acc_hist)>0: plt.plot(svm.test_acc_hist, label="test");
    plt.title("SVM: Accuracy per epoch"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.grid(True)

    # MLP：loss
    plt.figure(); plt.plot(mlp.losses); plt.title("MLP: Cross-Entropy Loss"); plt.xlabel(f"step / {LOG_EVERY}"); plt.ylabel("loss"); plt.grid(True)
    # MLP：‖ΔW1‖、‖ΔW2‖
    plt.figure(); plt.plot(mlp.dw1_norms, label="‖ΔW1‖"); plt.plot(mlp.dw2_norms, label="‖ΔW2‖"); plt.title("MLP: Weight Update Norms"); plt.xlabel(f"step / {LOG_EVERY}"); plt.legend(); plt.grid(True)
    # MLP：train/test acc
    plt.figure(); plt.plot(mlp.train_acc_hist, label="train");
    if len(mlp.test_acc_hist)>0: plt.plot(mlp.test_acc_hist, label="test");
    plt.title("MLP: Accuracy per epoch"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.grid(True)

    plt.show()
