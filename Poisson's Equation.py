import sys
import os
sys.path.insert(0, '../Utilities/')
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.control.control_plots import matplotlib
import time
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
from itertools import cycle

from helper import ScipyOptimizerInterface

np.random.seed(1234)
tf.set_random_seed(1234)

class XPINN:
    def __init__(self, xb0, yb0, xf1, x1f, y1f, xbf, ybf, n, m, layers1, w_ub1, w_f1):
        self.x_b0 = xb0[:, 0:1]
        self.t_b0 = xb0[:, 1:2]
        self.yb0 = yb0
        self.x_f1 = xf1[:, 0:1]
        self.t_f1 = xf1[:, 1:2]
        self.x1f = x1f[:, 0:1]
        self.t1f = x1f[:, 1:2]
        self.y1f = y1f
        self.xbf = xbf[:, 0:1]
        self.tbf = xbf[:, 1:2]
        self.ybf = ybf
        self.history_MSE = []
        self.MSE_LBFGS_hist1 = []
        self.layers1 = layers1
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.n = n
        self.m = m
        self.uf_x_segments = [[] for _ in range(self.n * self.m)]
        self.uf_t_segments = [[] for _ in range(self.n * self.m)]
        self.uf_y_segments = [[] for _ in range(self.n * self.m)]
        x_point_start = np.min(self.xbf)
        x_point_end = np.max(self.xbf)
        x_segment_length = (x_point_end - x_point_start) / self.n + 0.000001
        t_point_start = np.min(self.tbf)
        t_point_end = np.max(self.tbf)
        t_segment_length = (t_point_end - t_point_start) / self.m + 0.000001
        for idx, (xi, ti, yi) in enumerate(zip(self.xbf, self.tbf, self.ybf)):
            x_segment = int((xi - x_point_start) // x_segment_length)
            t_segment = int((ti - t_point_start) // t_segment_length)
            segment_index = t_segment * self.n + x_segment
            if 0 <= segment_index < self.n * self.m:
                self.uf_x_segments[segment_index].append(xi)
                self.uf_t_segments[segment_index].append(ti)
                self.uf_y_segments[segment_index].append(yi)
        for i in range(self.n * self.m):
            self.uf_x_segments[i] = np.array(self.uf_x_segments[i])
            self.uf_t_segments[i] = np.array(self.uf_t_segments[i])
            self.uf_y_segments[i] = np.array(self.uf_y_segments[i])
        for i in range(self.n * self.m):
            self.uf_x_segments[i] = self.uf_x_segments[i].flatten()[:, None]
            self.uf_t_segments[i] = self.uf_t_segments[i].flatten()[:, None]
            self.uf_y_segments[i] = self.uf_y_segments[i].flatten()[:, None]
        self.X_f_segments = [[] for _ in range(self.n * self.m)]
        self.T_f_segments = [[] for _ in range(self.n * self.m)]
        self.Uf_y_segments = [[] for _ in range(self.n * self.m)]
        idx_list = []
        for i in range(self.n * self.m):
            idx = np.random.choice(self.uf_y_segments[i].shape[0], N_uf, replace=False)
            self.X_f_segments[i] = self.uf_x_segments[i][idx, :]
            self.T_f_segments[i] = self.uf_t_segments[i][idx, :]
            self.Uf_y_segments[i] = self.uf_y_segments[i][idx, :]
            idx_list.append(idx)
        self.x_f = [[] for _ in range(self.n * self.m)]
        self.x_f_tf = [[] for _ in range(self.n * self.m)]
        self.t_f = [[] for _ in range(self.n * self.m)]
        self.t_f_tf = [[] for _ in range(self.n * self.m)]
        self.f1_preds = []
        for i in range(self.n * self.m):
            self.x_f[i] = self.X_f_segments[i]
            self.x_f_tf[i] = tf.placeholder(tf.float64, shape=[None, self.x_f[i].shape[1]])
            self.t_f[i] = self.T_f_segments[i]
            self.t_f_tf[i] = tf.placeholder(tf.float64, shape=[None, self.t_f[i].shape[1]])
            f1_pred = self.net_f(self.x_f_tf[i], self.t_f_tf[i])
            self.f1_preds.append(f1_pred)
        self.x_b0_tf = tf.placeholder(tf.float64, shape=[None, self.x_b0.shape[1]])
        self.t_b0_tf = tf.placeholder(tf.float64, shape=[None, self.t_b0.shape[1]])
        self.x_f1_tf = tf.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.t_f1_tf = tf.placeholder(tf.float64, shape=[None, self.t_f1.shape[1]])
        self.x1f_tf = tf.placeholder(tf.float64, shape=[None, self.x1f.shape[1]])
        self.t1f_tf = tf.placeholder(tf.float64, shape=[None, self.t1f.shape[1]])
        self.ub1_pred = self.net_u1(self.x_b0_tf, self.t_b0_tf)
        self.f_pred1 = self.net_f(self.x_f1_tf, self.t_f1_tf)
        self.uf1_pred = self.net_u1(self.x_f1_tf, self.t_f1_tf)
        self.w_ub1_hist1 = w_ub1 if w_ub1_hist1 is not None else np.array([0.0])
        self.w_f1_hist1 = w_f1 if w_f1_hist1 is not None else np.array([[0.0]] * (self.n * self.m))
        a, b, c = 1, 1, 2
        self.w_ub1 = tf.Variable(a, dtype=tf.float64) ** b
        self.sw_ub1 = tf.log(self.w_ub1 ** c)
        self.w_ub12 = 0.5 * tf.exp(-self.sw_ub1)
        d = 1
        self.w_uf = {}
        self.sw_f = {}
        self.w_f = {}
        self.sum1 = self.sw_ub1 ** d
        for i in range(0, self.n * self.m):
            w_uf_name = f'w_uf{i}'
            sw_f_name = f'sw_f{i}'
            w_f_name = f"w_f{i}2"
            self.w_uf[w_uf_name] = tf.Variable(a, dtype=tf.float64) ** b
            self.sw_f[sw_f_name] = tf.log(self.w_uf[w_uf_name] ** c)
            self.w_f[w_f_name] = 0.5 * tf.exp(-self.sw_f[sw_f_name])
            self.sum1 += self.sw_f[sw_f_name] ** d


        self.loss1 = tf.reduce_mean(tf.square(self.yb0 - self.ub1_pred)) + tf.reduce_mean(tf.square(self.f_pred1))
        self.loss2 = self.w_ub12/N_ub * tf.reduce_mean(tf.square(self.yb0 - self.ub1_pred)) + self.sum1 ** 2 / (N_uf * (self.n * self.m + 1))
        for i in range(self.n * self.m):
            w_f_name = f"w_f{i}2"
            self.loss2 += self.w_f[w_f_name]/N_uf * tf.reduce_mean(tf.square(self.f1_preds[i]))
        self.loss3 = self.w_ub1_hist1[-1]/N_ub * tf.reduce_mean(tf.square(self.yb0 - self.ub1_pred))
        for i in range(self.n * self.m):
            self.loss3 += self.w_f1_hist1[i][-1]/N_uf * tf.reduce_mean(tf.square(self.f1_preds[i]))


        self.optimizer_Adam1 = tf.train.AdamOptimizer(0.005)
        self.train_op_Adam1 = self.optimizer_Adam1.minimize(self.loss1)
        self.optimizer_Adam2 = tf.train.AdamOptimizer(0.005)
        self.train_op_Adam2 = self.optimizer_Adam2.minimize(self.loss2)
        self.optimizer_LBFGS = ScipyOptimizerInterface(self.loss3, method='L-BFGS-B',
                                                                options={'maxiter': 30000,
                                                                           'maxfun': 30000,
                                                                           'maxcor': 50,
                                                                           'maxls': 20,
                                                                           'ftol': 1e-50,
                                                                           'gtol': 1e-50})
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver(max_to_keep=1)
    def initialize_NN(self, layers):
        weights = []
        biases = []
        # A = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            w = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(w)
            biases.append(b)
        return weights, biases
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)
    def neural_net_sin(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, w), b))
        w = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, w), b)
        return Y
    def net_u1(self, x, y):
        u = self.neural_net_sin(tf.concat([x, y], 1), self.weights1, self.biases1)
        return u
    def f(self, x, y):
        pi = tf.constant(np.pi, dtype=tf.float64)
        f = (omega ** 2 * (1 - x ** 2) * (1 - y ** 2) * (tf.exp(x) - 1) * tf.sin(omega * y) + 4 * omega * y * (
                1 - x ** 2) * (tf.exp(x) - 1) * tf.cos(omega * y)
             + 4 * x * (1 - y ** 2) * tf.exp(x) * tf.sin(omega * y) - (1 - x ** 2) * (1 - y ** 2) * tf.exp(x) * tf.sin(
                    omega * y) -
             (2 * x ** 2 - 2) * (tf.exp(x) - 1) * tf.sin(omega * y) - (2 * y ** 2 - 2) * (tf.exp(x) - 1) * tf.sin(
                    omega * y))
        return f
    def net_f(self, x, y):
        # Sub-Net1
        u1 = self.net_u1(x, y)
        u1_x = tf.gradients(u1, x)[0]
        u1_xx = tf.gradients(u1_x, x)[0]
        u1_y = tf.gradients(u1, y)[0]
        u1_yy = tf.gradients(u1_y, y)[0]
        f1 = (u1_xx + u1_yy) + self.f(x, y)
        return f1
    def callback(self, loss):
        self.history_MSE.append(loss)
        step = len(self.history_MSE)
        if step % 250 == 0:
            loss1 = loss
            self.MSE_LBFGS_hist1.append(loss1)
            print('It: %d,' 'loss1: %.3e '% (step, loss1))
        return self.MSE_LBFGS_hist1
    def train(self, nIter, first_adarm, X_star1, X_star2, u_exactb1, u_exactf1):
        tf_dict = {self.x_b0_tf: self.x_b0, self.t_b0_tf: self.t_b0,
                   self.x_f1_tf: self.x_f1, self.t_f1_tf: self.t_f1}
        for i in range(self.n * self.m):
            tf_dict[self.x_f_tf[i]] = self.x_f[i]
            tf_dict[self.t_f_tf[i]] = self.t_f[i]
        MSE_history1 = []
        w_ub1_history1 = []
        w_f1_history1 = [[] for _ in range(self.n * self.m)]
        for it in range(nIter):
            if it <= first_adarm:
                self.sess.run(self.train_op_Adam1, tf_dict)
                if it % 250 == 0:
                    loss1_value = self.sess.run(self.loss1, tf_dict)
                    print('It: %d, Loss1: %.3e' % (it, loss1_value), end='')
                    print()
                    MSE_history1.append(loss1_value)
            else:
                self.sess.run(self.train_op_Adam2, tf_dict)
                if it % 250 == 0:
                    loss1_value = self.sess.run(self.loss2, tf_dict)
                    print('It: %d, Loss1: %.3e' % (it, loss1_value), end='')
                    print(' 0w_ub1: %.3e,' % (w_ub1), end='')
                    for i, w_f in enumerate(w_f_values):
                        print(' 0w_f%d: %.3e,' % (i, w_f), end='')
                    print()
                    MSE_history1.append(loss1_value)
                    w_ub1_history1.append(w_ub1)
                    for i, w_f in enumerate(w_f_values):
                        w_f1_history1[i].append(w_f)

        self.optimizer_LBFGS.minimize(self.sess, feed_dict=tf_dict,
                                      fetches=[self.loss3], loss_callback=self.callback)
        return MSE_history1, w_ub1_history1, w_f1_history1
    def save_model(self):
        model_file = f"./数据+结果/数据/w={omega}_test.ckpt"
        self.saver.save(self.sess, model_file)
    def load_model(self, Max_iter, X_star1, X_star2, u_exactb, u_exact1):
        model_file = f'./数据+结果/数据/w={omega}_test.ckpt'
        self.saver.restore(self.sess, model_file)
        load_b1 = self.sess.run(self.ub1_pred, {self.x_b0_tf: X_star1[:, 0:1], self.t_b0_tf: X_star1[:, 1:2]})
        load_f1 = self.sess.run(self.uf1_pred, {self.x_f1_tf: X_star2[:, 0:1], self.t_f1_tf: X_star2[:, 1:2]})
        return load_b1, load_f1
    def predict(self, X_star1, X_star2):
        u_star1 = self.sess.run(self.ub1_pred, {self.x_b0_tf: X_star1[:, 0:1], self.t_b0_tf: X_star1[:, 1:2]})
        u_star2 = self.sess.run(self.uf1_pred, {self.x_f1_tf: X_star2[:, 0:1], self.t_f1_tf: X_star2[:, 1:2]})
        return u_star1, u_star2

if __name__ == "__main__":
    omega = 25
    N_uf = 320
    N_ub = 400
    N_uf1 = 1024
    n = 5
    m = 1
    load_file = f"./数据+结果/数据/w={omega}_MSE_hist.npz"
    if os.path.exists(load_file):
        data1 = np.load(load_file)
        w_ub1_hist1 = data1['arr_11']
        w_f1_hist1 = data1['arr_12']
    else:
        w_ub1_hist1 = None
        w_f1_hist1 = None
    layers1 = [2, 20, 20, 1]
    data = np.load(f'./数据+结果/数据/w={omega}_shuju.npz')
    ub_x = data['arr_0']
    ub_t = data['arr_1']
    u_b = data['arr_2']
    uf_x = data['arr_3']
    uf_t = data['arr_4']
    u_f = data['arr_5']
    x_total = data['arr_6']
    t_total = data['arr_7']                
    u_total = data['arr_8']
    X_train_b = np.hstack((ub_x.flatten()[:, None], ub_t.flatten()[:, None]))
    X_train_f = np.hstack((uf_x.flatten()[:, None], uf_t.flatten()[:, None]))
    X_total = np.hstack((x_total.flatten()[:, None], t_total.flatten()[:, None]))
    y1_exactb = u_total[0:len(ub_x), ].flatten()[:, None]
    y1_exactf = u_total[len(ub_x):len(ub_x) + len(uf_x), ].flatten()[:, None]
    y1_total = u_total.flatten()[:, None]
    X_star1 = np.hstack((ub_x.flatten()[:, None], ub_t.flatten()[:, None]))
    X_star2 = np.hstack((uf_x.flatten()[:, None], uf_t.flatten()[:, None]))
    idx1 = np.random.choice(y1_exactb.shape[0], N_ub, replace=False)
    idx2 = np.random.choice(y1_exactf.shape[0], N_uf1, replace=False)
    X_train_1b = X_train_b[idx1, :]
    u_exactb1 = y1_exactb[idx1, :]
    X_train_1f = X_train_f[idx2, :]
    model = XPINN(X_train_1b, u_exactb1, X_train_1f, X_train_f, y1_exactf, X_total, y1_total, n, m, layers1, w_ub1_hist1, w_f1_hist1)
    Max_iter = 40001
    Step_first_adam = 10001
    start_time = time.time()
    MSE_hist1, w_ub1_hist1, w_f1_hist1\
        = model.train(Max_iter, Step_first_adam, X_star1, X_star2, y1_exactb, y1_exactf)
    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))
    model.save_model()
    load_b1, load_f1 = model.load_model(Max_iter, X_star1, X_star2, y1_exactb, y1_exactf)
    u_load_y1 = np.hstack((load_b1.flatten(), load_f1.flatten()))
    u_pred1 = load_b1
    u_pred2 = load_f1
    u_pred_u1 = u_load_y1
    u_exact = np.hstack((y1_exactb.flatten(), y1_exactf.flatten()))
    l2_error = 1/len(u_exact) * np.linalg.norm(u_exact-u_pred_u1, 2)
    l2_err = np.linalg.norm(u_exact-u_pred_u1, 2)/np.linalg.norm(u_exact, 2)
    max_err = np.linalg.norm(u_exact-u_pred_u1, np.inf)
    print('平均绝对Error u: %.3e' % (l2_error))
    print('二范数Error u: %.3e' % (l2_err))
    print('无穷范数Error u: %.3e' % (max_err))
