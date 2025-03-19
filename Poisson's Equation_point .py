import numpy as np

x_start = 0
x_end = 1.0
t_start = 0
t_end = 1
num = 50
numb = 100
omega = 25
uf_x = []; uf_t = []; u_f = []
ub_x = []; ub_t = []; u_b = []
for x in np.linspace(x_start, x_end, numb + 1, endpoint=True):
    for t in np.linspace(t_start, t_end, numb + 1, endpoint=True):
        if x == 0 or x == 1.0:
            u = np.sin(omega*t) * (np.exp(x)-1) * (1-x**2) * (1-t**2)
            ub_x.append(x)
            ub_t.append(t)
            u_b.append(u)
        elif t == 0 or t == 1.0:
            u = np.sin(omega*t) * (np.exp(x)-1) * (1-x**2) * (1-t**2)
            ub_x.append(x)
            ub_t.append(t)
            u_b.append(u)
for x in np.linspace(x_start, x_end, num + 1, endpoint=True):
    for t in np.linspace(t_start, t_end, num + 1, endpoint=True):
        if x != 0 and x != 1 and t != 0 and t != 1:
            u = np.sin(omega*t) * (np.exp(x)-1) * (1-x**2) * (1-t**2)
            uf_x.append(x)
            uf_t.append(t)
            u_f.append(u)
ub_x = np.array(ub_x); ub_t = np.array(ub_t); u_b = np.array(u_b)
uf_x = np.array(uf_x); uf_t = np.array(uf_t); u_f = np.array(u_f)
X, T, U = np.hstack((ub_x, uf_x)),np.hstack((ub_t, uf_t)), np.hstack((u_b, u_f))
np.savez(f'./数据+结果/数据/w={omega}_shuju.npz',
         ub_x, ub_t, u_b, uf_x, uf_t, u_f, X, T, U)
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