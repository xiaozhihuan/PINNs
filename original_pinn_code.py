"""
PINN Method for the 2D Steady State Incompressible Navier-Stokes Equation
please refer to "problem.md" for problem defenition.
"""


import sys
import matplotlib.pyplot as plt
sys.path.append(".")
#from network import DNN
#from Utils import *
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0

r = 0.01
xc = 0.5
yc = 0.5


ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])


N_b = 800  # 入口和出口边界点数
N_w = 800  # 墙面边界点数
N_s = 800  # 圆柱表面点数
N_c = 10000  # 配点数量
N_r = 10000  # 精化区域的配点数量

#参数定义
#rho = 1800
#c = 900
#K = 0.8
#k = 5e-7
E = 15e6
nu = 0.35
alph = 3e-5
lemda = E * nu/((1+nu)*(1-2*nu))
L = 5
mu = 0.5*E / (1 + nu)

U = 0.0001

deta_T = 20
T_max = 30
T0 = 10

bate = (3*lemda+2*mu)*alph
A = mu / (lemda+2*mu)
B = (lemda+mu) / (lemda+2*mu)
C = (bate * deta_T * L)/(U*(lemda+2*mu))

def getData():
    left_xy = np.random.uniform([x_min, y_min], [x_min, y_max], (N_b, 2))
    right_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_b, 2))
    up_xy = np.random.uniform([x_min, y_max], [x_max, y_max], (N_b, 2))
    down_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_b, 2))
    # 圆柱表面，u=v=0
    theta = np.linspace(0.0, 2 * np.pi, N_s)
    cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)

    # 配点生成（包括圆柱周围的精化点）
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # refine points around cylider
    refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    refine_lb = np.array([xc - 2 * r, yc - 2 * r])

    xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder

    dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)


    # convert to tensor
    left_xy = torch.tensor(left_xy, dtype=torch.float32).to(device)
    right_xy = torch.tensor(right_xy, dtype=torch.float32).to(device)
    up_xy = torch.tensor(up_xy, dtype=torch.float32).to(device)
    down_xy = torch.tensor(down_xy, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    cyl_xy = torch.tensor(cyl_xy, dtype=torch.float32).to(device)

    return left_xy, right_xy, up_xy, down_xy, xy_col,  cyl_xy


left_xy, right_xy, up_xy, down_xy, xy_col, cyl_xy = getData()

import torch
import torch.nn as nn
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = (
    False  # This is for Nvidia Ampere GPU Architechture
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)


class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        for layer in self.net:
            out = layer(out)
        return out


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


#定义网络
class PINN:
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=3, n_layer=4, n_node=50, ub=ub, lb=lb).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=2,
            max_eval=2,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.losses = {"bc": [], "outlet": [], "pde": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = out[:, 0:1] * (x - 1) * x  # 左右边界u=0
        v = out[:, 1:2] * (y - 1) * y  # 上下边界v=0
        # 温度场分层约束
        d_cyl = ((x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.05 ** 2)
        T = out[:, 2:3]
        # 外部边界条件 (T=0.0)
        T = T * (x - 1) * x * (y - 1) * y

        return u, v, T

    def compute_von_mises(self, xy, mu, lam):
        xy = xy.clone().detach().requires_grad_(True)
        u, v, T = self.predict(xy)

        grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]

        eps_xx = u_x*(U/L)
        eps_yy = v_y*(U/L)
        eps_xy = 0.5 * (u_y + v_x)*(U/L)

        sigma_xx = 2 * mu * eps_xx + lam * (eps_xx + eps_yy)
        sigma_yy = 2 * mu * eps_yy + lam * (eps_xx + eps_yy)
        tau_xy = 2 * mu * eps_xy

        sigma_vm = torch.sqrt(sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3 * tau_xy ** 2)
        return sigma_vm

    def right_loss(self, xyt):
        u, v, T = self.predict(xyt)[0:3]
        mse_right = (torch.mean(torch.square(u)) + torch.mean(torch.square(T)))
        return mse_right

    def left_loss(self, xyt):
        u, v, T = self.predict(xyt)[0:3]
        mse_left = (torch.mean(torch.square(u)) + torch.mean(torch.square(T)))
        return mse_left

    def up_loss(self, xyt):
        u, v, T = self.predict(xyt)[0:3]
        mse_up = (torch.mean(torch.square(v)) + torch.mean(torch.square(T)))
        return mse_up

    def down_loss(self, xyt):
        u, v, T = self.predict(xyt)[0:3]
        mse_down = (torch.mean(torch.square(v)) + torch.mean(torch.square(T)))
        return mse_down

    def cycle_loss(self, xyt):
        u, v, T = self.predict(xyt)[0:3]
        T_cyl = (T_max - T0) / deta_T  # = 1.0
        mse_cycle = (torch.mean(torch.square(T-T_cyl)) + torch.mean(torch.square(u)) + torch.mean(torch.square(v)))
        return mse_cycle


    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, T = self.predict(xy)
        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        T_out = grad(T.sum(), xy, create_graph=True)[0]
        T_x = T_out[:, 0:1]
        T_y = T_out[:, 1:2]
        T_xx = grad(T_out[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        T_yy = grad(T_out[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        u_xx = grad(u_out[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_out[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        v_xx = grad(v_out[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_out[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        u_xy = grad(u_out[:, 0:1].sum(), xy, create_graph=True)[0][:, 1:2]
        v_xy = grad(v_out[:, 0:1].sum(), xy, create_graph=True)[0][:, 1:2]
        f0 = u_xx + A * u_yy + B * v_xy - C * T_x
        f1 = v_yy + A * v_xx + B * u_xy - C * T_y
        f2 = T_xx + T_yy
        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_pde = mse_f0 + mse_f1 + mse_f2

        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_right = self.right_loss(right_xy)
        mse_left = self.left_loss(left_xy)
        mse_up = self.up_loss(up_xy)
        mse_down = self.down_loss(down_xy)
        mse_cycle = self.cycle_loss(cyl_xy)
        mse_pde = self.pde_loss(xy_col)
        mse_bc = mse_right + mse_left + mse_up + mse_down + mse_cycle*110
        loss = mse_bc + mse_pde

        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e} ",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss


if __name__ == "__main__":
    pinn = PINN()
    for i in range(1600):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "C:/Users/86188/Desktop/weight.pt")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设已定义的 PINN 类并加载权重
pinn = PINN()
pinn.net.load_state_dict(torch.load("C:/Users/86188/Desktop/weight.pt"))

# 定义区域范围
x_min, x_max = 0, 1.0  # 根据你的实际范围调整
y_min, y_max = 0, 1.0
xc, yc = 0.5, 0.5  # 圆柱体的圆心
r = 0.05  # 圆柱体的半径

# 生成网格点
x = np.arange(x_min, x_max, 0.004)
y = np.arange(y_min, y_max, 0.004)
X, Y = np.meshgrid(x, y)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

# 计算到圆柱体的距离，并创建遮罩
dst_from_cyl = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
cyl_mask = dst_from_cyl > r  # 圆柱体区域将被屏蔽

# 准备输入坐标
xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float32).to(device)

# >>> 添加材料参数（假设常量，可替换为实际值） <<<
mu = torch.tensor(mu, dtype=torch.float32).to(device)
lam = torch.tensor(lemda, dtype=torch.float32).to(device)

# 通过 PINN 预测 u, v, T
with torch.no_grad():
    u, v, T = pinn.predict(xy)
    u = u.cpu().numpy()*U
    v = v.cpu().numpy()*U
    T = T.cpu().numpy()*deta_T+T0

# >>> 计算 von Mises 应力 <<<
xy.requires_grad_(True)
von_mises = pinn.compute_von_mises(xy, mu, lam).detach().cpu().numpy()

# >>> 屏蔽圆柱内部 <<<
u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)
v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)
T = np.where(cyl_mask, T, np.nan).reshape(Y.shape)
disp = np.sqrt(u**2 + v**2)
von_mises = np.where(cyl_mask, von_mises, np.nan).reshape(Y.shape)
import pandas as pd
# 保存数据到Excel
results = pd.DataFrame({
    'x': X.ravel(),
    'y': Y.ravel(),
    'u': u.ravel(),
    'v': v.ravel(),
    'T': T.ravel(),
    'disp': disp.ravel(),
    'von_mises': von_mises.ravel(),
    'in_cylinder': ~cyl_mask.ravel()
})
results = results[results['in_cylinder'] == False].drop(columns=['in_cylinder'])
output_path = "C:/Users/86188/Desktop/pinn_results.xlsx"
results.to_excel(output_path, index=False)
print(f"结果已保存至: {output_path}")

# >>> 添加 von Mises 应力到绘图 <<<
fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

data = (u, v, T, disp, von_mises)
labels = ["$u(x,y)$", "$v(x,y)$", "$T(x,y)$", "$d(x,y)$", r"$\sigma_{vm}(x,y)$"]

for i in range(5):
    ax = axes[i]
    im = ax.imshow(
        data[i], cmap="jet", extent=[x_min, x_max, y_min, y_max], origin="lower"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="3%")
    fig.colorbar(im, cax=cax, label=labels[i])
    ax.set_title(labels[i])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")

fig.tight_layout()
plt.show()
