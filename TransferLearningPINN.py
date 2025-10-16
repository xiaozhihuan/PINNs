import torch
from original_pinn_code import PINN  # 导入原始PINN类
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransferLearningPINN:
    def __init__(self, original_weight_path, new_params):
        """
        初始化迁移学习PINN

        参数:
            original_weight_path: 预训练权重文件路径
            new_params: 新参数的字典，包含:
                {
                    'T0': 新的初始温度,
                    'T_max': 新的最高温度,
                    'E': 新的弹性模量,
                    'nu': 新的泊松比,
                    'U': 新的特征速度,
                    'deta_T': 新的温度变化量,
                    'L': 新的特征长度
                }
        """
        # 初始化PINN模型
        self.pinn = PINN()

        # 加载预训练权重
        self.pinn.net.load_state_dict(torch.load(original_weight_path))

        # 更新物理参数
        self.update_parameters(new_params)

        # 重置优化器
        self.reset_optimizers()

    def update_parameters(self, params):
        """更新物理参数"""
        # 温度相关参数
        E = 15e6
        nu = 0.35
        alph = 3e-5
        L = 5
        mu = 0.5 * E / (1 + nu)

        U = 0.0001

        deta_T = 20
        T_max = 30
        T0 = 10
        self.T0 = params.get('T0', T0)  # 默认使用原值
        self.T_max = params.get('T_max', T_max)
        self.deta_T = params.get('deta_T', deta_T)

        # 材料参数
        self.E = params.get('E', E)
        self.nu = params.get('nu', nu)
        self.alph = params.get('alph', alph)
        self.U = params.get('U', U)
        self.L = params.get('L', L)

        # 重新计算派生参数
        self.lemda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = 0.5 * self.E / (1 + self.nu)
        self.bate = (3 * self.lemda + 2 * self.mu) * self.alph
        self.A = self.mu / (self.lemda + 2 * self.mu)
        self.B = (self.lemda + self.mu) / (self.lemda + 2 * self.mu)
        self.C = (self.bate * self.deta_T * self.L) / (self.U * (self.lemda + 2 * self.mu))

    def reset_optimizers(self):
        """重置优化器，保持网络参数但重新初始化优化器状态"""
        self.pinn.lbfgs = torch.optim.LBFGS(
            self.pinn.net.parameters(),
            lr=1,
            max_iter=500,
            max_eval=500,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.pinn.adam = torch.optim.Adam(self.pinn.net.parameters(), lr=1e-3)
        self.pinn.iter = 0

    def train(self, epochs_adam=500):
        """
        训练迁移学习模型

        参数:
            epochs_adam: Adam优化器迭代次数
            epochs_lbfgs: LBFGS优化器迭代次数
        """
        # 第一阶段: Adam优化
        for i in range(epochs_adam):
            self.pinn.closure()
            self.pinn.adam.step()

        # 第二阶段: LBFGS优化
        self.pinn.lbfgs.step(self.pinn.closure)

        # 保存迁移学习后的权重
        torch.save(self.pinn.net.state_dict(), "C:/Users/86188/Desktop/weight_transfer.pt")

    def predict(self, xy):
        """预测结果"""
        return self.pinn.predict(xy)

    def compute_von_mises(self, xy):
        """计算Von Mises应力"""
        mu_tensor = torch.tensor(self.mu, dtype=torch.float32).to(device)
        lam_tensor = torch.tensor(self.lemda, dtype=torch.float32).to(device)
        return self.pinn.compute_von_mises(xy, mu_tensor, lam_tensor)


if __name__ == "__main__":
    # 定义新的物理参数
    new_params = {
        'T0': 10,  # 新的初始温度
        'T_max': 30,  # 新的最高温度
        'E': 15e6,  # 新的弹性模量
        'nu': 0.3,  # 新的泊松比
        'U': 0.0001,  # 新的特征速度
        'deta_T': 20,  # 新的温度变化量
        'L': 5  # 新的特征长度
    }

    # 创建迁移学习模型
    transfer_model = TransferLearningPINN("C:/Users/86188/Desktop/weight.pt", new_params)

    # 训练模型 (比原始训练更少的epochs)
    transfer_model.train(epochs_adam=600)


import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
pinn = PINN()
pinn.net.load_state_dict(torch.load("C:/Users/86188/Desktop/weight_transfer.pt"))

# 定义区域参数
x_min, x_max = 0, 1.0
y_min, y_max = 0, 1.0
xc, yc = 0.5, 0.5
r = 0.05

# 生成网格
x = np.linspace(x_min, x_max, 250)
y = np.linspace(y_min, y_max, 250)
X, Y = np.meshgrid(x, y)
xy = np.hstack([X.reshape(-1,1), Y.reshape(-1,1)])

# 创建圆柱遮罩
dst = np.sqrt((xy[:,0]-xc)**2 + (xy[:,1]-yc)**2)
cyl_mask = dst > r

# 转换为tensor
xy_tensor = torch.tensor(xy, dtype=torch.float32, device=device)
E = new_params['E']          # 使用迁移学习时定义的参数
nu = new_params['nu']
U = new_params['U']
deta_T = new_params['deta_T']
T0 = new_params['T0']
lemda = E * nu/((1+nu)*(1-2*nu))

mu = 0.5*E / (1 + nu)

# 材料参数
mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device)
lam_tensor = torch.tensor(lemda, dtype=torch.float32, device=device)

# 预测场变量
with torch.no_grad():
    u, v, T = pinn.predict(xy_tensor)
    u = (u.cpu().numpy() * U).reshape(X.shape)
    v = (v.cpu().numpy() * U).reshape(X.shape)
    T = (T.cpu().numpy() * deta_T + T0).reshape(X.shape)

# 计算Von Mises应力
xy_tensor.requires_grad_(True)
with torch.enable_grad():
    vm = pinn.compute_von_mises(xy_tensor, mu_tensor, lam_tensor)
    von_mises = vm.detach().cpu().numpy().reshape(X.shape)

# 应用圆柱遮罩
u[~cyl_mask.reshape(X.shape)] = np.nan
v[~cyl_mask.reshape(X.shape)] = np.nan
T[~cyl_mask.reshape(X.shape)] = np.nan
disp = np.sqrt(u**2 + v**2)
von_mises[~cyl_mask.reshape(X.shape)] = np.nan

# 保存结果
results = pd.DataFrame({
    'x': xy[:,0],
    'y': xy[:,1],
    'u': u.ravel(),
    'v': v.ravel(),
    'T': T.ravel(),
    'disp': disp.ravel(),
    'von_mises': von_mises.ravel(),
    'in_cylinder': cyl_mask
})
results = results[results.in_cylinder].drop('in_cylinder', axis=1)
results.to_excel("C:/Users/86188/Desktop/pinn_results1.xlsx", index=False)

# 可视化（修改了LaTeX字符串处理）
fig, axes = plt.subplots(5, 1, figsize=(11, 15))
titles = [
    "Displacement u (x,y)",
    "Displacement v (x,y)",
    "Temperature T (x,y)",
    "Total displacement (x,y)",
    "Von Mises Stress (x,y)"
]

for i, (ax, data, title) in enumerate(zip(axes, [u, v, T, disp, von_mises], titles)):
    im = ax.imshow(data, extent=[x_min, x_max, y_min, y_max],
                  origin='lower', cmap='jet')
    plt.colorbar(im, ax=ax, label=title)
    ax.set_title(title)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("C:/Users/86188/Desktop/pinn_results.png", dpi=300)
plt.show()