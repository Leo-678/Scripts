# NEP errors plot  16/3/2025
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 在无图形界面的服务器上运行时，取消下行注释
matplotlib.use('Agg')

# 设置全局字体和字号
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["font.size"]   = 14

##############################################################################
# 1. 读取 loss.out
##############################################################################
loss_data = np.loadtxt('loss.out')
gen         = loss_data[:, 0]
gen1        = loss_data[:, 1]
gen2        = loss_data[:, 2]
gen3        = loss_data[:, 3]
E_train_all = loss_data[:, 4]
F_train_all = loss_data[:, 5]
V_train_all = loss_data[:, 6]
E_test_all  = loss_data[:, 7]
F_test_all  = loss_data[:, 8]
V_test_all  = loss_data[:, 9]

# 取最后一行的训练/测试 RMSE，并转换单位 (meV)
E_train_final = E_train_all[-1] * 1000
E_test_final  = E_test_all[-1]  * 1000
F_train_final = F_train_all[-1] * 1000
F_test_final  = F_test_all[-1]  * 1000
V_train_final = V_train_all[-1] * 1000
V_test_final  = V_test_all[-1]  * 1000

##############################################################################
# 2. 读取能量数据 (2列: [DFT, NEP])
##############################################################################
energy_train = np.loadtxt('energy_train.out')
energy_test  = np.loadtxt('energy_test.out')

##############################################################################
# 3. 读取力数据 (6列: [DFT_x, DFT_y, DFT_z, NEP_x, NEP_y, NEP_z])
##############################################################################
force_train = np.loadtxt('force_train.out')
force_test  = np.loadtxt('force_test.out')

# 合并 x, y, z 方向数据
dft_force_train = force_train[:, :3].flatten()
nep_force_train = force_train[:, 3:].flatten()
dft_force_test  = force_test[:, :3].flatten()
nep_force_test  = force_test[:, 3:].flatten()

##############################################################################
# 4. 读取应力/virial 数据 (12列: [DFT_xx..DFT_zx, NEP_xx..NEP_zx])
##############################################################################
virial_train = np.loadtxt('virial_train.out')
virial_test  = np.loadtxt('virial_test.out')

# 合并所有 6 个分量
dft_virial_train = virial_train[:, :6].flatten()
nep_virial_train = virial_train[:, 6:].flatten()
dft_virial_test  = virial_test[:, :6].flatten()
nep_virial_test  = virial_test[:, 6:].flatten()

##############################################################################
# 5. 创建 2×2 子图
##############################################################################
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
ax_loss   = axs[0, 0]  # Loss曲线
ax_energy = axs[0, 1]  # 能量散点
ax_force  = axs[1, 0]  # 力散点
ax_virial = axs[1, 1]  # Virial散点

##############################################################################
# 5.1 左上子图：loss.out (对数纵坐标)
##############################################################################
ax_loss.plot(gen, E_train_all, color='tab:blue',  alpha=0.8, label='E_train')
ax_loss.plot(gen, E_test_all,  color='tab:red',   alpha=0.8, label='E_test')
ax_loss.plot(gen, F_train_all, color='tab:green', alpha=0.6, label='F_train')
ax_loss.plot(gen, F_test_all,  color='tab:orange',alpha=0.6, label='F_test')
ax_loss.plot(gen, V_train_all, color='tab:purple',alpha=0.5, label='V_train')
ax_loss.plot(gen, V_test_all,  color='tab:brown', alpha=0.5, label='V_test')
ax_loss.plot(gen, gen1,  color='tab:blue', alpha=0.9, label='L_t')
ax_loss.plot(gen, gen2,  color='tab:red', alpha=0.9, label='L_1')
ax_loss.plot(gen, gen3,  color='tab:green', alpha=0.9, label='L_2')

ax_loss.set_yscale('log')
ax_loss.set_xlabel('Generation')
ax_loss.set_ylabel('RMSE (log scale)')
ax_loss.set_title('Training/Testing RMSE vs. Generation')
ax_loss.legend()
ax_loss.grid(True)

##############################################################################
# 5.2 右上子图：能量散点 (单位 eV/atom)
##############################################################################
ax_energy.scatter(energy_train[:, 0], energy_train[:, 1],
                  color='#1f77b4', alpha=0.4, s=15, label='Train')  # 深蓝
ax_energy.scatter(energy_test[:, 0], energy_test[:, 1],
                  color='#d62728', alpha=0.4, s=15, label='Test')  # 深红

# 对角线 y=x
xy_min = min(energy_train.min(), energy_test.min())
xy_max = max(energy_train.max(), energy_test.max())
ax_energy.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', linewidth=1.2)

ax_energy.set_xlabel('DFT energy (eV/atom)')
ax_energy.set_ylabel('NEP energy (eV/atom)')
ax_energy.set_title('Energy: NEP vs. DFT')
ax_energy.grid(True)

ax_energy.text(0.05, 0.95, f"Train RMSE = {E_train_final:.2f} meV/atom\nTest RMSE  = {E_test_final:.2f} meV/atom",
               transform=ax_energy.transAxes, va='top', ha='left', fontsize=16)
ax_energy.legend()

##############################################################################
# 5.3 左下子图：力散点 (单位 eV/Å)
##############################################################################
ax_force.scatter(dft_force_train, nep_force_train, 
                 color='#1f77b4', alpha=0.4, s=15, label='Train')  
ax_force.scatter(dft_force_test, nep_force_test,  
                 color='#d62728', alpha=0.4, s=15, label='Test')  

xy_min = min(dft_force_train.min(), dft_force_test.min())
xy_max = max(nep_force_train.max(), nep_force_test.max())
ax_force.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', linewidth=1.2)

ax_force.set_xlabel('DFT force (eV/Å)')
ax_force.set_ylabel('NEP force (eV/Å)')
ax_force.set_title('Force: NEP vs. DFT')
ax_force.grid(True)

ax_force.text(0.05, 0.95, f"Train RMSE = {F_train_final:.2f} meV/Å\nTest RMSE  = {F_test_final:.2f} meV/Å",
              transform=ax_force.transAxes, va='top', ha='left', fontsize=16)
ax_force.legend()

##############################################################################
# 5.4 右下子图：Virial 散点 (单位 eV/atom)
##############################################################################
ax_virial.scatter(dft_virial_train, nep_virial_train,
                  color='#1f77b4', alpha=0.4, s=15, label='Train')  
ax_virial.scatter(dft_virial_test, nep_virial_test,
                  color='#d62728', alpha=0.4, s=15, label='Test')  

xy_min = min(dft_virial_train.min(), dft_virial_test.min())
xy_max = max(nep_virial_train.max(), nep_virial_test.max())
ax_virial.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', linewidth=1.2)

ax_virial.set_xlabel('DFT virial (eV/atom)')
ax_virial.set_ylabel('NEP virial (eV/atom)')
ax_virial.set_title('Virial: NEP vs. DFT')
ax_virial.grid(True)

ax_virial.text(0.05, 0.95, f"Train RMSE = {V_train_final:.2f} meV/atom\nTest RMSE  = {V_test_final:.2f} meV/atom",
               transform=ax_virial.transAxes, va='top', ha='left', fontsize=16)
ax_virial.legend()

##############################################################################
# 6. 保存
##############################################################################
plt.tight_layout()
plt.savefig('NEP.png', dpi=300)
