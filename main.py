import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from IEEE118 import IEEE118

def get_sigma_vector(sigma):
    
    sigma_array = np.array(sigma, dtype=float)
    
    return sigma_array.copy()

def branch_model_gbytheta(case, branch_idx, Vj, Vk, theta_jk, g_s, b_s, b_c):
    """
    构造目标支路的六维局部测量：
        z = [Pjk, Pkj, Qjk, Qkj, Vj, Vk]

    状态：
        x = [Vj, Vk, theta_jk]
    """
    Vm = np.ones(case.n_bus, dtype=float)
    Va = np.zeros(case.n_bus, dtype=float)

    j = case.from_bus[branch_idx]
    k = case.to_bus[branch_idx]

    Vm[j] = Vj
    Vm[k] = Vk
    Va[j] = theta_jk
    Va[k] = 0.0

    z_hat = case.h(branch_idx, Vm, Va, g_s=g_s, b_s=b_s, b_c=b_c)
    return z_hat


def build_measurement_gbytheta(case, branch_idx, Vm, Va, sigma, g_s, b_s, b_c, seed=None):
    """
    生成一组带噪声的局部支路测量。
    """
    z_true = case.h(branch_idx, Vm, Va, g_s=g_s, b_s=b_s, b_c=b_c)

    sigma_vector = get_sigma_vector(sigma)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_vector, size=6)

    z = z_true + noise
    return z, z_true


def estimate_branch_state_gbytheta(case, branch_idx, z, sigma, g_s, b_s, b_c, x0=None):
    """
    已知支路参数时，估计局部支路状态：
        x = [Vj, Vk, theta_jk]
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 0.0], dtype=float)

    sigma_vector = get_sigma_vector(sigma)

    def residual(x):
        z_hat = branch_model_gbytheta(
            case,
            branch_idx,
            x[0],
            x[1],
            x[2],
            g_s,
            b_s,
            b_c,
        )
        return (z_hat - z) / sigma_vector

    result = least_squares(
        residual,
        x0,
        bounds=([0.8, 0.8, -1.0], [1.2, 1.2, 1.0]),
        max_nfev=300,
    )

    x_est = result.x

    z_hat = branch_model_gbytheta(
        case,
        branch_idx,
        x_est[0],
        x_est[1],
        x_est[2],
        g_s,
        b_s,
        b_c,
    )

    residual_value = z - z_hat
    J = np.sum((residual_value / sigma_vector) ** 2)

    return x_est, z_hat, residual_value, J


def chi_square_branch_detector(residual, sigma, confidence=0.95):
    """
    局部支路估计器对应的卡方检测器。

    测量维数 m = 6
    状态维数 n = 3
    自由度 = m - n = 3
    """
    sigma_vector = get_sigma_vector(sigma)

    dof = len(residual) - 3
    J = np.sum((residual / sigma_vector) ** 2)
    threshold = chi2.ppf(confidence, dof)
    detected = J > threshold

    return detected, J, threshold


def apply_mtd_gbytheta(case, branch_idx, g_ratio, b_ratio, y_ratio):
    """
    MTD：同时修改目标支路的 g、b、y。
    """
    g_s_new = case.g_s.copy()
    b_s_new = case.b_s.copy()
    b_c_new = case.b_c.copy()

    g_s_new[branch_idx] = g_s_new[branch_idx] * g_ratio
    b_s_new[branch_idx] = b_s_new[branch_idx] * b_ratio
    b_c_new[branch_idx] = b_c_new[branch_idx] * y_ratio

    return g_s_new, b_s_new, b_c_new


def build_local_parameter(case, branch_idx, g_value, b_value, b_c_value):
    """
    保留全网参数，只替换目标支路参数。
    """
    g_s = case.g_s.copy()
    b_s = case.b_s.copy()
    b_c = case.b_c.copy()

    g_s[branch_idx] = g_value
    b_s[branch_idx] = b_value
    b_c[branch_idx] = b_c_value

    return g_s, b_s, b_c


def build_joint_initial_value():
    """
    共享状态版本的联合估计初值。

    待估变量：
        x_array = [g, b, y, Vj, Vk, theta]
    """

    x0 = [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]

    return np.array(x0, dtype=float)


def build_joint_bounds():
    """
    共享状态版本的联合估计上下界。

    x_array = [g, b, y, Vj, Vk, theta]

    """
    lower = [-20.0, -20.0, -5.0, 0.8, 0.8, -1.0]
    upper = [20.0, 20.0, 5.0, 1.2, 1.2, 1.0]

    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def unpack_joint_vector(x_array):
    """
    拆分联合估计变量。

    输入：
        x_array = [g, b, y, Vj, Vk, theta]

    返回：
        g_value, b_value, y_value, state
    """
    g_value = float(x_array[0])
    b_value = float(x_array[1])
    y_value = float(x_array[2])

    state = np.array(x_array[3:6], dtype=float)

    return g_value, b_value, y_value, state


def joint_parameter_state_residual(case, branch_idx, z_list, sigma, x_array):
    """
    共享状态版本的联合估计残差。

    样本：
        p = [g, b, y]
        x = [Vj, Vk, theta]
    """
    sigma_vector = get_sigma_vector(sigma)

    g_value, b_value, y_value, state = unpack_joint_vector(x_array)

    g_s_use, b_s_use, b_c_use = build_local_parameter(
        case,
        branch_idx,
        g_value,
        b_value,
        y_value,
    )

    residual_value = []

    for z in z_list:
        z_hat = branch_model_gbytheta(
            case,
            branch_idx,
            state[0],
            state[1],
            state[2],
            g_s_use,
            b_s_use,
            b_c_use,
        )

        residual_value.extend((z_hat - z) / sigma_vector)

    return np.array(residual_value, dtype=float)


def estimate_branch_parameter_blind_gbytheta(case, branch_idx, z_list, sigma):
    """
    PEF-FDI 的参数估计步骤。

    联合估计：
        [g, b, y, Vj, Vk, theta]

    返回：
        g_est, b_est, y_est, theta_est, state_est
    """
    x0 = build_joint_initial_value()
    lower, upper = build_joint_bounds()

    result = least_squares(
        lambda x_array: joint_parameter_state_residual(
            case,
            branch_idx,
            z_list,
            sigma,
            x_array,
        ),
        x0,
        bounds=(lower, upper),
        max_nfev=400,
    )

    g_est, b_est, y_est, state_est = unpack_joint_vector(result.x)
    theta_est = float(state_est[2])

    return g_est, b_est, y_est, theta_est, state_est


def build_attack_from_estimated_state_gbytheta(
    case,
    branch_idx,
    z,
    g_s_use,
    b_s_use,
    b_c_use,
    state_est,
    delta_theta,
    scale_vk,
):
    """
    构造攻击：

    a = h(p_hat, x_hat + c) - z
    z_a = h(p_hat, x_hat + c)

    """
    x_att = state_est.copy()

    x_att[1] = x_att[1] * (1.0 + scale_vk)
    x_att[2] = x_att[2] - delta_theta

    z_att = branch_model_gbytheta(
        case,
        branch_idx,
        x_att[0],
        x_att[1],
        x_att[2],
        g_s_use,
        b_s_use,
        b_c_use,
    )

    a = z_att - z
    z_a = z_att

    return z_a, a, state_est, x_att


def main(
    from_id=12,
    to_id=117,
    sigma=0.01,
    g_ratio=1.10,
    b_ratio=0.78,
    y_ratio=1.10,
    delta_theta=0.15,
    scale_vk=0.15,
    sample_number=50,
):
    """
    运行一个局部支路 PEF-FDI + MTD demo。

    输出：
        baseline: MTD 后正常测量
        old:      用旧参数构造攻击
        blind:    先估计新参数，再构造 PEF-FDI 攻击
    """

    case = IEEE118()
    Vm, Va = case.Vm.copy(), case.Va.copy()
    branch_idx = case.find_branch(from_id, to_id)

    g_s_old = case.g_s.copy()
    b_s_old = case.b_s.copy()
    b_c_old = case.b_c.copy()

    g_s_new, b_s_new, b_c_new = apply_mtd_gbytheta(
        case,
        branch_idx,
        g_ratio,
        b_ratio,
        y_ratio,
    )

    # 当前时刻的合法测量，后面攻击就是针对这个 z 构造
    z, z_true = build_measurement_gbytheta(
        case,
        branch_idx,
        Vm,
        Va,
        sigma,
        g_s_new,
        b_s_new,
        b_c_new,
    )

    # [1] 正常 MTD 后测量检测
    _, _, residual_def, _ = estimate_branch_state_gbytheta(
        case,
        branch_idx,
        z,
        sigma,
        g_s_new,
        b_s_new,
        b_c_new,
    )

    detected_def, J0, threshold = chi_square_branch_detector(residual_def, sigma)

    # [2] 用旧参数构造攻击：攻击者不知道 MTD 已经发生，仍然使用旧参数估计状态并构造攻击。
    # 先用旧参数估计攻击者眼里的 state，再构造 old attack。
    x_est_old, _, _, _ = estimate_branch_state_gbytheta(
        case,
        branch_idx,
        z,
        sigma,
        g_s_old,
        b_s_old,
        b_c_old,
    )

    z_old, a_old, x_est_old, x_att_old = build_attack_from_estimated_state_gbytheta(
        case,
        branch_idx,
        z,
        g_s_old,
        b_s_old,
        b_c_old,
        x_est_old,
        delta_theta,
        scale_vk,
    )

    _, _, residual_old, _ = estimate_branch_state_gbytheta(
        case,
        branch_idx,
        z_old,
        sigma,
        g_s_new,
        b_s_new,
        b_c_new,
    )

    detected_old, J1, _ = chi_square_branch_detector(residual_old, sigma)

    # [3] PEF-FDI：用当前 z 加上同一 MTD 周期内的其他样本估计参数
    z_list = [z.copy()]

    for _ in range(sample_number - 1):
        z_sample, _ = build_measurement_gbytheta(
            case,
            branch_idx,
            Vm,
            Va,
            sigma,
            g_s_new,
            b_s_new,
            b_c_new,
        )
        z_list.append(z_sample)

    g_est_blind, b_est_blind, y_est_blind, theta_est_blind, state_est_blind = (
        estimate_branch_parameter_blind_gbytheta(
            case,
            branch_idx,
            z_list,
            sigma,
        )
    )

    g_s_blind, b_s_blind, b_c_blind = build_local_parameter(
        case,
        branch_idx,
        g_est_blind,
        b_est_blind,
        y_est_blind,
    )

    z_blind, a_blind, x_est_blind, x_att_blind = build_attack_from_estimated_state_gbytheta(
        case,
        branch_idx,
        z,
        g_s_blind,
        b_s_blind,
        b_c_blind,
        state_est_blind,
        delta_theta,
        scale_vk,
    )

    _, _, residual_blind, _ = estimate_branch_state_gbytheta(
        case,
        branch_idx,
        z_blind,
        sigma,
        g_s_new,
        b_s_new,
        b_c_new,
    )

    detected_blind, J2, _ = chi_square_branch_detector(residual_blind, sigma)

    result = {
        "branch_idx": branch_idx,
        "z": z,
        "z_true": z_true,

        "baseline_detected": detected_def,
        "baseline_J": J0,

        "old_detected": detected_old,
        "old_J": J1,

        "blind_detected": detected_blind,
        "blind_J": J2,

        "threshold": threshold,

        "g_s_old": g_s_old,
        "b_s_old": b_s_old,
        "b_c_old": b_c_old,

        "g_s_new": g_s_new,
        "b_s_new": b_s_new,
        "b_c_new": b_c_new,

        "g_est_blind": g_est_blind,
        "b_est_blind": b_est_blind,
        "y_est_blind": y_est_blind,
        "theta_est_blind": theta_est_blind,
        "state_est_blind": state_est_blind,

        "x_est_old": x_est_old,
        "x_att_old": x_att_old,

        "x_est_blind": x_est_blind,
        "x_att_blind": x_att_blind,

        "a_old": a_old,
        "a_blind": a_blind,
    }

    return result