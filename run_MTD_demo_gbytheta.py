import numpy as np

from MTD_system_gbytheta import run_mtd_demo_gbytheta

SIGMA = np.full(6, 0.01, dtype=float)
DEMO_CONFIG = {
    "from_id": 12,
    "to_id": 117,
    "sigma": SIGMA,
    "g_ratio": 1.10,
    "b_ratio": 0.80,
    "y_ratio": 1.20,
    "delta_theta": 0.08,
    "scale_vk": 0.15,
    "sample_number": 1,
}


def safe_relative_error(estimate, truth):
    denominator = max(abs(truth), 1e-12)
    return abs(estimate - truth) / denominator


def print_detection_result(title, j_value, detected):
    print(title)
    print("J =", j_value)
    print("detected =", detected)
    print("")


def main():
    result = run_mtd_demo_gbytheta(**DEMO_CONFIG)

    print("========== MTD Demo (g, b, y) ==========")
    print("target branch =", f"({DEMO_CONFIG['from_id']}, {DEMO_CONFIG['to_id']})")
    print("branch_idx =", result["branch_idx"])
    print("")
    print("threshold =", result["threshold"])
    print("")
    print_detection_result(
        "[1] normal measurement after MTD",
        result["baseline_J"],
        result["baseline_detected"],
    )
    print_detection_result(
        "[2] attack with old parameters",
        result["old_J"],
        result["old_detected"],
    )
    print_detection_result(
        "[3] blind estimate-first attack",
        result["blind_J"],
        result["blind_detected"],
    )
    print("g_est_blind =", result["g_est_blind"])
    print("b_est_blind =", result["b_est_blind"])
    print("y_est_blind =", result["y_est_blind"])
    print("theta_est_blind =", result["theta_est_blind"])
    

    branch_idx = result["branch_idx"]

    g_true = result["g_s_new"][branch_idx]
    b_true = result["b_s_new"][branch_idx]
    y_true = result["b_c_new"][branch_idx]

    g_est = result["g_est_blind"]
    b_est = result["b_est_blind"]
    y_est = result["y_est_blind"]

    print("true g,b,y =", g_true, b_true, y_true)
    print("est  g,b,y =", g_est, b_est, y_est)

    print("rel err g =", safe_relative_error(g_est, g_true))
    print("rel err b =", safe_relative_error(b_est, b_true))
    print("rel err y =", safe_relative_error(y_est, y_true))

if __name__ == "__main__":
    main()
    
