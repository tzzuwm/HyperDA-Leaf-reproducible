import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import uniform_filter1d
from pathlib import Path

np.random.seed(42)

# ============================================================
# EnKF update
# ============================================================
def enkf_update(ensemble, observation, R):
    """Perform EnKF update with perturbed observations."""
    N = ensemble.shape[0]
    obs_perturbed = observation + np.random.normal(0, np.sqrt(R), N)
    P = np.var(ensemble)
    K = P / (P + R)
    updated = ensemble + K * (obs_perturbed - ensemble)
    residual = abs(observation - np.mean(ensemble))
    return updated, K, residual


# ============================================================
# Convert DAP to YYYYDOY
# ============================================================
def dap_to_yrdoy(dap_array, planting_date):
    """Convert DAP to YYYYDOY format."""
    return np.array([
        int((planting_date + timedelta(days=int(dap))).strftime('%Y%j'))
        for dap in dap_array
    ])


# ============================================================
# Enhanced Recursive Correction (HyperDA-Leaf version)
# ============================================================
def recursive_correction_enhanced(
        var_enkf, PLAG, PLAS, PLA, SENLA,
        doy_sim, doy_obs, var_obs,
        kalman_gains, var_min, var_max,
        error_threshold=0.1):
    """
    Enhanced structure-driven recursive correction.
    Includes:
        - structural trend prediction
        - daily linear drift toward next observation
        - monotonicity constraints
        - physical bounds
    """

    T = len(doy_sim)
    corrected = np.zeros(T)
    corrected[0] = var_enkf[0]

    obs_dict = dict(zip(doy_obs, var_obs))

    for t in range(T - 1):

        current_day = doy_sim[t]

        # Determine the next observation day
        future_obs = [d for d in doy_obs if d > current_day]
        if future_obs:
            next_obs_day = future_obs[0]
            obs_next = obs_dict[next_obs_day]
        else:
            next_obs_day = doy_sim[t + 1]
            obs_next = var_enkf[t + 1]

        # Compute interval
        interval_days = (
            datetime.strptime(str(next_obs_day), "%Y%j")
            - datetime.strptime(str(current_day), "%Y%j")
        ).days

        # Structural terms
        delta_struct = (PLAG[t] - PLAS[t]) / 100
        delta_growth = PLA[t] - SENLA[t]

        best_val = corrected[t]
        best_a = 0.0

        # Structural scanning for optimal a
        for a in np.arange(-1.0, 1.0, 0.01):
            pred = corrected[t] + interval_days * a * delta_struct
            err = abs(obs_next - pred)
            if err < error_threshold:
                best_val = corrected[t] + a * delta_struct
                best_a = a
                break

        # Structural update constraint
        next_val = np.clip(best_val, corrected[t] - 1.2, corrected[t] + 1.2)

        # Daily linear drift toward next observation
        next_val += (obs_next - corrected[t]) / max(interval_days, 3)

        # Monotonic constraint (leaf senescence)
        if delta_struct < 0 and delta_growth < 0:
            next_val = min(next_val, corrected[t])

        # Apply physical bounds
        next_val = np.clip(next_val, var_min[t + 1], var_max[t + 1])

        corrected[t + 1] = next_val

    return corrected


# ============================================================
# Daily Logging
# ============================================================
def log_daily_values(doy_list, lai_obs_dict, lai_enkf, lai_corr,
                     fapar_obs_dict, fapar_enkf, fapar_corr):
    """Print daily LAI/FAPAR values for debugging."""
    print("\n>>> Daily LAI/FAPAR Summary:")
    for i, doy in enumerate(doy_list):
        lai_obs = lai_obs_dict.get(doy, None)
        fapar_obs = fapar_obs_dict.get(doy, None)
        print(
            f"[DOY={doy}] ",
            f"LAI_obs={lai_obs:.3f}" if lai_obs is not None else "LAI_obs=None",
            f"LAI_enkf={lai_enkf[i]:.3f}",
            f"LAI_corr={lai_corr[i]:.3f} |",
            f"FAPAR_obs={fapar_obs:.3f}" if fapar_obs is not None else "FAPAR_obs=None",
            f"FAPAR_enkf={fapar_enkf[i]:.3f}",
            f"FAPAR_corr={fapar_corr[i]:.3f}"
        )


# ============================================================
# Main
# ============================================================
def main(input_struct_path, output_txt_path):

    planting_date = datetime(2022, 5, 23)

    print(">>> Reading input files...")
    df_sim = pd.read_excel("E:/DSSATpaper/LAIassimilation/lai_hybrids_bounds22.xlsx", header=None)
    df_obs = pd.read_excel("C:/DSSAT48/Maize/LAI_observations.xlsx", header=None)
    df_fapar = pd.read_excel("E:/DSSATpaper/FAPARassimilation/fapar_hybrids_bounds22.xlsx", header=None)
    df_fapar_obs = pd.read_excel("C:/DSSAT48/Maize/FAPAR_observations.xlsx", header=None)
    df_struct = pd.read_csv(input_struct_path, delim_whitespace=True, header=None)

    # --------------------------
    # Convert to DOY
    # --------------------------
    dap_sim = df_sim.iloc[0, 1:].values.astype(int)
    doy_sim = dap_to_yrdoy(dap_sim, planting_date)

    dap_obs = df_obs.iloc[0, 1:].values.astype(int)
    doy_obs = dap_to_yrdoy(dap_obs, planting_date)

    var_min = df_sim.iloc[2, 1:].values
    var_max = df_sim.iloc[3, 1:].values
    lai_obs = df_obs.iloc[1, 1:].values

    fapar_min = df_fapar.iloc[2, 1:].values
    fapar_max = df_fapar.iloc[3, 1:].values
    fapar_obs = df_fapar_obs.iloc[1, 1:].values

    # ============================================================
    # EnKF for LAI
    # ============================================================
    # Generate ensembles based on genotype upper and lower bounds.
    N = 500
    T = len(dap_sim)

    LAI_ens = np.array([
        np.random.uniform(var_min[t], var_max[t], N)
        for t in range(T)
    ]).T

    kalman_gains_lai = {}

    # Default placeholder only.
    # Replace with retrieval-specific observation error variance:
    # R = Var(LAI/FAPAR values from the five lowest-MSE LUT solutions)
    R = 0.01 ** 2

    for doy, v in zip(doy_obs, lai_obs):
        idx = np.where(doy_sim == doy)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        LAI_ens[:, i], K, res = enkf_update(LAI_ens[:, i], v, R)
        kalman_gains_lai[doy] = K

    lai_enkf_smooth = uniform_filter1d(LAI_ens.mean(axis=0), 5)

    # ============================================================
    # EnKF for FAPAR
    # ============================================================
    FAPAR_ens = np.array([
        np.random.uniform(fapar_min[t], fapar_max[t], N)
        for t in range(T)
    ]).T

    kalman_gains_fapar = {}

    for doy, v in zip(doy_obs, fapar_obs):
        idx = np.where(doy_sim == doy)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        FAPAR_ens[:, i], K, res = enkf_update(FAPAR_ens[:, i], v, R)
        kalman_gains_fapar[doy] = K

    fapar_enkf_smooth = uniform_filter1d(FAPAR_ens.mean(axis=0), 5)

    # ============================================================
    # Select structural-valid days
    # ============================================================
    mask = np.isin(doy_sim, df_struct.iloc[:, 0].values)
    doy_valid = doy_sim[mask]

    struct_df = df_struct.set_index(0).loc[doy_valid]
    PLA = struct_df[2].values
    SENLA = struct_df[3].values
    PLAG = struct_df[4].values
    PLAS = struct_df[5].values

    # ============================================================
    # HyperDA-Leaf Correction for LAI and FAPAR
    # ============================================================
    corrected_lai = recursive_correction_enhanced(
        lai_enkf_smooth[mask], PLAG, PLAS, PLA, SENLA,
        doy_valid, doy_obs, lai_obs,
        kalman_gains_lai, var_min[mask], var_max[mask]
    )

    corrected_fapar = recursive_correction_enhanced(
        fapar_enkf_smooth[mask], PLAG, PLAS, PLA, SENLA,
        doy_valid, doy_obs, fapar_obs,
        kalman_gains_fapar, fapar_min[mask], fapar_max[mask]
    )

    # ============================================================
    # Daily Log (optional)
    # ============================================================
    lai_obs_dict = dict(zip(doy_obs, lai_obs))
    fapar_obs_dict = dict(zip(doy_obs, fapar_obs))

    log_daily_values(
        doy_valid,
        lai_obs_dict, lai_enkf_smooth[mask], corrected_lai,
        fapar_obs_dict, fapar_enkf_smooth[mask], corrected_fapar
    )

    # ============================================================
    # Write final-day output for Fortran
    # ============================================================
    with open(output_txt_path, "w") as f:
        f.write(f"{corrected_lai[-1]:.6f}\t{corrected_fapar[-1]:.6f}\n")

    print(">>> Final LAI =", corrected_lai[-1])
    print(">>> Final FAPAR =", corrected_fapar[-1])


# ============================================================
# Command Line Entry
# ============================================================
if __name__ == "__main__":
    try:
        print(">>> Script started")
        main(sys.argv[1], sys.argv[2])
        print(">>> Script finished successfully")
    except Exception as e:
        print(">>> Error occurred:", e)
