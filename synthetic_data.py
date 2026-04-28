
import numpy as np
from config import (NUM_PARTICIPANTS, BASELINE_DURATION, STRESS_DURATION,
                    RECOVERY_DURATION, THERMAL_FPS, FACIAL_ROIS, RANDOM_SEED)

ROI_SENSITIVITY = {
    "forehead":    {"base": 33.8, "stress_rise": 0.9,  "noise": 0.08},
    "periorbital": {"base": 33.2, "stress_rise": 0.5,  "noise": 0.12},
    "nose":        {"base": 34.1, "stress_rise": 1.2,  "noise": 0.10},
    "cheeks":      {"base": 33.5, "stress_rise": 0.7,  "noise": 0.09},
}

def _temperature_signal(duration_s, phase, roi, pf=1.0, fps=THERMAL_FPS):
    n   = duration_s * fps
    t   = np.linspace(0, duration_s, n)
    cfg = ROI_SENSITIVITY[roi]
    base   = cfg["base"] + np.random.normal(0, 0.3) * pf
    signal = base + 0.10 * np.sin(2 * np.pi * 0.05 * t)
    if phase == "stress":
        signal += np.linspace(0, cfg["stress_rise"] * pf, n)
        signal += 0.15 * np.sin(2 * np.pi * 0.1 * t)
    elif phase == "recovery":
        signal += cfg["stress_rise"] * pf * np.exp(-0.025 * t)
    signal += np.random.normal(0, cfg["noise"], n)
    return signal

def _heart_rate_signal(duration_s, phase, pf=1.0, fps=THERMAL_FPS):
    n       = duration_s * fps
    hr_base = {"baseline": 65, "stress": 85, "recovery": 70}[phase]
    hr_var  = {"baseline": 3,  "stress": 10, "recovery": 5}[phase]
    hr_base += np.random.normal(0, 5) * pf
    hr       = hr_base + hr_var * np.sin(np.linspace(0, 4 * np.pi, n))
    if phase == "stress":
        hr += np.linspace(0, 8 * pf, n)
    elif phase == "recovery":
        hr += 8 * pf * np.exp(-0.02 * np.arange(n) / fps)
    return hr + np.random.normal(0, 2.5, n)

def _respiration_signal(duration_s, phase, pf=1.0, fps=THERMAL_FPS):
    n    = duration_s * fps
    base = {"baseline": 13, "stress": 19, "recovery": 15}[phase]
    return base + np.random.normal(0, 1.5) * pf + np.random.normal(0, 1.8, n)

def generate_session(pid):
    np.random.seed(RANDOM_SEED + pid)
    pf      = np.random.uniform(0.7, 1.4)
    session = {"participant_id": pid, "participant_factor": pf, "phases": {}}
    for phase, dur in [("baseline", BASELINE_DURATION),
                       ("stress",   STRESS_DURATION),
                       ("recovery", RECOVERY_DURATION)]:
        roi_temps = {roi: _temperature_signal(dur, phase, roi, pf)
                     for roi in FACIAL_ROIS}
        hr        = _heart_rate_signal(dur, phase, pf)
        resp      = _respiration_signal(dur, phase, pf)
        session["phases"][phase] = {
            "thermal":         roi_temps,
            "heart_rate":      hr,
            "respiration":     resp,
            "blood_perfusion": roi_temps["cheeks"] * 0.55 + hr * 0.012,
        }
    return session

def generate_dataset(n=NUM_PARTICIPANTS):
    return [generate_session(i) for i in range(n)]
