import serial
import numpy as np
import time
import threading
from scipy.signal import butter, filtfilt, find_peaks, detrend
import csv
import datetime
import json

# --- 1. CONFIGURATION (CRITICAL: MUST MATCH ARDUINO) ---
SERIAL_PORT = 'COM6'  # <<< CURRENTLY SET TO COM6 - CHANGE AS NEEDED!
BAUD_RATE = 115200  # Must match Arduino

# Signal Processing Constants
FS = 100.0
WINDOW_SECONDS = 30
BUFFER_SIZE = int(WINDOW_SECONDS * FS)
CALCULATION_INTERVAL = 1.0

# --- DATA LOGGING CONSTANTS ---
LOGGING_INTERVAL = 30.0  # Log data every 30 seconds
LOG_FILE_NAME = 'patient_vitals_log.csv'
# ------------------------------

RAW_WAVEFORM_WINDOW_SECONDS = 5

# --- SPO2 CALIBRATION COEFFICIENTS ---
SPO2_A = 110.0
SPO2_B = 18.0

# --- FIREBASE SETUP ---
FIREBASE_CRED_PATH = "C:/Users/USER/Downloads/ambulance-monitoring-5ad1a-firebase-adminsdk-fbsvc-dd6120eecf.json"
FIREBASE_DATABASE_URL = "https://ambulance-monitoring-5ad1a-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_NODE_LIVE = 'vitals'  # Node for live data (fast updates)
FIREBASE_NODE_HISTORY = 'vitals_history'  # NEW: Node for historical data (slow updates)
# ----------------------------------------------------


# --- INITIALIZE FIREBASE ADMIN SDK ---
try:
    import firebase_admin
    from firebase_admin import credentials, db

    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
    ref_live = db.reference(FIREBASE_NODE_LIVE)
    ref_history = db.reference(FIREBASE_NODE_HISTORY)  # NEW: Reference for history
    print("--- Firebase Connection Successful ---")
except ImportError:
    ref_live = None
    ref_history = None
    print("\nFATAL ERROR: Firebase dependencies not found. Install 'firebase-admin'.")
except Exception as e:
    ref_live = None
    ref_history = None
    print(f"\nFATAL ERROR: Could not initialize Firebase. Details: {e}")

# --- 2. SIGNAL VALIDATION THRESHOLDS ---
ECG_NOISE_THRESHOLD = 30.0
PPG_MIN_DC_LEVEL = 10000.0

# --- 3. Signal Processing Constants (Python Side) ---
ECG_LOW_CUTOFF = 5.0
ECG_HIGH_CUTOFF = 15.0
ECG_FILTER_ORDER = 3
EDR_CUTOFF = 0.4
EDR_FILTER_ORDER = 3
PPG_LOW_CUTOFF = 0.5
PPG_HIGH_CUTOFF = 5.0
PPG_FILTER_ORDER = 3

# --- 4. Global Data Buffer, State, and Metrics ---
data_buffer_ecg = np.zeros(BUFFER_SIZE)
data_buffer_ir = np.zeros(BUFFER_SIZE)
data_buffer_red = np.zeros(BUFFER_SIZE)
data_buffer_temp = np.zeros(BUFFER_SIZE)

buffer_index = 0
data_points_received = 0
last_calculation_time = 0.0
last_logging_time = 0.0

# Initial state (used before first calculation)
latest_metrics = {
    "timestamp": int(time.time() * 1000),
    "hr": 0.0,
    "hrv": 0.0,
    "rr": 0.0,
    "spo2": 0.0,
    "temp": 0.0,
    "pds_score": 0,
    "pds_classification": "N/A",
    "pds_color": "gray",
    "hr_score": 0,  # Added for history logging
    "rr_score": 0,  # Added for history logging
    "spo2_score": 0,  # Added for history logging
    "hrv_score": 0,  # Added for history logging
    "ecg_waveform": [0.0] * int(RAW_WAVEFORM_WINDOW_SECONDS * FS),
    "ppg_waveform": [0.0] * int(RAW_WAVEFORM_WINDOW_SECONDS * FS),
    "rr_waveform": [0.0] * int(RAW_WAVEFORM_WINDOW_SECONDS * FS),
    "waveform_times": list(np.arange(0, RAW_WAVEFORM_WINDOW_SECONDS, 1 / FS)),
    "status": "INITIALIZING"
}


# --- 5. Filter and Calculation Functions ---
def bandpass_filter(data, lowcut, highcut, fs, order):
    """Designs and applies a Butterworth Bandpass filter."""
    if len(data) < 2 or np.std(data) < 1e-6:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def lowpass_filter(data, cutoff, fs, order):
    """Designs and applies a Butterworth Lowpass filter."""
    if len(data) < 2 or np.std(data) < 1e-6:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, data)


def calculate_hr_hrv(ecg_signal, fs):
    """Calculates HR, HRV (RMSSD) from ECG, and returns filtered ECG waveform."""
    ecg_detrended = detrend(ecg_signal)
    if np.std(ecg_detrended) < ECG_NOISE_THRESHOLD:
        return 0.0, 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    filtered_ecg = bandpass_filter(ecg_detrended, ECG_LOW_CUTOFF, ECG_HIGH_CUTOFF, fs, ECG_FILTER_ORDER)
    if np.std(filtered_ecg) < 1e-6:
        return 0.0, 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    peak_height_threshold = np.max(filtered_ecg) * 0.6
    min_peak_distance = int(fs * 0.3)
    peaks, _ = find_peaks(filtered_ecg, height=peak_height_threshold, distance=min_peak_distance)

    avg_hr = 0.0
    rmssd = 0.0
    if len(peaks) >= 2:
        r_peak_times_s = peaks / fs
        ibi_s = np.diff(r_peak_times_s)
        # Filter implausible IBI values (e.g., HR > 200 or HR < 50)
        ibi_s = ibi_s[(ibi_s > 0.3) & (ibi_s < 1.2)]
        if len(ibi_s) >= 2:
            avg_hr = 60.0 / np.mean(ibi_s)
            diff_ibi = np.diff(ibi_s)
            # RMSSD in milliseconds
            rmssd = np.sqrt(np.mean(diff_ibi ** 2)) * 1000

    return avg_hr, rmssd, filtered_ecg[-int(FS * RAW_WAVEFORM_WINDOW_SECONDS):]


def calculate_rr(ecg_signal, fs):
    """Calculates Respiratory Rate (RR) from ECG (EDR method), and returns EDR waveform."""
    ecg_detrended = detrend(ecg_signal)
    if np.std(ecg_detrended) < ECG_NOISE_THRESHOLD:
        return 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    edr_signal_raw = lowpass_filter(ecg_detrended, EDR_CUTOFF, fs, EDR_FILTER_ORDER)
    if np.std(edr_signal_raw) < 1e-6:
        return 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    edr_signal = detrend(edr_signal_raw)
    edr_signal_norm = (edr_signal - np.mean(edr_signal)) / np.std(edr_signal)

    min_respiration_period_samples = int((60.0 / 35.0) * fs)
    peaks, _ = find_peaks(edr_signal_norm, distance=min_respiration_period_samples, height=0.5)

    avg_rr_bpm = 0.0
    if len(peaks) >= 2:
        peak_times_s = peaks / fs
        ibi_s = np.diff(peak_times_s)
        avg_rr_bpm = 60.0 / np.mean(ibi_s)
        # Basic check to filter implausible RR (e.g., RR > 35 or RR < 5)
        if avg_rr_bpm > 35.0 or avg_rr_bpm < 5.0:
            avg_rr_bpm = 0.0

    return avg_rr_bpm, edr_signal[-int(FS * RAW_WAVEFORM_WINDOW_SECONDS):]


def calculate_spo2(ir_signal, red_signal, fs):
    """Calculates SpO2 using the AC/DC ratio method, and returns filtered PPG IR waveform."""
    global SPO2_A, SPO2_B

    if np.mean(ir_signal) < PPG_MIN_DC_LEVEL or np.mean(red_signal) < PPG_MIN_DC_LEVEL:
        return 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    ir_ac = bandpass_filter(detrend(ir_signal), PPG_LOW_CUTOFF, PPG_HIGH_CUTOFF, fs, PPG_FILTER_ORDER)
    red_ac = bandpass_filter(detrend(red_signal), PPG_LOW_CUTOFF, PPG_HIGH_CUTOFF, fs, PPG_FILTER_ORDER)

    if np.std(ir_ac) < 1e-6 or np.std(red_ac) < 1e-6:
        return 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    ir_dc = np.mean(ir_signal)
    red_dc = np.mean(red_signal)
    spo2 = 0.0

    # Calculate Ratio of Ratios (R)
    if ir_dc > 0 and red_dc > 0:
        R_red = np.mean(np.abs(red_ac)) / red_dc
        R_ir = np.mean(np.abs(ir_ac)) / ir_dc

        if R_ir > 0:
            R = R_red / R_ir
            spo2 = np.clip(SPO2_A - SPO2_B * R, 0.0, 100.0)

    # Filter implausible SpO2 (e.g., if calculation results in 0/100 without reason)
    if spo2 < 80.0 or spo2 > 100.0:
        return 0.0, np.zeros(int(FS * RAW_WAVEFORM_WINDOW_SECONDS))

    return spo2, ir_ac[-int(FS * RAW_WAVEFORM_WINDOW_SECONDS):]


def get_score_hr(hr):
    """Scores Heart Rate (HR) based on the Weight Distribution table."""
    if hr <= 0.0: return 0  # Treat 0.0 (error/N/A) as normal range for scoring stability check
    if 60 <= hr <= 100:
        return 0
    elif (50 <= hr <= 59) or (101 <= hr <= 110):
        return 1
    elif (40 <= hr <= 49) or (111 <= hr <= 120):
        return 2
    elif (hr < 40) or (hr > 120):
        return 3
    return 0


def get_score_rr(rr):
    """Scores Respiratory Rate (RR) based on the Weight Distribution table."""
    if rr <= 0.0: return 0
    if 12 <= rr <= 18:
        return 0
    elif (8 <= rr <= 11) or (19 <= rr <= 24):
        return 1
    elif 5 <= rr <= 7 or 25 <= rr <= 30:
        return 2
    elif (rr < 5) or (rr > 30):
        return 3
    return 0


def get_score_spo2(spo2):
    """Scores SpO2 based on the Weight Distribution table."""
    if spo2 <= 0.0: return 0
    if spo2 >= 95:
        return 0
    elif 93 <= spo2 <= 94:
        return 1
    elif 90 <= spo2 <= 92:
        return 2
    elif spo2 < 90:
        return 3
    return 0


def get_score_hrv(hrv):
    """Scores Heart Rate Variability (HRV) based on the Weight Distribution table (RMSSD is the HRV measure)."""
    if hrv <= 0.0: return 0
    if hrv > 30:
        return 0
    elif 25 <= hrv <= 29:
        return 1
    elif 20 <= hrv <= 24:
        return 2
    elif hrv < 20:
        return 3
    return 0


def calculate_pds(hr, rr, spo2, hrv):
    """
    Calculates the Patient Deterioration Score (PDS) and determines the triage protocol.
    """
    # If any core vital is 0 (due to sensor off/error), skip scoring and mark N/A/Gray.
    #if hr == 0.0 or rr == 0.0 or spo2 == 0.0:
        #return {
         #   "score": 0,
          #  "classification": "N/A - Sensor Error",
          #  "color": "gray",
           # "hr_score": 0,
            #"rr_score": 0,
            #"spo2_score": 0,
            #"hrv_score": 0
        #}

    # 1. Calculate individual scores
    score_hr = get_score_hr(hr)
    score_rr = get_score_rr(rr)
    score_spo2 = get_score_spo2(spo2)
    score_hrv = get_score_hrv(hrv)

    # 2. Total PDS Score
    total_pds_score = score_hr + score_rr + score_spo2 + score_hrv

    # 3. Determine Classification, Triage Protocol, and Color
    classification = "N/A"
    color_code = "gray"

    # Check for Critical Alert condition
    is_critical_alert = (8 <= total_pds_score <= 12) or (score_hr == 3) or (score_rr == 3) or (score_spo2 == 3) or (
            score_hrv == 3)

    if is_critical_alert:
        classification = "Critical/ High risk"
        color_code = "Red"
    elif 4 <= total_pds_score <= 7:
        classification = "Urgent/ Moderate risk"
        color_code = "Yellow"
    elif 0 <= total_pds_score <= 3:
        classification = "Stable/low risk"
        color_code = "Green"

    # Return comprehensive results
    return {
        "score": int(total_pds_score),
        "classification": classification,
        "color": color_code,
        "hr_score": score_hr,
        "rr_score": score_rr,
        "spo2_score": score_spo2,
        "hrv_score": score_hrv
    }


# --- 6. Data Parsing and Buffer Management ---
def parse_and_update_buffer(line):
    """Parses a serial line and updates the circular data buffers."""
    global buffer_index, data_points_received
    try:
        parts = line.split(',')
        if len(parts) != 6: return False

        # Check for expected non-zero raw data from sensors to skip noisy/empty samples
        ecg_val = float(parts[1])
        ir_val = float(parts[3])
        red_val = float(parts[4])
        temp_val = float(parts[5])

        # Basic Sanity check on raw data to ensure a proper reading
        if ir_val < 100 or red_val < 100 or ecg_val == 0:
            return False

        data_buffer_ecg[buffer_index] = ecg_val
        data_buffer_ir[buffer_index] = ir_val
        data_buffer_red[buffer_index] = red_val
        data_buffer_temp[buffer_index] = temp_val

        if data_points_received < BUFFER_SIZE:
            data_points_received += 1

        buffer_index = (buffer_index + 1) % BUFFER_SIZE
        return True
    except (ValueError, IndexError):
        return False


def get_chronological_buffer():
    """Reorders the circular buffer to be chronologically continuous."""
    # This is the standard, correct way to read a circular buffer
    ecg_c = np.concatenate((data_buffer_ecg[buffer_index:], data_buffer_ecg[:buffer_index]))
    ir_c = np.concatenate((data_buffer_ir[buffer_index:], data_buffer_ir[:buffer_index]))
    red_c = np.concatenate((data_buffer_red[buffer_index:], data_buffer_red[:buffer_index]))
    temp_c = np.concatenate((data_buffer_temp[buffer_index:], data_buffer_temp[:buffer_index]))

    # CRUCIAL FIX: Slice to only include received data points during initialization
    return (
        ecg_c[-data_points_received:],
        ir_c[-data_points_received:],
        red_c[-data_points_received:],
        temp_c[-data_points_received:]
    )


# --- 7. FIREBASE PUSH FUNCTION ---
def push_to_firebase(metrics, is_history=False):
    """Pushes the latest metrics object to the Firebase Realtime Database."""
    global ref_live, ref_history
    if ref_live is None or ref_history is None: return

    try:
        if is_history:
            # PUSH to a dedicated history node using timestamp as the key
            history_data = {k: v for k, v in metrics.items() if not k.endswith('_waveform') and k != 'waveform_times'}

            # Use timestamp string (ms since epoch) as key for chronological ordering in Firebase
            key = str(metrics['timestamp'])
            ref_history.child(key).set(history_data)
        else:
            # PUSH to the live vitals node
            ref_live.set(metrics)
    except Exception as e:
        # print(f"ERROR: Failed to push data to Firebase. History: {is_history}. Details: {e}")
        pass

    # --- 8. FIREBASE STATUS CHECK ---


def get_vitals_status_from_firebase():
    """Reads the status flag set by the HTML dashboard to control data stream."""
    global ref_live
    if ref_live is None: return 'OFFLINE'

    try:
        status_ref = db.reference(FIREBASE_NODE_LIVE + '/status')
        status_snapshot = status_ref.get()
        return status_snapshot if status_snapshot else 'OFFLINE'
    except Exception:
        return 'OFFLINE'


def get_active_logistics_id():
    """Reads the patient_id from the /logistics node for synchronization."""
    # NOTE: This is crucial for synchronizing data to the correct patient.
    try:
        logistics_ref = db.reference('logistics/patient_id')
        active_id = logistics_ref.get()
        return active_id if active_id and active_id != '--' else None
    except Exception as e:
        return None


# --- 9. LOGGING FUNCTION (Local CSV logging removed as per user request) ---


# --- 10. Serial Data Pipeline Thread (FIXED MAIN LOOP) ---
def serial_data_pipeline():
    """Reads serial data, calculates metrics, and updates the Firebase state."""
    global last_calculation_time, latest_metrics, data_points_received, last_logging_time

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        print(f"--- Serial Listener Started on {SERIAL_PORT} ---")
    except serial.SerialException as e:
        print(f"\nFATAL ERROR: Could not open serial port {SERIAL_PORT}. Details: {e}")
        latest_metrics["status"] = "SERIAL_ERROR"
        push_to_firebase(latest_metrics)
        return

    while True:
        try:
            # 1. Read all waiting data
            while ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("Timestamp_ms"): continue
                parse_and_update_buffer(line)

            current_time = time.time()

            # --- CHECK FOR ACTIVE PATIENT AND CONTROL SIGNAL ---
            active_patient_id = get_active_logistics_id()
            dashboard_status = get_vitals_status_from_firebase()
            # ---------------------------------------------------

            if dashboard_status != 'LIVE':
                if latest_metrics["status"] != 'OFFLINE':
                    latest_metrics["status"] = 'OFFLINE'
                    push_to_firebase({"status": "OFFLINE", "timestamp": int(current_time * 1000)})
                time.sleep(1)
                continue

            # 2. Check calculation interval
            if current_time - last_calculation_time >= CALCULATION_INTERVAL:
                last_calculation_time = current_time

                # 3. Buffer Check - **MUST** ensure the buffer is full before running calculations
                if data_points_received < BUFFER_SIZE:
                    latest_metrics["status"] = "BUFFERING"
                    print(f"Buffer collecting data: {data_points_received}/{BUFFER_SIZE} points...")
                    push_to_firebase({"status": "BUFFERING", "timestamp": int(current_time * 1000)})
                    time.sleep(CALCULATION_INTERVAL)
                    continue

                # 4. Perform Vitals Calculations
                ecg_c, ir_c, red_c, temp_c = get_chronological_buffer()

                avg_hr, rmssd, ecg_waveform = calculate_hr_hrv(ecg_c, FS)
                avg_rr, rr_waveform = calculate_rr(ecg_c, FS)
                spo2_val, ppg_waveform = calculate_spo2(ir_c, red_c, FS)
                avg_temp = np.mean(temp_c)

                # 4b. Perform PDS Calculation
                pds_results = calculate_pds(avg_hr, avg_rr, spo2_val, rmssd)

                # 5. Update Local Metrics (Rounding final output values)
                latest_metrics["timestamp"] = int(current_time * 1000)
                latest_metrics["patient_id"] = active_patient_id  # Sync Patient ID to live node
                latest_metrics["hr"] = round(avg_hr, 1) if avg_hr > 0 else 0.0
                latest_metrics["hrv"] = round(rmssd, 2) if rmssd > 0 else 0.0
                latest_metrics["rr"] = round(avg_rr, 2) if avg_rr > 0 else 0.0
                latest_metrics["spo2"] = round(spo2_val, 1) if spo2_val > 0 else 0.0
                latest_metrics["temp"] = round(avg_temp, 2) if avg_temp > 0 else 0.0
                latest_metrics["pds_score"] = pds_results["score"]
                latest_metrics["pds_classification"] = pds_results["classification"]
                latest_metrics["pds_color"] = pds_results["color"]

                # NEW: Add individual scores to metrics (crucial for history)
                latest_metrics["hr_score"] = pds_results["hr_score"]
                latest_metrics["rr_score"] = pds_results["rr_score"]
                latest_metrics["spo2_score"] = pds_results["spo2_score"]
                latest_metrics["hrv_score"] = pds_results["hrv_score"]

                latest_metrics["ecg_waveform"] = ecg_waveform.tolist()
                latest_metrics["ppg_waveform"] = ppg_waveform.tolist()
                latest_metrics["rr_waveform"] = rr_waveform.tolist()
                latest_metrics["status"] = "LIVE"

                # 6. PUSH LIVE DATA TO FIREBASE
                push_to_firebase(latest_metrics, is_history=False)

                # 7. LOG DATA TO FIREBASE HISTORY NODE
                if current_time - last_logging_time >= LOGGING_INTERVAL:
                    # Use a copy of the metrics that excludes waveform data
                    history_metrics = {k: v for k, v in latest_metrics.items() if
                                       not k.endswith('_waveform') and k != 'waveform_times'}
                    push_to_firebase(history_metrics, is_history=True)
                    last_logging_time = current_time
                    print(f"--- Data Logged to Firebase History ---")

                print(
                    f"| HR: {latest_metrics['hr']} BPM | RR: {latest_metrics['rr']} BPM | SpO2: {latest_metrics['spo2']}% | PDS: {latest_metrics['pds_score']} ({latest_metrics['pds_color']}) | -> Data Sent")

            time.sleep(0.001)

        except Exception as e:
            # Reopen the serial port if it fails due to permission/connection issues
            if "Access is denied" in str(e) or "ClearCommError" in str(e) or "device reports readiness failure" in str(
                    e):
                print(f"SERIAL CONNECTION INTERRUPTED: Attempting to reconnect on {SERIAL_PORT}...")
                latest_metrics["status"] = "RECONNECTING"
                push_to_firebase({"status": "RECONNECTING", "timestamp": int(time.time() * 1000)})
                time.sleep(2)
                try:
                    ser.close()
                    time.sleep(1)
                    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
                    ser.reset_input_buffer()
                    print(f"--- Reconnection Successful on {SERIAL_PORT} ---")
                except Exception as reconnect_e:
                    print(f"RECONNECTION FAILED: {reconnect_e}")
                    time.sleep(5)
            else:
                # General unexpected error
                print(f"An unexpected error occurred in serial pipeline: {e}")
                latest_metrics["status"] = "PROCESSING_ERROR"
                push_to_firebase(latest_metrics)
                time.sleep(1)


# --- 11. Main Execution Block ---

if __name__ == '__main__':
    print("--- Ambulatory Data Sender Initializing ---")

    # Start the serial data pipeline in a separate thread
    data_thread = threading.Thread(target=serial_data_pipeline, daemon=True)
    data_thread.start()

    # Keep the main thread alive so the daemon thread can run
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nSender stopped by user.")
            break
