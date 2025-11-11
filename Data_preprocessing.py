import serial
import numpy as np
import time
import threading
from scipy.signal import butter, filtfilt, find_peaks, detrend
from datetime import datetime
import uuid  # Used for generating a unique patient ID

# --- FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import exceptions as fb_exceptions  # Import Firebase exceptions

# ----------------------------

# --- 1. CONFIGURATION (CRITICAL: MUST MATCH ARDUINO) ---
SERIAL_PORT = 'COM6'  # <<< CURRENTLY SET TO COM6 - CHANGE AS NEEDED!
BAUD_RATE = 115200  # Must match Arduino

# Signal Processing Constants
FS = 100.0
WINDOW_SECONDS = 30
BUFFER_SIZE = int(WINDOW_SECONDS * FS)
CALCULATION_INTERVAL = 1.0
EPCR_INTERVAL = 30.0  # NEW: Interval for e-PCR snapshot
RAW_WAVEFORM_WINDOW_SECONDS = 5

# --- SPO2 CALIBRATION COEFFICIENTS ---
SPO2_A = 110.0
SPO2_B = 18.0

# --- FIREBASE SETUP ---
FIREBASE_CRED_PATH = "C:/Users/USER/Downloads/ambulance-monitoring-5ad1a-firebase-adminsdk-fbsvc-dd6120eecf.json"
FIREBASE_DATABASE_URL = "https://ambulance-monitoring-5ad1a-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_NODE = 'vitals'
# ----------------------------------------------------


# --- INITIALIZE FIREBASE ADMIN SDK ---
ref = None
try:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
    ref = db.reference(FIREBASE_NODE)
    print("--- Firebase Connection Successful ---")
except Exception as e:
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

# --- 4. Global Data Buffer, State, and Metrics (UPDATED) ---
data_buffer_ecg = np.zeros(BUFFER_SIZE)
data_buffer_ir = np.zeros(BUFFER_SIZE)
data_buffer_red = np.zeros(BUFFER_SIZE)
data_buffer_temp = np.zeros(BUFFER_SIZE)

buffer_index = 0
data_points_received = 0
last_calculation_time = 0.0
last_epcr_time = 0.0  # NEW: Global variable for tracking e-PCR time

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
        ibi_s = ibi_s[(ibi_s > 0.3) & (ibi_s < 1.2)]
        if len(ibi_s) >= 2:
            avg_hr = 60.0 / np.mean(ibi_s)
            diff_ibi = np.diff(ibi_s)
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
            # Use empirical linear approximation formula with tunable coefficients
            spo2 = np.clip(SPO2_A - SPO2_B * R, 0.0, 100.0)

    return spo2, ir_ac[-int(FS * RAW_WAVEFORM_WINDOW_SECONDS):]


# --- 6. PDS SCORING FUNCTIONS ---
def get_score_hr(hr):
    """Scores Heart Rate (HR) based on the Weight Distribution table."""
    if hr <= 0.0: return 0
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
    """Calculates the Patient Deterioration Score (PDS) and determines the triage protocol."""
    score_hr = get_score_hr(hr)
    score_rr = get_score_rr(rr)
    score_spo2 = get_score_spo2(spo2)
    score_hrv = get_score_hrv(hrv)

    total_pds_score = score_hr + score_rr + score_spo2 + score_hrv
    classification = "N/A"
    color_code = "gray"

    is_critical_alert = (8 <= total_pds_score) or (score_hr == 3) or (score_rr == 3) or (score_spo2 == 3) or (
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

    return int(total_pds_score), classification, color_code


# --- END PDS SCORING FUNCTIONS ---


# --- 7. Data Parsing and Buffer Management ---
def parse_and_update_buffer(line):
    """Parses a serial line and updates the circular data buffers."""
    global buffer_index, data_points_received

    if not line:
        return False

    try:
        parts = line.split(',')
        if len(parts) != 6: return False

        # Parse data values
        ecg_val = float(parts[1])
        ir_val = float(parts[3])
        red_val = float(parts[4])
        temp_val = float(parts[5])

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
    ecg_c = np.concatenate((data_buffer_ecg[buffer_index:], data_buffer_ecg[:buffer_index]))
    ir_c = np.concatenate((data_buffer_ir[buffer_index:], data_buffer_ir[:buffer_index]))
    red_c = np.concatenate((data_buffer_red[buffer_index:], data_buffer_red[:buffer_index]))
    temp_c = np.concatenate((data_buffer_temp[buffer_index:], data_buffer_temp[:buffer_index]))
    return ecg_c, ir_c, red_c, temp_c


# --- 8. FIREBASE PUSH FUNCTIONS (MODIFIED for e-PCR) ---
def push_to_firebase(metrics):
    """Pushes the latest metrics object to the Firebase Realtime Database."""
    global ref
    if ref is None:
        return

    try:
        ref.set(metrics)
    except fb_exceptions.FirebaseError as e:
        print(f"ERROR: Failed to push data to Firebase (Push): {e}")


def push_epcr_to_firebase(metrics):
    """Pushes a snapshot of all metrics to the dedicated e_pcr_data node."""
    global ref
    if ref is None:
        return

    # Use a child reference to keep a historical log, keyed by timestamp
    epcr_ref = db.reference('e_pcr_data').child(str(metrics["timestamp"]))

    try:
        # Create a clean, flat dictionary for the report
        report_data = {
            "timestamp_ms": metrics["timestamp"],
            "patient_id": metrics.get("patient_id", "N/A"),
            "hr": metrics["hr"],
            "hrv": metrics["hrv"],
            "rr": metrics["rr"],
            "spo2": metrics["spo2"],
            "temp": metrics["temp"],
            "pds_score": metrics["pds_score"],
            "pds_classification": metrics["pds_classification"],
            "status": metrics["status"],
            "generated_at": datetime.now().isoformat()
        }

        epcr_ref.set(report_data)
        print("-> e-PCR Snapshot Saved")
    except fb_exceptions.FirebaseError as e:
        print(f"ERROR: Failed to push data to Firebase (e-PCR): {e}")


# --- 9. FIREBASE STATUS CHECK & UTILITIES ---
def get_vitals_status_from_firebase():
    """Reads the status flag set by the HTML dashboard to control data stream."""
    global ref
    if ref is None:
        return 'OFFLINE'

    try:
        status_ref = db.reference(FIREBASE_NODE + '/status')
        status_snapshot = status_ref.get()
        return status_snapshot if status_snapshot else 'OFFLINE'
    except fb_exceptions.FirebaseError as e:
        print(f"Warning: Failed to read Firebase status: {e}")
        return 'OFFLINE'


def get_active_logistics_id():
    """Reads the patient_id from the /logistics node for synchronization."""
    try:
        logistics_ref = db.reference('logistics/patient_id')
        active_id = logistics_ref.get()
        return active_id if active_id and active_id != '--' else None
    except fb_exceptions.FirebaseError:
        return None


def check_and_initialize_patient_record(patient_id):
    """
    Checks if a patient record exists for the ID found in /logistics.
    If not, creates a minimal, active record to enable Reception Dashboard's auto-assignment.
    This handles cases where the monitor sets logistics but doesn't fully register the patient.
    """
    if not patient_id:
        return

    patient_ref = db.reference(f'patients/{patient_id}')
    patient_record = patient_ref.get()

    # If the patient record does NOT exist or is marked DISCHARGED
    if not patient_record or patient_record.get('current_room') == 'DISCHARGED':
        # Fetch logistics data for name/age (optional, use placeholder if missing)
        logistics_data = db.reference('logistics').get()

        # Create a minimal patient record for auto-assignment
        patient_ref.set({
            'id': patient_id,
            'name': logistics_data.get('patient_name') if logistics_data else f"Ambulance Patient {patient_id}",
            'age': logistics_data.get('patient_age') if logistics_data else 0,
            'gender': logistics_data.get('patient_gender') if logistics_data else 'O',
            'condition': 'Incoming - Vitals Monitor',
            'is_active': True,  # CRITICAL: Must be True
            'registration_time': datetime.now().isoformat(),
            'current_room': '',  # CRITICAL: Must be empty for auto-assignment
            'assigned_doctor_id': '',
            'assigned_nurse_id': ''
        })
        print(f"--- CREATED PATIENT RECORD {patient_id} for auto-assignment. ---")


# --- END FIREBASE UTILITIES ---


# --- 10. Serial Data Pipeline Thread (MODIFIED for e-PCR) ---
def serial_data_pipeline():
    """Reads serial data, calculates metrics, and updates the Firebase state."""
    global last_calculation_time, last_epcr_time, latest_metrics, data_points_received, EPCR_INTERVAL

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        print(f"--- Serial Listener Started on {SERIAL_PORT} ---")
    except serial.SerialException as e:
        print(f"\nFATAL ERROR: Could not open serial port {SERIAL_PORT}. Details: {e}")
        latest_metrics["status"] = "SERIAL_ERROR"
        push_to_firebase(latest_metrics)
        return

    # Track if we have performed the check for the current active patient
    last_checked_patient_id = None

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

            # CRITICAL CHECK FOR AMBULATORY REGISTERED PATIENTS
            if active_patient_id and active_patient_id != last_checked_patient_id:
                check_and_initialize_patient_record(active_patient_id)
                last_checked_patient_id = active_patient_id

            # --- Condition 1: Monitor is offline (no patient registered) ---
            if not active_patient_id:
                if latest_metrics["status"] != 'MONITOR_OFFLINE':
                    latest_metrics["status"] = 'MONITOR_OFFLINE'
                    # Only push minimal status data to save bandwidth
                    push_to_firebase({"status": "MONITOR_OFFLINE", "timestamp": int(current_time * 1000)})
                time.sleep(1)
                continue

            # 2. Check calculation interval
            if current_time - last_calculation_time >= CALCULATION_INTERVAL:
                last_calculation_time = current_time

                # 3. Buffer Check (Logs BUFFERING status until full)
                if data_points_received < BUFFER_SIZE:
                    latest_metrics["status"] = "BUFFERING"
                    # Push buffer status to Firebase so dashboard knows it's waiting
                    push_to_firebase({"status": "BUFFERING", "timestamp": int(current_time * 1000)})
                    time.sleep(0.01)
                    continue

                # --- Condition 3: Dashboard has gone OFFLINE after buffering finished ---
                if dashboard_status != 'LIVE':
                    if latest_metrics["status"] != 'OFFLINE':
                        latest_metrics["status"] = 'OFFLINE'
                        push_to_firebase({"status": "OFFLINE", "timestamp": int(current_time * 1000)})
                    time.sleep(1)
                    continue

                # --- CODE REACHES HERE ONLY IF BUFFER IS FULL AND STATUS IS LIVE ---

                # 4. Perform Calculations
                ecg_c, ir_c, red_c, temp_c = get_chronological_buffer()

                avg_hr, rmssd, ecg_waveform = calculate_hr_hrv(ecg_c, FS)
                avg_rr, rr_waveform = calculate_rr(ecg_c, FS)
                spo2_val, ppg_waveform = calculate_spo2(ir_c, red_c, FS)
                avg_temp = np.mean(temp_c)

                # --- PDS Calculation ---
                pds_score, pds_classification, pds_color = calculate_pds(avg_hr, avg_rr, spo2_val, rmssd)
                # -----------------------

                # 5. Update Local Metrics (and prepare for push)
                latest_metrics["timestamp"] = int(current_time * 1000)
                latest_metrics["patient_id"] = active_patient_id
                latest_metrics["hr"] = round(avg_hr, 1)
                latest_metrics["hrv"] = round(rmssd, 2)
                latest_metrics["rr"] = round(avg_rr, 2)
                latest_metrics["spo2"] = round(spo2_val, 1)
                latest_metrics["temp"] = round(avg_temp, 2)
                latest_metrics["pds_score"] = pds_score
                latest_metrics["pds_classification"] = pds_classification
                latest_metrics["pds_color"] = pds_color
                latest_metrics["ecg_waveform"] = ecg_waveform.tolist()
                latest_metrics["ppg_waveform"] = ppg_waveform.tolist()
                latest_metrics["rr_waveform"] = rr_waveform.tolist()
                latest_metrics["status"] = "LIVE"

                # 6. PUSH TO FIREBASE (LIVE VITALS)
                push_to_firebase(latest_metrics)

                # 7. e-PCR SNAPSHOT (30-second interval)
                if current_time - last_epcr_time >= EPCR_INTERVAL and latest_metrics["status"] == "LIVE":
                    last_epcr_time = current_time
                    push_epcr_to_firebase(latest_metrics)

                print(
                    f"| HR: {latest_metrics['hr']} BPM | RR: {latest_metrics['rr']} BPM | SpO2: {latest_metrics['spo2']}% | PDS: {latest_metrics['pds_score']} ({latest_metrics['pds_color']}) | -> Data Sent")

            time.sleep(0.001)

        except Exception as e:
            # Reopening serial connection on failure
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

    # If there is a patient ID in logistics that wasn't properly registered,
    # the check will handle it when the loop starts.

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
