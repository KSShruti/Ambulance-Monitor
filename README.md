🏥 Ambulatory Patient Monitoring & ER Triage System
This project implements a complete, real-time patient monitoring system designed for emergency response vehicles (ambulances) that seamlessly integrates with a hospital reception dashboard. It uses a Python backend to process raw sensor data and Firebase Realtime Database (RTDB) for real-time synchronization between all components.

✨ Features
Real-time Vitals Monitoring: Calculates Heart Rate (HR), SpO₂, Respiration Rate (RR), Temperature, and Heart Rate Variability (HRV) from simulated or live sensor data.

Patient Deterioration Score (PDS): Dynamically calculates a PDS (similar to MEWS/NEWS) for automated triage classification (Green/Yellow/Red risk).

Ambulance Monitor (Web): Allows EMTs to register a patient, input manual EPCR data (Mechanism of Injury, Notes), and view live vitals and ETA.

Hospital Reception Dashboard (Web): Provides a comprehensive view for reception staff to manage patients, view live vitals/PDS, assign ER rooms and staff based on triage, and generate an archival EPCR PDF on discharge.

Dynamic Room Management: ER rooms are categorized (Critical, Urgent, Less Urgent) and dynamically assigned by the system.
