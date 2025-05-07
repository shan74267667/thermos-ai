import psutil, time
import numpy as np
from sklearn.ensemble import IsolationForest

# 1) Collect “normal” CPU data
print("Collecting baseline CPU data (10s)...")
baseline = []
for _ in range(20):
    baseline.append(psutil.cpu_percent(interval=0.5))
baseline = np.array(baseline).reshape(-1,1)

# 2) Train detector
clf = IsolationForest(contamination=0.05)
clf.fit(baseline)

# 3) Monitor in real‑time
print("Monitoring for anomalies. Press Ctrl+C to stop.")
while True:
    cpu = psutil.cpu_percent(interval=1)
    label = clf.predict([[cpu]])  # 1 = normal, -1 = anomaly
    if label[0] == -1:
        print(f"⚠️  Anomaly detected! CPU at {cpu}%")
    else:
        print(f"CPU {cpu}% — OK")
