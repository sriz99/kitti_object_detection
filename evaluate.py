import sys
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

model_path = sys.argv[1]
model = YOLO(model_path)

metrics = model.val(
    data = 'data.yaml',
    split = 'val',
    imgsz = 640,
    batch = 16
)
print(metrics)
save_dir = Path(metrics.save_dir)

df = pd.DataFrame([metrics.results_dict])
csv_path = save_dir/'results_summary.csv'
df.to_csv(csv_path, index=False)

print(f'Evaluation metrics saved to {csv_path}')
