# Model weights

Place the trained PPE model at `models/best.pt` for the root dashboard.

Model weights are excluded from Git by `.gitignore` because they can be large and may have separate distribution terms. Obtain the correct trained model from your own training run or an authorized project artifact. Do not rename the generic root-level `yolo11n.pt` to `best.pt`; a generic pretrained model does not provide the project's custom PPE classes.

Before using a model, verify its class names and validate precision/recall on representative footage. Do not rely on unvalidated predictions for real workplace safety decisions.
