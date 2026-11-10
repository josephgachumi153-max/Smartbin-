# SmartBin Web UI

This project provides a Flask interface for classifying waste images with a
MobileNetV2 model. The repository includes a trained model at
`model/smartbin2_best.keras`, its eight-class label map, and training metadata.

## Run the app

From the project root:

```powershell
.\venv\Scripts\python.exe -m web_app.app
```

Open `http://127.0.0.1:5000` and upload an image.

For a new environment, install the dependencies first:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r .\web_app\requirements.txt
```

## Train from RealWaste

`train_model.py` trains the eight SmartBin classes from the RealWaste archive.
The archive's `Food Organics` folder is excluded because it is not part of the
SmartBin taxonomy.

```powershell
.\venv\Scripts\python.exe train_model.py --archive C:\path\to\archive.zip
```

The script extracts the dataset into `training_data/` (ignored by Git), trains
the MobileNetV2 head and final layers, evaluates the held-out test split, and
writes the model and metadata into `model/`.

## Classes

`cardboard`, `paper`, `textile_trash`, `glass`, `metal`, `vegetation`,
`miscellaneous_trash`, and `plastic`.
