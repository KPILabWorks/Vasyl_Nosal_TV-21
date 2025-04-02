import pandas as pd
import modin.pandas as mpd
from PyQt6.QtCore import QThread, pyqtSignal
import concurrent.futures


def read_file_efficient(file_path, chunksize=100000):
    try:
        if file_path.endswith(".csv"):
            chunks = pd.read_csv(file_path, chunksize=chunksize)
            df = pd.concat(chunks, ignore_index=True)
        elif file_path.endswith(".xlsx"):
            df = mpd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


class OptimizedLoadingThread(QThread):
    file_loaded = pyqtSignal(pd.DataFrame, str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(read_file_efficient, self.file_path)
            df = future.result()
        if df is None:
            self.file_loaded.emit(pd.DataFrame(), f"Failed to load {self.file_path}")
        else:
            self.file_loaded.emit(df, "")
