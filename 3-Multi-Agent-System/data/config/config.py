from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "indexes"

QAVANIN_CONFIG = DATA_DIR / "config" / "qavanin_karbordi.txt"
