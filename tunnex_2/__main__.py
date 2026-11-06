# tunnex_2/__main__.py
import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")