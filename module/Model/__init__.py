import importlib
from pathlib import Path

current_dir = Path(__file__).parent

for path in current_dir.iterdir():
    if (path.suffix == '.py' and path.name != '__init__.py'):
        module_name = path.stem
        module = importlib.import_module(f'.{module_name}', __name__)

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (isinstance(obj, type)):
                globals()[attr_name] = obj