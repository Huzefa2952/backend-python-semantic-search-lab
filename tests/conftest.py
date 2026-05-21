import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = PROJECT_ROOT / "lib"

for path in [PROJECT_ROOT, LIB_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    import ollama  # noqa: F401
except ModuleNotFoundError:
    fake_ollama = types.ModuleType("ollama")

    def unconfigured_chat(*args, **kwargs):
        raise RuntimeError("ollama.chat was called before being mocked by the tests.")

    fake_ollama.chat = unconfigured_chat
    sys.modules["ollama"] = fake_ollama