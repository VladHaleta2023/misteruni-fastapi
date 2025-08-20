import os
import subprocess
import webbrowser
import logging
import shutil

# ================== Настройка логирования ==================
logger = logging.getLogger("anti_gui")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ================== Список запрещённых бинарей ==================
BLOCKED_KEYWORDS = [
    "chrome", "firefox", "safari", "edge", "explorer",
    "start", "xdg-open", "open"
]

# ================== Вспомогательные функции ==================
def block_action(action, details):
    logger.warning(f"Blocked {action}: {details}")
    raise RuntimeError(f"Blocked {action}: {details}")

def is_blocked_command(cmd):
    if isinstance(cmd, (list, tuple)):
        cmd_str = " ".join(str(c) for c in cmd)
    else:
        cmd_str = str(cmd)
    cmd_str = cmd_str.lower()
    return any(word in cmd_str for word in BLOCKED_KEYWORDS)

# ================== Сохраняем оригиналы ==================
_real_webbrowser_open = webbrowser.open
_real_system = os.system
_real_startfile = getattr(os, "startfile", None)

# Сохраняем все функции subprocess
_subprocess_funcs = {}
for func_name in ["Popen", "run", "call", "check_call", "check_output"]:
    _subprocess_funcs[func_name] = getattr(subprocess, func_name)

_real_shutil_which = shutil.which

# ================== Патчим функции ==================
def patch_functions():
    # webbrowser.open
    webbrowser.open = lambda *args, **kwargs: block_action("webbrowser.open", args)

    # os.system
    os.system = lambda cmd: block_action("os.system", cmd) if is_blocked_command(cmd) else _real_system(cmd)

    # os.startfile (только Windows)
    if _real_startfile:
        os.startfile = lambda *args, **kwargs: block_action("os.startfile", args) if is_blocked_command(args) else _real_startfile(*args, **kwargs)

    # subprocess
    for func_name, orig_func in _subprocess_funcs.items():
        def make_wrapper(orig):
            return lambda *args, **kwargs: block_action(func_name, args[0] if args else kwargs.get("args")) if is_blocked_command(args[0] if args else kwargs.get("args")) else orig(*args, **kwargs)
        setattr(subprocess, func_name, make_wrapper(orig_func))

    # shutil.which
    shutil.which = lambda cmd, *args, **kwargs: None if cmd.lower() in BLOCKED_KEYWORDS else _real_shutil_which(cmd, *args, **kwargs)

    # Selenium / Playwright / pyppeteer патчинг (если импортированы)
    try:
        from selenium.webdriver.chrome.webdriver import WebDriver
        WebDriver.__init__ = lambda *args, **kwargs: block_action("Selenium Chrome", args)
    except ImportError:
        pass

    try:
        import playwright.sync_api
        playwright.sync_api.sync_playwright = lambda *args, **kwargs: block_action("Playwright", args)
    except ImportError:
        pass

# Патчим сразу при импорте
patch_functions()