import traceback
import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    print("[Wrapper] 开始执行 prepare_data.py", flush=True)
    from prepare_data import main
    main()
    print("[Wrapper] 完成", flush=True)
except SystemExit as e:
    print(f"[Wrapper] sys.exit({e.code})", flush=True)
    sys.exit(e.code)
except Exception as e:
    print(f"[Wrapper] 异常: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
