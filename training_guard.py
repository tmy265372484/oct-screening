"""Watch the existing training process and resume it if its terminal disappears."""

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from config import BASE_DIR
from train_resume import RUN_DIR, atomic_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def process_alive(pid, updated_at=None, expected_image=None):
    if not isinstance(pid, int) or pid <= 0:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 5:
            raise PermissionError(f"无法读取进程 {pid} 的状态，拒绝猜测或重复启动。")
        if error == 87:
            return False
        raise ctypes.WinError(error)
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return False
        if expected_image is not None:
            size = wintypes.DWORD(32768)
            image = ctypes.create_unicode_buffer(size.value)
            if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            if os.path.normcase(image.value) != os.path.normcase(str(expected_image)):
                return False
        if updated_at:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            created_at = ((created.dwHighDateTime << 32) + created.dwLowDateTime) / 10_000_000 - 11644473600
            if created_at > datetime.fromisoformat(updated_at).timestamp() + 2:
                return False  # The PID has been reused by a newer process.
        return True
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def guard_lock(run_dir):
    import msvcrt
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "guard.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("后台守护已经运行，无需重复启动。") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def keep_awake():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    active = bool(kernel.SetThreadExecutionState(0x80000001))  # Continuous + system required; screen may turn off.
    try:
        yield active
    finally:
        if active:
            kernel.SetThreadExecutionState(0x80000000)


def launch_training(run_dir, target_epochs):
    python = Path(sys.executable).with_name("python.exe")
    environment = dict(os.environ, PYTHONUTF8="1")
    with (run_dir / "background_console.log").open("ab", buffering=0) as console:
        return subprocess.Popen(
            [str(python), "-X", "utf8", "-u", str(BASE_DIR / "train_resume.py"), "--epochs", str(target_epochs),
             "--run-dir", str(run_dir)],
            cwd=BASE_DIR, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
            env=environment, close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def supervise(run_dir, target_epochs=50, interval=20, max_restarts=3):
    logger = logging.getLogger("oct_guard")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(run_dir / "guard.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    status = dict(supervisor_pid=os.getpid(), phase="starting", target_epochs=target_epochs,
                  restarts=0, training_pid=None, keep_awake=False)

    def record(**updates):
        status.update(updates, updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        atomic_json(run_dir / "guard_status.json", status)

    child = None
    pending_pid = None
    pending_since = 0
    missing_since = None
    try:
        with keep_awake() as awake:
            record(keep_awake=awake)
            logger.info("后台守护启动 PID=%s；临时阻止自动睡眠=%s。", os.getpid(), awake)
            if not awake:
                logger.warning("防自动睡眠请求失败，请检查电脑电源状态。")
            while True:
                if (run_dir / "guard.stop").exists():
                    record(phase="stopped")
                    logger.info("收到停止监督标记；保留正在运行的训练。")
                    return
                if child is not None:
                    child.poll()
                progress = read_json(run_dir / "progress.json")
                completed = progress.get("completed_epoch", 0)
                phase = progress.get("phase")
                pid = progress.get("pid")
                python = Path(sys.executable).with_name("python.exe")
                alive = process_alive(pid, progress.get("updated_at"), python)
                record(training_pid=pid, current_epoch=progress.get("current_epoch"),
                       batch=progress.get("batch"), batches=progress.get("batches"), completed_epoch=completed)
                if phase == "completed" and completed >= target_epochs:
                    record(phase="completed")
                    logger.info("训练已完成 %s 轮，停止监督并释放防睡眠请求。", completed)
                    return
                if phase == "interrupted":
                    record(phase="interrupted")
                    logger.info("训练收到人工中断，停止自动恢复。")
                    return
                if alive:
                    missing_since = None
                    pending_pid = None
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(progress["updated_at"])).total_seconds()
                    record(phase="progress_stale" if age > 1200 else "monitoring", progress_age_seconds=round(age), error=None)
                elif pending_pid and process_alive(pending_pid, expected_image=python):
                    record(phase="starting_training", training_pid=pending_pid)
                else:
                    if missing_since is None:
                        missing_since = time.monotonic()
                        logger.warning("训练进程已退出或不存在，等待 60 秒后尝试从完整检查点恢复。")
                    elapsed = time.monotonic() - missing_since
                    if elapsed >= 60:
                        if status["restarts"] >= max_restarts:
                            record(phase="failed", error=f"已达到 {max_restarts} 次自动恢复上限，请检查 training.log。")
                            logger.error(status["error"])
                            return
                        child = launch_training(run_dir, target_epochs)
                        pending_pid = child.pid
                        pending_since = time.monotonic()
                        missing_since = None
                        record(phase="restarting", training_pid=pending_pid, restarts=status["restarts"] + 1)
                        logger.warning("独立后台恢复训练 PID=%s，第 %s 次恢复。", pending_pid, status["restarts"])
                    else:
                        record(phase="waiting_to_restart")
                if pending_pid and time.monotonic() - pending_since > 300:
                    record(phase="startup_stale", error="后台训练启动超过 5 分钟，需检查日志；守护不会强行终止进程。")
                time.sleep(interval)
    except Exception as error:
        record(phase="failed", error=str(error))
        logger.exception("后台守护失败；未终止训练进程。")
        raise
    finally:
        status["keep_awake"] = False
        atomic_json(run_dir / "guard_status.json", status)
        handler.close()
        logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="监督已有训练；终端意外退出后独立恢复。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="只查询守护状态")
    mode.add_argument("--stop", action="store_true", help="停止监督，不结束训练")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    if args.status:
        state = read_json(args.run_dir / "guard_status.json")
        if state:
            state["supervisor_alive_now"] = process_alive(state.get("supervisor_pid"), state.get("updated_at"))
        print(json.dumps(state, ensure_ascii=False, indent=2))
    elif args.stop:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "guard.stop").write_text(datetime.now().isoformat(), encoding="utf-8")
        print("已请求停止后台监督；正在运行的训练不会被结束。")
    else:
        with guard_lock(args.run_dir):
            supervise(args.run_dir, args.epochs)


if __name__ == "__main__":
    main()
