import os
import threading
import time

from fetch_news import run_news_pipeline

_scheduler_thread = None
_scheduler_lock = threading.Lock()


def env_flag(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def run_fetch_cycle():
    print("开始更新新闻...")
    train_enabled = env_flag("NEWS_AUTO_TRAIN_ENABLED", True)
    run_news_pipeline(train_transformer=train_enabled)


def scheduler_loop():
    interval = int(os.getenv("NEWS_REFRESH_INTERVAL_SECONDS", "1800"))
    refresh_on_start = env_flag("NEWS_REFRESH_ON_START", True)

    if refresh_on_start:
        try:
            run_fetch_cycle()
        except Exception as exc:
            print(f"新闻自动更新启动任务失败: {exc}")

    while True:
        print(f"休眠 {interval} 秒...\n")
        time.sleep(interval)
        try:
            run_fetch_cycle()
        except Exception as exc:
            print(f"新闻自动更新失败: {exc}")


def start_background_scheduler():
    global _scheduler_thread

    if not env_flag("NEWS_AUTO_REFRESH_ENABLED", True):
        print("新闻自动更新已关闭")
        return

    with _scheduler_lock:
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            return

        _scheduler_thread = threading.Thread(
            target=scheduler_loop,
            name="news-auto-refresh",
            daemon=True,
        )
        _scheduler_thread.start()
        print("新闻自动更新线程已启动")

if __name__ == "__main__":
    scheduler_loop()
