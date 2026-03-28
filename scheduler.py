import time
import subprocess
import os

def run_fetch():
    print("开始更新新闻...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["python", "fetch_news.py"], cwd=base_dir)

if __name__ == "__main__":
    while True:
        run_fetch()

        print("休眠30分钟...\n")
        time.sleep(1800)