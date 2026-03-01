import requests
import os
import ray


ray.init(ignore_reinit_error=True)


@ray.remote
def download_paper(paper_name: str, paper_url: str, data_dir: str):
    print(f"[{paper_name}] Requesting download...")

    file_path = os.path.join(data_dir, f"{paper_name}_paper.pdf")

    response = requests.get(paper_url)

    with open(file_path, "wb") as f:
        f.write(response.content)

    return f"[{paper_name}] Successfully saved to {file_path}"


if __name__ == "__main__":
    papers = {
        "mamba": "https://arxiv.org/pdf/2312.00752.pdf",
        "chronos": "https://arxiv.org/pdf/2403.07815.pdf",
    }

    DATA_DIR = "./data"

    # 2. Main Thread Preparation
    # It is safer to create the directory once in the main thread
    # to avoid race conditions where multiple parallel workers try to create it simultaneously.
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Dispatching parallel download tasks to Ray cluster...")

    # 3. Parallel Dispatch (Map Phase)
    futures = [
        download_paper.remote(name, url, DATA_DIR) for name, url in papers.items()
    ]

    # 4. Await Completion (Synchronization Barrier)
    results = ray.get(futures)

    for result in results:
        print(result)

    print("All datasets downloaded successfully!")
