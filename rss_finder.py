from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bson import ObjectId
from pymongo.collection import Collection


@dataclass
class UrlDoc:
    _id: ObjectId
    url: str
    rss: str


class RssFinder:
    RSS_LIST = ["/ruff.xml", "/rss", "/feed"]

    def __init__(self, url_collection: Collection) -> None:
        self.url_col = url_collection
        self.docs = self.get_docs()
        self.urls_to_process = []

    def start(self) -> None:
        for doc in self.docs:
            url = self.add_scheme(doc.url)
            self.urls_to_process.append((url, doc._id))
        self.start_thread_pool()

    def get_docs(self) -> list[UrlDoc]:
        docs = list(self.url_col.find({}, {"url": 1, "rss": 1, "_id": 1}))
        return [UrlDoc(**d) for d in docs]

    def worker(self, base_url: str, doc_id: ObjectId) -> None:
        for rss in self.RSS_LIST:
            url = f"{base_url}{rss}"
            try:
                resp = requests.get(url, timeout=2, allow_redirects=True)
                if resp.status_code == 200:
                    self.url_col.update_one({"_id": doc_id}, {"$set": {"rss": url}})
                    return
            except requests.exceptions.RequestException as e:
                print("exception", e)
                pass
        self.url_col.update_one({"_id": doc_id}, {"$set": {"rss": "N/A"}})

    def start_thread_pool(self) -> None:
        with ThreadPoolExecutor(max_workers=100) as ex:
            for url, doc_id in self.urls_to_process:
                ex.submit(self.worker, url, doc_id)

    def add_scheme(self, url: str, scheme="https") -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"{scheme}://{url.lstrip('/')}"
        return url
