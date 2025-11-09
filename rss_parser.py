from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from pymongo.collection import Collection


@dataclass
class Article:
    title: str
    link: str
    publication_date: str
    description: str


class RssParser:
    def __init__(
        self, url_collection: Collection, articles_collection: Collection
    ) -> None:
        self.url_col = url_collection
        self.art_col = articles_collection
        self.urls = self.get_docs()

    def parse(self):
        articles = self.start_thread_pool()
        print(articles[1])
        # articles_as_dict = articles_as_dict = [asdict(article) for article in articles]
        # self.art_col.instert_many(articles_as_dict)

    def worker(self, url: str) -> list[Article] | None:
        article_list = []
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                parser = BeautifulSoup(resp.content, "lxml-xml")
                articles = parser.find_all("item")

                for item in articles:
                    title = item.title.text if item.title else None
                    link = item.link.text if item.link else None
                    pub_date = item.pubDate.text if item.pubDate else None
                    description = item.description.text if item.description else None
                    items = (title, link, pub_date, description)
                    if all(items):
                        article_list.append(Article(*items))  # type: ignore

        except requests.exceptions.RequestException:
            return None

        return article_list

    def get_docs(self) -> list[str]:
        docs = list(
            self.url_col.find(
                {"rss": {"$nin": ["unknown", "N/A"]}}, {"rss": 1, "_id": 0}
            )
        )
        return [doc["rss"] for doc in docs]

    def start_thread_pool(self) -> list[Article]:
        results = []

        with ThreadPoolExecutor(max_workers=50) as ex:
            futures = {ex.submit(self.worker, url): url for url in self.urls}

            for future in as_completed(futures):
                result = future.result()

                if result:
                    results.extend(result)

        return results
