from flask import Flask, render_template
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["sa"]     # change as needed
sources = db["urls"]       # change as needed

@app.route("/")
def index():
    # Pull 20 rows from MongoDB
    data = list(sources.find({}, {"url": 1, "rss": 1, "_id": 0}).limit(20))
    return render_template("index.html", rows=data)

if __name__ == "__main__":
    app.run(debug=True)
