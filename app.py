from flask import Flask, render_template, request
from detect import detect_vehicle
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    count = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = "static/input.jpg"
        file.save(image_path)

        count, _ = detect_vehicle(image_path)

    return render_template("index.html", count=count)

if __name__ == "__main__":
    app.run(debug=True)
