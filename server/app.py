from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image = request.files.get("file")
    response = jsonify(util.classify_image(image))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/", methods=["GET"])
def home():
    return render_template("app.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)