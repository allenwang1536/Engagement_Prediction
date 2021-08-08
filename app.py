from flask import Flask, redirect, url_for, render_template, request

import Engagement

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST", "GET"])
def predict():
    if Engagement.predict_engagement()==0:
        return render_template("index.html", pred="disengaged")
    else:
        return render_template("index.html", pred="engaged")

# @app.route("/<name>")
# def user(name):
#     return f"Hello {name}!"

# @app.route("/admin")
# def admin():
#     return redirect(url_for("user", name ="Admin!"))

if __name__ == "__main__":
    app.run()

