from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


app = Flask(__name__)

bot = ChatBot("CoreMate", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(bot)

trainer.train("chatterbot.corpus.spanish")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_respose():
    user_text = request.args.get("msg")

    return str(bot.get_response(user_text))


if __name__ == "__main__":
    app.run(debug=True, port=8000)