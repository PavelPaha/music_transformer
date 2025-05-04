from flask import Flask, render_template_string, request, send_file
from transformers import pipeline
import scipy.io.wavfile
import io
import consts

app = Flask(__name__)


def load_model():
    return  pipeline(
        consts.task,
        model=consts.actual_checkpoint,
        device=4,  # cuda:4
    )

model = load_model()


@app.route('/')
def index():
    return render_template_string(consts.html)

@app.route('/generate/<genre>')
def generate(genre):
    if genre not in consts.GENRE_TOKENS:
        return "Invalid genre", 400

    try:
        max_tokens = int(request.args.get('max_tokens', 500))
    except ValueError:
        return "Invalid max_tokens", 400
    max_tokens = max(1, min(max_tokens, 2000))

    start_token = consts.GENRE_TOKENS[genre]
    infer_params = {
        "do_sample": True,
        "temperature": 1.1,
        "max_new_tokens": max_tokens
    }
    out = model(start_token, forward_params=infer_params)
    rate = out["sampling_rate"]
    data = out["audio"]
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate, data)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='audio/wav',
        as_attachment=False,
        download_name=f"{genre}.wav"
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
