from flask import Flask, render_template
from flask import request
import json
import inference

 # -- coding: utf-8 --
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 뒤에 resource path X, 브라우저 접근
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    print(data)
    
    context, answer = inference.inference(data)
# 챗봇 엔진 질의 요청
    json_data = {
        'context': context,
        'answer': answer
    }
    send = json.dumps(json_data)
#     print(context, answer)
    return send

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=True)