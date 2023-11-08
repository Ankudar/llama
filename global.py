from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, DecimalField, TextAreaField
from wtforms.validators import DataRequired, Length, NumberRange
from celery import Celery

import datetime
import textwrap
from llama_cpp import Llama
import pandas as pd
import os
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

# Настройка Celery
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

model_path = "./models/7B/kimiko-mistral-7b.Q5_K_M.gguf"

N_THREADS = 15
N_BATCH = 512
N_GPU_LAYERS = 100
N_CTX = 10000 #8192
MAX_TOKENS = 200#get_max_tokens #100-1000
TEMPERATURE = 0.1#get_temperature #0 - 0.9
TOP_P = 0.5
REPEAT_PENALTY = 1.1

LLM = Llama(
    model_path=f"{model_path}",
    n_threads = N_THREADS,
    n_batch = N_BATCH, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers = N_GPU_LAYERS,
    n_ctx = N_CTX
    )

ai_settings = """Ты интеллектуальный ассистент"""#get_ai_settings

def generate_answer(ai_settings, text, LLM, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY):
    result = []
    prompt_parts = textwrap.wrap(text, N_CTX)
    for part in prompt_parts:
        if len(part.split()) > N_CTX:
            part_parts = textwrap.wrap(part, N_CTX)
            for subpart in part_parts:
                output = LLM(
                    f"{ai_settings}\n\nUSER: {subpart}\nASSISTANT:",
                    max_tokens = MAX_TOKENS,
                    temperature = TEMPERATURE,
                    top_p = TOP_P,
                    repeat_penalty = REPEAT_PENALTY
                )
                generated_text = output["choices"][0]["text"]
                if "USER" or "User" in generated_text:
                    generated_text = generated_text.split("USER")[0]
                result.append(generated_text)
        else:
            output = LLM(
                f"{ai_settings}\n\nUSER: {part}\nASSISTANT:",
                max_tokens = MAX_TOKENS,
                temperature = TEMPERATURE,
                top_p = TOP_P,
                repeat_penalty = REPEAT_PENALTY
            )
            generated_text = output["choices"][0]["text"]
            if "USER" or "User" in generated_text:
                generated_text = generated_text.split("USER")[0]
            result.append(generated_text)
    return ' '.join(result)

class MyForm(FlaskForm):
    temperature = DecimalField('Температура', validators=[DataRequired(), NumberRange(min=0.09, max=1)])
    top_p = DecimalField('TOP_P', validators=[DataRequired(), NumberRange(min=0.09, max=1)])
    tokens = IntegerField('Число токенов', validators=[DataRequired(), NumberRange(min=49, max=1001)])
    model_config = TextAreaField('Маска модели', render_kw={"style": "resize:both; overflow:auto;"})
    query_text = TextAreaField('Текст запроса', render_kw={"style": "resize:both; overflow:auto;"})
    submit = SubmitField('Отправить запрос')

@app.route('/api', methods=['POST'])
def handle_request():
    try:
        data = request.get_json()
        text = data.get('text', '')
        # Запускаем задачу в фоновом режиме
        result = generate_answer(form.model_config.data, form.query_text.data, LLM, N_CTX, form.tokens.data, form.temperature.data, form.top_p.data, REPEAT_PENALTY)
        return {'result': result.get(timeout=60)}  # Получаем результат задачи, ждем не более 60 секунд
    except Exception as e:
        return {'error': str(e)}

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        form = MyForm()
        if form.validate_on_submit():
            result = generate_answer(form.model_config.data, form.query_text.data, LLM, N_CTX, form.tokens.data, form.temperature.data, form.top_p.data, REPEAT_PENALTY)
            return render_template('index.html', form=form, result=result)
        else:
            print("Form not validated or not submitted.")
        return render_template('index.html', form=form)
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)