import datetime
import textwrap
from llama_cpp import Llama
import pandas as pd
from itertools import product

model_path = "./models/13B/yulan-chat-2-13b.Q5_K_M.gguf"

param_grid = {
    'N_THREADS': [15],
    'N_BATCH': [512],
    'N_GPU_LAYERS': [100],
    'N_CTX': [8192],
    'MAX_TOKENS': [2000],
    'TEMPERATURE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'TOP_P': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'TOP_K': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'REPEAT_PENALTY': [1.1]
}

text = """
your text
"""

settings = """
Вы - искусственный интеллект - помощник, специализирующийся на аналитике. 
Ваши ответы точны, подробны и направлены исключительно на решение поставленной задачи, без лишних фраз. 
Вы ничего не выдумываете, а используете исключительно предоставленный контекст.
Все коммуникации происходят на русском языке и ответ ОБЯЗАТЕЛЬНО надо давать на русском языке исключая другие языки.
"""

summary_prompt = """На основе предоставленной стенограммы встречи подготовь краткие итоги встречи, включая достигнутые договоренности и планы на будущее. Ответ разбей на несколько логических и последовательных блоков"""

def generate_summary(LLM, prompt, text, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY, N_THREADS, N_GPU_LAYERS, N_BATCH):
    result = []
    prompt_parts = textwrap.wrap(text, N_CTX)
    for part in prompt_parts:
        if len(part.split()) > N_CTX:
            part_parts = textwrap.wrap(part, N_CTX)
            for subpart in part_parts:
                output = LLM(
                    f"{settings}\n[|Human|]:{prompt} {subpart}\n[|AI|]:",
                    max_tokens = MAX_TOKENS,
                    temperature = TEMPERATURE,
                    top_p = TOP_P,
                    top_k = TOP_K,
                    repeat_penalty = REPEAT_PENALTY
                )
                result.append(output["choices"][0]["text"])
        else:
            output = LLM(
                f"{settings}\n[|Human|]:{prompt} {part}\n[|AI|]:",
                max_tokens = MAX_TOKENS,
                temperature = TEMPERATURE,
                top_p = TOP_P,
                top_k = TOP_K,
                repeat_penalty = REPEAT_PENALTY
            )
            result.append(output["choices"][0]["text"])
    return ' '.join(result)

combinations = list(product(*param_grid.values()))

for combination in combinations:
    params = dict(zip(param_grid.keys(), combination))

    LLM = Llama(
        model_path=f"{model_path}",
        n_threads = params['N_THREADS'],
        n_batch = params['N_BATCH'],
        n_gpu_layers = params['N_GPU_LAYERS'],
        n_ctx = params['N_CTX']
    )

    start_time = datetime.datetime.now()

    output = generate_summary(LLM, summary_prompt, text, params['N_CTX'], params['MAX_TOKENS'], params['TEMPERATURE'], params['TOP_P'], params['TOP_K'], params['REPEAT_PENALTY'], params['N_THREADS'], params['N_GPU_LAYERS'], params['N_BATCH'])

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    params.update({
        'start_time': start_time,
        'end_time': end_time,
        'total_time': f'{hours}:{minutes}:{seconds}',
        'N_THREADS': params['N_THREADS'],
        'N_BATCH': params['N_BATCH'],
        'N_GPU_LAYERS': params['N_GPU_LAYERS'],
        'N_CTX': params['N_CTX'],
        'MAX_TOKENS': params['MAX_TOKENS'],
        'TEMPERATURE': params['TEMPERATURE'],
        'TOP_P': params['TOP_P'],
        'TOP_K': params['TOP_K'],
        'REPEAT_PENALTY': params['REPEAT_PENALTY'],
        'result': output
    })

    params['start_time'] = params['start_time'].strftime('%d.%m.%Y %H:%M:%S')
    params['end_time'] = params['end_time'].strftime('%d.%m.%Y %H:%M:%S')

    new_df = pd.DataFrame(params, index=[0])

    try:
        df = pd.read_excel('statistics.xlsx')
    except FileNotFoundError:
        df = pd.DataFrame()

    df = pd.concat([df, new_df], ignore_index=True)

    df.to_excel('statistics.xlsx', index=False)