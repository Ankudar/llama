import datetime
import textwrap
import re
from llama_cpp import Llama
import pandas as pd
import os
import time
from fuzzywuzzy import process
from rapidfuzz import process, fuzz

model_path = "./models/7B/kimiko-mistral-7b.Q5_K_M.gguf"
input_title = "./title/input/"
output_title = "./title/output/cold_lined_base.xlsx"
with open("./title/OKVED.txt", 'r', encoding = 'utf-8') as f:
    okveds = [line.strip() for line in f]

N_THREADS = 15
N_BATCH = 512
N_GPU_LAYERS = 100
N_CTX = 10000 #8192
MAX_TOKENS = 100
TEMPERATURE = 0
TOP_P = 0
REPEAT_PENALTY = 1.1

LLM = Llama(
    model_path=f"{model_path}",
    n_threads = N_THREADS,
    n_batch = N_BATCH, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers = N_GPU_LAYERS,
    n_ctx = N_CTX
    )

ai_settings = """
Тебе нужно по названию сайт, title сайта и description определить номер ОКВЭДА и его описание
Для ответа можно использовать только название сайта, title сайта и description сайта
Ответ должен содержать только название оквэда, без его кода
Ответ должен быть на русском языке
"""

def generate_okved(ai_settings, text, LLM, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY):
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
                stop_index = re.search(r'\n', generated_text)
                if stop_index != -1:
                    generated_text = generated_text[:stop_index+1]
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
            stop_index = re.search(r'\n', generated_text)
            if stop_index:
                generated_text = generated_text[:stop_index.start()+1]
            result.append(generated_text)
    return ' '.join(result)

def get_okved(ai_settings):
    while True:
        files = os.listdir(input_title)
        if len(files) > 0:
            for file in files:
                filepath = os.path.join(input_title, file)
                df = pd.read_csv(filepath) if file.endswith('.csv') else pd.read_excel(filepath)
                for index, row in df.iterrows():
                    url = str(row[0])  # get the URL from the first cell
                    # Check if the URL already exists in the output file
                    if os.path.exists(output_title):
                        df_out_existing = pd.read_excel(output_title)
                        if url in df_out_existing.iloc[:, 0].values:
                            continue

                    text = ' '.join(str(row[i]) for i in range(3))
                    ai_settings = ai_settings.replace('\n', ' ')
                    text = text.replace('\n', ' ')
                    result = generate_okved(ai_settings, text, LLM, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY)
                    result = result.split(' - ', 1)[-1].strip()

                    # Find the most relevant OKVED description
                    best_match = process.extractOne(result, okveds, scorer=fuzz.token_sort_ratio)
                    okved_number = re.search(r'^[A-Z0-9.]+', best_match[0]).group()

                    # Write the URL, result and OKVED number to the output file
                    data = {'URL': [url], 'RESULT': [result], 'OKVED': [okved_number]}
                    df_out_new = pd.DataFrame(data)
                    if os.path.exists(output_title):
                        df_out_existing = pd.read_excel(output_title)
                        df_out = pd.concat([df_out_existing, df_out_new])
                    else:
                        df_out = df_out_new
                    df_out.to_excel(output_title, index=False)
                os.remove(filepath)  # remove the file after processing
        else:
            print("Ожидается загрузка новых тайтлов для анализа...")
            time.sleep(10)  # wait for 10 seconds before checking again

if __name__ == "__main__":
    get_okved(ai_settings)