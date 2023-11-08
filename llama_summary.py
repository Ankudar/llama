import datetime
import textwrap
import re
from llama_cpp import Llama

model_path = "./models/7B/kimiko-mistral-7b.Q5_K_M.gguf"

N_THREADS = 15
N_BATCH = 512
N_GPU_LAYERS = 100
N_CTX = 10000 #8192
MAX_TOKENS = 2000
TEMPERATURE = 0.7
TOP_P = 0.7
REPEAT_PENALTY = 1.1

LLM = Llama(
    model_path=f"{model_path}",
    n_threads = N_THREADS,
    n_batch = N_BATCH, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers = N_GPU_LAYERS,
    n_ctx = N_CTX
    )

text = """
your text
"""

start_time = datetime.datetime.now()
current_time = start_time.strftime("%d-%m-%Y %H:%M:%S")
print(f"   |--- время старта {current_time}")

ai_settings = """
your llama setting
example - you are ai assistent etc
"""

answer_settings = """
answer setting
example - send answer only russia language
"""

prompt_1 = """
[Тональность]
Оцени характер общения в этом тексте: является ли он дружественным, нейтральным или возможно, ты заметил признаки конфликта?
"""
prompt_2 = """
[Увольнение]
В тексте есть упоминания или признаки того, что кто-то из участников беседы рассматривает возможность увольнения? Если да, то по каким причинам?
"""
prompt_3 = """
[Вредительство]
Содержит ли представленный текст информацию о вредительстве, порче имущества или сокрытии какой-либо важной информации?
"""
prompt_4 = """
[Усталость]
Наблюдаются ли в тексте признаки усталости, нежелания выполнять свои задачи, ухода от ответственности и обязанностей, игнорирования задач и встреч?
"""
prompt_5 = """
[Стресс]
Оцени наличие эмоционального дискомфорта в тексте. Можно ли обнаружить признаки стресса, тревоги или депрессии у участников беседы?
"""
prompt_6 = """
[Личная_жизнь]
В тексте есть упоминания или признаки того, что кто-то из участников беседы испытывает проблемы в личной жизни, которые могут влиять на его работу или взаимодействие с коллегами?
"""
prompt_7 = """
[Конфликт]
Содержит ли представленный текст информацию о возможных конфликтах интересов, коррупции, недобросовестной конкуренции или других неправомерных действиях?
"""
prompt_8 = """
[Мотивация]
Есть ли в тексте признаки того, что участники беседы испытывают проблемы с мотивацией, уровнем удовлетворенности работой или отношением к своим обязанностям?
"""
prompt_9 = """
[Карьера]
В тексте есть упоминания или признаки того, что кто-то из участников беседы планирует изменить свою карьеру, перейти на другую должность или полностью поменять сферу деятельности?
"""

def generate_summary(ai_settings, answer_settings, prompt, text, LLM, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY):
    result = []
    prompt_parts = textwrap.wrap(text, N_CTX)
    for part in prompt_parts:
        if len(part.split()) > N_CTX:
            part_parts = textwrap.wrap(part, N_CTX)
            for subpart in part_parts:
                output = LLM(
                    f"{ai_settings}\n\nUSER: {subpart} {answer_settings}\nASSISTANT:",
                    max_tokens = MAX_TOKENS,
                    temperature = TEMPERATURE,
                    top_p = TOP_P,
                    repeat_penalty = REPEAT_PENALTY
                )
                generated_text = output["choices"][0]["text"]
                # stop_index = generated_text.find('.')
                # if stop_index != -1:
                #     generated_text = generated_text[:stop_index+1]
                # result.append(generated_text)
        else:
            output = LLM(
                f"{ai_settings}\n\nUSER: {part} {answer_settings}\nASSISTANT:",
                max_tokens = MAX_TOKENS,
                temperature = TEMPERATURE,
                top_p = TOP_P,
                repeat_penalty = REPEAT_PENALTY
            )
            generated_text = output["choices"][0]["text"]
            # stop_index = re.search(r'\.|\n', generated_text)
            # if stop_index:
            #     generated_text = generated_text[:stop_index.start()+1]
            # result.append(generated_text)
    return ' '.join(result)

def write_to_file(filename, text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)

def print_execution_time(start_time, end_time):
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"   |--- время обработки {hours}:{minutes}:{seconds}")

start_time = datetime.datetime.now()

summary_prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7, prompt_8, prompt_9]
all_summaries = ""

for prompt in summary_prompts:
    ai_settings = ai_settings.replace('\n', ' ')
    answer_settings = answer_settings.replace('\n', ' ')
    prompt = prompt.replace('\n', ' ')
    text = text.replace('\n', ' ')
    summaries = generate_summary(ai_settings, answer_settings, prompt, text, LLM, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY)
    all_summaries += summaries + "\n"

write_to_file('summary_middle_itter.txt', all_summaries)

end_time = datetime.datetime.now()
print_execution_time(start_time, end_time)