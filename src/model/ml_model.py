import json

from llama_cpp import Llama

from model.entities import Profession, Question, Test


QUESTIONS_BY_SKILL = 3


class MLModel:
    def __init__(self) -> None:
        self.llm = Llama(
            model_path='model/mistral-7b-instruct-v0.1.Q5_K_M.gguf',  # Скачайте заранее
            n_ctx=2048,
            n_threads=14,  # Используем 14 ядер вашего i7-12700H
            n_gpu_layers=0  # Только CPU
        )

    def _generate_question(self, skill: str) -> dict:
        prompt = f"""Сгенерируй {QUESTIONS_BY_SKILL} технических вопросов для проверки навыка '{skill}'.

Всего есть 3 типа вопросов:
    - variants - это вопрос на который ты предоставляешь варианты ответа, при чем среди них может быть несколько
    правильных
    - test - тоже вопрос с вариантами ответа, но правильный только один
    - text - это вопрос с развернутым ответом. на этот вопрос ты предоставляешь то, как примерно должен выглядеть
    правильный ответ

Первый вопрос должен быть типа test, второй типа variants и последний типа text.

Строго соблюдай следующий формат JSON:
{{
    "questions": [
        {{
            "question": "текст вопроса",
            "type": "test",
            "answers": ["вариант 1", ...],
            "correct_answer": "номер правильного ответа"
        }},
        {{
            "question": "текст вопроса",
            "type": "variants",
            "answers": ["вариант 1", ...],
            "correct_answer": "номера правильных ответов"
        }},
        {{
            "question": "текст вопроса",
            "type": "text",
            "answers": [],
            "correct_answer": "пример правильного ответа"
        }}
    ]
}}

Пример ответа:
'{{\\n"questions":[\\n{{\\n"question":"Какой оператор SQL используется для извлечения данных из таблицы?",\\n"type":"test","answers":[\\n"INSERT",\\n"SELECT",\\n"UPDATE",\\n"DELETE"\\n],\\n"correct_answer":"2"\\n}},\\n{{\\n"question":"Какие из следующих операторов SQL используются для изменения данных в таблице?",\\n"type":"variants",\\n"answers":[\\n"SELECT",\\n"INSERT",\\n"UPDATE",\\n"DELETE",\\n"ALTER"\\n],\\n"correct_answer":"234"\\n}},\\n{{\\n"question":"Объясните, что делает оператор GROUP BY в SQL.",\\n"type":"text",\\n"answers":[],\\n"correct_answer":(\\n"Оператор GROUP BY в SQL используется для группировки строк в результирующем наборе по одному или нескольким столбцам. Он часто применяется с агрегатными функциями (такими как COUNT, SUM, AVG и др.) для выполнения вычислений по каждой группе. Например, GROUP BY может использоваться для подсчета количества записей в каждой группе или вычисления среднего значения по группе."\\n)\\n}}\\n]\\n}}'
ОТВЕТ ДОЛЖЕН БЫТЬ БЕЗ ФОРМАТИРОВАНИЯ, добавления ```json в начало и ``` в конец, просто СЫРОЙ JSON

Требования к полям ответа:
correct_answer:
    - для type = 'variants' - только номера правильных ответов (например, '12').
    - для type = 'test' - только номер правильного ответа.
    - для type = 'text' - пример правильного развернутого ответа.
answers: ответы не должны содержать номера, только сам текст ответа, например, не '1. <содержание ответа>',
а просто '<содержание ответа>'. если тип вопроса text - это поле должно быть пустым списком.

также все ответы в поле answers и сам вопрос в поле question должны быть на русском языке, но допускается использование
английского для названий технологий и других технических терминов.
"""  # noqa: E501, RUF001

        response = self.llm.create_chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=500,
            temperature=0.7
        )
        message = response['choices'][0]['message']['content']
        print(f'{message=}')
        while message.startswith(' '):
            message = message[1:]
        if message.startswith('```json'):
            message = message[7:]
        if message.endswith('```'):
            message = message[:-3]
        print(f'{message=}')
        return json.loads(message)

    def get_test(self, profession: Profession) -> Test:
        """Returns test for check skills for provided profession."""
        questions = []
        for req in profession.employee_requirements:
            data = self._generate_question(req)
            questions.append(Question(
                question=data['question'],
                type=data['type'],
                answers=data.get('answers', []),
                correct_answer=data.get('correct_answer', '')
            ))
        return Test(title=f'Тест для {profession.name}', questions=questions)


if __name__ == '__main__':
    model = MLModel()
    profession = Profession(
        'Python Junior Backend разработчик',
        [
            'Знание основ языка SQL',
            'Знание REST API',
            'Опыт работы с Docker и Docker Compose',  # noqa: RUF001
            'Знание протокола HTTP',
            'Умение работы с Git'  # noqa: RUF001
        ]
    )
    test = model.get_test(profession)
    print(test)
