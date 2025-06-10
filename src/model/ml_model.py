from transformers import pipeline

from model.entities import Profession, Question, QuestionType, Test


class MLModel:
    def __init__(self) -> None:
        self.generator = pipeline('text-generation', model='deepseek-ai/deepseek-llm-7b-base')

    def _generate_question(self, skill: str, question_type: QuestionType) -> str:
        prompt = f"""
    Сгенерируй технический вопрос для проверки навыка '{skill}'.
    Требования:
    1. Тип вопроса {question_type}.
    2. Для типа 'variants' приведи 4 варианта ответа и укажи правильные (только цифры, например '1' или '14').
    3. Для типа 'test' укажи цифру правильного ответа.
    4. Вопрос должен быть строгим и проверять конкретный аспект навыка.

    Выведи ответ в JSON-формате:
    {{
        "question": "текст вопроса",
        "type": "{question_type}",
        "answers": ["вариант 1", "вариант 2", ...]  # только для 'variants'
        "correct_answer": "A"
    }}
    """
        return self.generator(prompt, max_length=100)[0]['generated_text']

    def get_test(self, profession: Profession) -> Test:
        """Generates a survey for the provided profession."""
        questions = []
        for req in profession.employee_requirements:
            question_text = self._generate_question(req.requirement)
            questions.append(
                Question(
                    question=question_text,
                    type='text',
                    answers=[],
                )
            )

        return Test(title=f'Тест для {profession.name}', questions=questions)



if __name__ == '__main__':
    model = MLModel()
    test = Profession(
        'Python Junior Backend разработчик',
        [
            'Знание базы SQL',
            'Знание REST API',
            'Опыт работы с Docker и Docker Compose',  # noqa: RUF001
            'Знание протокола HTTP',
            'Умение работы с Git'  # noqa: RUF001
        ]
    )
    model.get_test(test)
