from typing import Literal
from dataclasses import dataclass

from model.exceptions import QuestionDoesNotHaveCorrectAnswer


type QuestionType = Literal['test', 'variants', 'text']
type EmployeeRequirement = str


@dataclass
class Question:
    question: str
    type: QuestionType
    answers: list[str]
    correct_answer: str

    def validate(self) -> bool:
        """Verifies correctness."""
        # if self.correct_answer not in self.answers:
        #     raise QuestionDoesNotHaveCorrectAnswer

    def __post_init__(self) -> None:  # noqa: D105
        self.validate()


@dataclass
class Test:
    title: str
    questions: list[Question]


@dataclass
class Profession:
    name: str
    employee_requirements: list[EmployeeRequirement]
