from typing import Literal
from dataclasses import dataclass


type QuestionType = Literal['test', 'variants', 'text']
type EmployeeRequirement = str


@dataclass
class Question:
    question: str
    type: QuestionType
    answers: list[str]


@dataclass
class Test:
    title: str
    questions: list[Question]


@dataclass
class Profession:
    name: str
    employee_requirements: list[EmployeeRequirement]
