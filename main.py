"""Minimal connectivity check: one Noul through the Vercel AI Gateway.

Reads TYPESAFE_API_KEY from .env. Usage: uv run ask_single_question.py
"""

import json

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, NoulAnswer

load_dotenv()

GATEWAY_URL = "https://api.typesafe.ai"

# Letters + whitespace
letters = "abcdefghijklmnopqrstuvwxyz"
class Token:
    WHITESPACE = "whitespace"
    DOT = "dot"
    QUESTION_MARK = "question_mark"
    FINISHED = "end_of_text"
    BACKSPACE = "backspace"

def questions(only_letters: list[str] | None = None) -> dict[str, Noul]:
    print(f"ONLY LETTERS: {only_letters}")
    questions_dict = {}
    for letter in only_letters or letters:
        questions_dict[letter] = Noul(instructions=f"Should the next letter of the `response` be '{letter}'?")
    questions_dict[Token.WHITESPACE] = Noul(instructions=f"Should the next letter of the `response` be a whitespace?")
    questions_dict[Token.DOT] = Noul(instructions=f"Should the next letter of the `response` be a dot?")
    questions_dict[Token.QUESTION_MARK] = Noul(instructions=f"Should the next letter of the `response` be a question mark?")
    questions_dict[Token.FINISHED] = Noul(instructions=f"Is the `response` complete?")
    questions_dict[Token.BACKSPACE] = Noul(instructions=f"Should the last letter of the `response` be removed?")

    return questions_dict

def top_letters(response: dict[str, NoulAnswer]) -> list[str, float]:
    top_5_letters_with_p = sorted(response, key=lambda l: response[l].noul, reverse=True)[:5]
    return [(letter, response[letter].noul) for letter in top_5_letters_with_p]

def best_letter(response: dict[str, NoulAnswer]) -> str:
    return max(response, key=lambda l: response[l].noul)

class Client:
    def __init__(self):
        self.client = TypeSafeClient(base_url=GATEWAY_URL)
        self.response = ""
        self.finished = False

    def complete(self, system_prompt: str, input_text: str, letters_count: int, correction_loops: int) -> str:
        for _ in range(letters_count):
            letter = self.next_letter(system_prompt, input_text, self.response, correction_loops)
            self.response += letter
            print(f"Current response: {self.response}")
        return self.response

    def next_letter(self, system_prompt: str, input_text: str, current_response: str, correction_loops: int) -> str:
        print(f"\n[x] Determining next letter. Current response so far: '{current_response}'")

        response = self.client.system_one(
            state={
                "system_prompt": system_prompt,
                "text": input_text,
                "response": current_response,
            },
            questions=questions(),
        )

        top_5 = top_letters(response.nouls)
        print(f"Probabilities[0]: {top_5}")

        next_letter = best_letter(response.nouls)

        for correction_loop in range(correction_loops):
            print(f"Correction loop {correction_loop + 1}")

            correction_response = self.client.system_one(
                state={
                    "system_prompt": system_prompt,
                    "text": input_text,
                    "response": current_response,
                    "previous_analysis": response.nouls,
                },
                questions=questions([l for l, _ in top_5]),
            )
            top_5_correction = top_letters(correction_response.nouls)
            print(f"Probabilities[{correction_loop + 1}]: {top_5_correction}")
            next_letter_correction = best_letter(correction_response.nouls)
            if next_letter_correction != next_letter:
                print(f"Correction applied: {next_letter} -> {next_letter_correction}")
                next_letter = next_letter_correction

        match next_letter:
            case Token.WHITESPACE:
                next_letter = " "
            case Token.DOT:
                next_letter = "."
            case Token.QUESTION_MARK:
                next_letter = "?"
            case Token.FINISHED:
                self.finished = True
                next_letter = ""
            case Token.BACKSPACE:
                self.response = current_response[:-1]
                next_letter = ""
        # top_5_letters_with_p = sorted(second_response.nouls, key=lambda l: second_response.nouls[l].noul, reverse=True)[:5]
        # print(f"Top 5 letters with probabilities: {[(letter, second_response.nouls[letter].noul) for letter in top_5_letters_with_p]}")
        # second_next_letter = max(second_response.nouls, key=lambda l: second_response.nouls[l].noul)

        # if second_next_letter != next_letter:
        #     print(f"Correction applied: {next_letter} -> {second_next_letter}")
        #     next_letter = second_next_letter

        # Determine the next letter based on the highest probability
        return next_letter


def main() -> None:
    client_instance = Client()
    response = client_instance.complete(
        system_prompt="Follow the instruction in `text`. Respond one letter at the time. Lowercase only.",
        input_text="Respond with: 'hello world!'",
        letters_count=15,
        correction_loops=3,
    )
    print(response)


if __name__ == "__main__":
    main()
