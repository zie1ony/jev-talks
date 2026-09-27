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
    BACKSPACE_TWO = "backspace_two"
    BACKSPACE_THREE = "backspace_three"

def questions(only_letters: list[str] | None = None) -> dict[str, Noul]:
    # print(f"ONLY LETTERS: {only_letters}")
    questions_dict = {}
    for letter in letters:
        questions_dict[letter] = Noul(
            instructions=f"Should the next character appended to `response` be the letter '{letter}'?"
        )
    questions_dict[Token.WHITESPACE] = Noul(
        instructions="Should the next character appended to `response` be a single space (' ')?"
    )
    questions_dict[Token.DOT] = Noul(
        instructions="Should the next character appended to `response` be a period ('.')?"
    )
    questions_dict[Token.QUESTION_MARK] = Noul(
        instructions="Should the next character appended to `response` be a question mark ('?')?"
    )
    questions_dict[Token.FINISHED] = Noul(
        instructions="Is `response` already the complete, finished answer to the instruction in `text`?",
        criteria={
            "true": "`response` fulfills the instruction in `text` exactly; appending any further character would make it wrong.",
            "false": "`response` is empty, unfinished, or still contains a mistake.",
        },
    )
    questions_dict[Token.BACKSPACE] = Noul(
        instructions="Is the last character of `response` a mistake that should be deleted?",
        criteria={
            "true": "`response` is not a correct start of the answer requested in `text`; deleting its last character brings it closer.",
            "false": "`response` so far is a correct start of the answer requested in `text`, or `response` is empty.",
        },
    )
    questions_dict[Token.BACKSPACE_TWO] = Noul(
        instructions="Are the last two characters of `response` a mistake that should be deleted?",
        criteria={
            "true": "`response` is not a correct start of the answer requested in `text`; deleting its last two characters brings it closer.",
            "false": "`response` so far is a correct start of the answer requested in `text`, or `response` has fewer than two characters.",
        },
    )
    questions_dict[Token.BACKSPACE_THREE] = Noul(
        instructions="Are the last three characters of `response` a mistake that should be deleted?",
        criteria={
            "true": "`response` is not a correct start of the answer requested in `text`; deleting its last three characters brings it closer.",
            "false": "`response` so far is a correct start of the answer requested in `text`, or `response` has fewer than three characters.",
        },
    )

    if only_letters is not None:
        questions_dict = {k: v for k, v in questions_dict.items() if k in only_letters}

    # print(f"Generated questions: {list(questions_dict.keys())}")
    return questions_dict

def top_letters(response: dict[str, NoulAnswer], n: int) -> list[str, float]:
    top_5_letters_with_p = sorted(response, key=lambda l: response[l].noul, reverse=True)[:n]
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

        top = top_letters(response.nouls, 10)
        print(f"Probabilities[0]: {top}")

        next_letter = top[0][0]
        next_letter_probability = top[0][1]

        skip_correction = False
        if next_letter_probability > 0.9:
            skip_correction = True

        if not skip_correction:
            for correction_loop in range(correction_loops):
                print(f"Correction loop {correction_loop + 1}")

                response = self.client.system_one(
                    state={
                        "system_prompt": system_prompt + (
                            " This is a verification pass over a shortlist of candidates for the same next character."
                            " `previous_analysis` holds the first pass's probability for each candidate;"
                            " treat it as a prior and correct it where it looks wrong."
                        ),
                        "text": input_text,
                        "response": current_response,
                        "previous_analysis": response.nouls,
                    },
                    questions=questions([l for l, _ in top]),
                )
                top = top_letters(response.nouls, 10)
                print(f"Probabilities[{correction_loop + 1}]: {top}")
                next_letter = best_letter(response.nouls)

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
            case Token.BACKSPACE_TWO:
                self.response = current_response[:-2]
                next_letter = ""
            case Token.BACKSPACE_THREE:
                self.response = current_response[:-3]
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
        system_prompt=(
            "You are composing `response` one character per step, so that it carries out the instruction in `text`."
            " `response` is the partial draft written so far."
            " Judge the candidates for the single next character. Lowercase letters only."
        ),
        input_text="Respond with: 'hello world!'",
        letters_count=30,
        correction_loops=3,
    )
    print(response)


if __name__ == "__main__":
    main()
