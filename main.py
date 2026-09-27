"""Minimal connectivity check: one Noul through the Vercel AI Gateway.

Reads TYPESAFE_API_KEY from .env. Usage: uv run ask_single_question.py
"""

import json

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, NoulAnswer

load_dotenv()

GATEWAY_URL = "https://api.typesafe.ai"

N_LAST_ACTIONS = 10

# Letters + whitespace
letters = "abcdefghijklmnopqrstuvwxyz1234567890"
class Token:
    WHITESPACE = "whitespace"
    DOT = "dot"
    QUESTION_MARK = "question_mark"
    FINISHED = "end_of_text"
    BACKSPACE = "backspace"
    BACKSPACE_TWO = "backspace_two"
    BACKSPACE_THREE = "backspace_three"

def questions(only_letters: list[str] | None = None, current_response: str = "") -> dict[str, Noul]:
    # print(f"ONLY LETTERS: {only_letters}")
    questions_dict = {}
    for letter in letters:
        appended = current_response + letter
        questions_dict[letter] = Noul(
            instructions=(
                f"The draft answer so far is {current_response!r}."
                f" Appending the letter {letter!r} would make it {appended!r}."
                f" Would {appended!r} be a correct start of the answer that `text` asks for?"
            )
        )

    with_space = current_response + " "
    questions_dict[Token.WHITESPACE] = Noul(
        instructions=(
            f"The draft answer so far is {current_response!r}."
            f" Appending a single space (' ') would make it {with_space!r}."
            f" Would {with_space!r} be a correct start of the answer that `text` asks for?"
        )
    )
    with_dot = current_response + "."
    questions_dict[Token.DOT] = Noul(
        instructions=(
            f"The draft answer so far is {current_response!r}."
            f" Appending a period ('.') would make it {with_dot!r}."
            f" Would {with_dot!r} be a correct start of the answer that `text` asks for?"
        )
    )
    with_question_mark = current_response + "?"
    questions_dict[Token.QUESTION_MARK] = Noul(
        instructions=(
            f"The draft answer so far is {current_response!r}."
            f" Appending a question mark ('?') would make it {with_question_mark!r}."
            f" Would {with_question_mark!r} be a correct start of the answer that `text` asks for?"
        )
    )
    questions_dict[Token.FINISHED] = Noul(
        instructions=(
            f"The draft answer is {current_response!r}."
            " Is it already the complete, finished answer to the instruction in `text`,"
            " so nothing more should be appended?"
        ),
        criteria={
            "true": "The draft fulfills the instruction in `text` exactly; appending any further character would make it wrong.",
            "false": "The draft is empty, unfinished, or contains a mistake.",
        },
    )
    questions_dict[Token.BACKSPACE] = Noul(
        instructions=(
            f"The draft answer is {current_response!r}."
            f" Is its last character wrong, so that deleting just it (leaving {current_response[:-1]!r})"
            " is the correct next edit?"
        ),
        criteria={
            "true": "The last character of the draft is wrong; removing only it leaves a correct start of the requested answer.",
            "false": "The draft as it stands is a correct start of the requested answer, or it is empty.",
        },
    )
    questions_dict[Token.BACKSPACE_TWO] = Noul(
        instructions=(
            f"The draft answer is {current_response!r}."
            f" Are its last two characters both wrong, so that deleting both (leaving {current_response[:-2]!r})"
            " is the correct next edit?"
        ),
        criteria={
            "true": "The last two characters of the draft are both wrong; removing them leaves a correct start of the requested answer.",
            "false": (
                "The draft as it stands is a correct start of the requested answer,"
                " deleting two characters would also remove a correct one,"
                " or the draft has fewer than two characters."
            ),
        },
    )
    questions_dict[Token.BACKSPACE_THREE] = Noul(
        instructions=(
            f"The draft answer is {current_response!r}."
            f" Are its last three characters all wrong, so that deleting all three (leaving {current_response[:-3]!r})"
            " is the correct next edit?"
        ),
        criteria={
            "true": "The last three characters of the draft are all wrong; removing them leaves a correct start of the requested answer.",
            "false": (
                "The draft as it stands is a correct start of the requested answer,"
                " deleting three characters would also remove a correct one,"
                " or the draft has fewer than three characters."
            ),
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
        self.last_actions = []

    def complete(self, system_prompt: str, input_text: str, letters_count: int, correction_loops: int) -> str:
        for _ in range(letters_count):
            letter = self.next_letter(system_prompt, input_text, self.response, correction_loops)
            if self.finished:
                print("End of text — response complete.")
                break
            self.response += letter
            print(f"Current response: {self.response}")
            # print(f"Last actions: {self.last_actions}")
        return self.response

    def next_letter(self, system_prompt: str, input_text: str, current_response: str, correction_loops: int) -> str:
        print(f"\n[x] Determining next letter. Current response so far: '{current_response}'")

        response = self.client.system_one(
            state={
                "system_prompt": system_prompt,
                "text": input_text,
                "response": current_response,
                "last_actions": self.last_actions,
            },
            questions=questions(current_response=current_response),
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
                # print(f"Correction loop {correction_loop + 1}")

                previous_analysis = {}
                for l, p in top:
                    previous_analysis[l] = p
                # print(f"Previous analysis: {previous_analysis}")

                response = self.client.system_one(
                    state={
                        "system_prompt": system_prompt + (
                            " This is a second, verification pass on the same decision."
                            " `previous_analysis` maps the previously top-rated edits to their estimated"
                            " probability of being the correct one. Use it as a prior: confirm it where it"
                            " holds up and overrule it where it looks wrong."
                        ),
                        "text": input_text,
                        "response": current_response,
                        "last_actions": self.last_actions,
                        "previous_analysis": previous_analysis,
                    },
                    questions=questions(current_response=current_response),
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

        if next_letter:
            action = f"appended {next_letter!r}, draft became {current_response + next_letter!r}"
        elif self.finished:
            action = f"declared the draft {current_response!r} finished"
        else:
            deleted = len(current_response) - len(self.response)
            action = f"deleted the last {deleted} character(s), draft became {self.response!r}"
        self.last_actions = (self.last_actions + [action])[-N_LAST_ACTIONS:]

        return next_letter


def main() -> None:
    client_instance = Client()
    response = client_instance.complete(
        system_prompt=(
            "You are writing an answer one character at a time, so that it ends up carrying out the"
            " instruction in `text` exactly. The draft written so far is in `response` and is repeated"
            " inside each question. Each question proposes one possible next edit: append a specific"
            " character, delete recent characters, or declare the answer finished. Judge every proposal"
            " independently on whether it is the correct next edit. Lowercase letters only."
            " `last_actions` lists the edits already applied to the draft, oldest first. Use it to stay"
            " consistent and to avoid loops: keep extending the wording the draft has already committed to;"
            " if `last_actions` shows an edit was applied and then deleted again, that edit was wrong —"
            " do not endorse it again; and if the draft keeps returning to the same text, favor a"
            " different edit than the one tried before."
        ),
        input_text="What are you?",
        letters_count=1300,
        correction_loops=0,
    )
    print(response)


if __name__ == "__main__":
    main()
