"""Minimal connectivity check: one Noul through the Vercel AI Gateway.

Reads TYPESAFE_API_KEY from .env. Usage: uv run ask_single_question.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, NoulAnswer

load_dotenv()

GATEWAY_URL = "https://api.typesafe.ai"

N_LAST_ACTIONS = 10

class Token:
    DOT = "dot"
    QUESTION_MARK = "question_mark"
    FINISHED = "end_of_text"
    BACKSPACE = "backspace"
    BACKSPACE_TWO = "backspace_two"
    BACKSPACE_THREE = "backspace_three"


_TOKEN_NAMES = {
    Token.DOT, Token.QUESTION_MARK, Token.FINISHED,
    Token.BACKSPACE, Token.BACKSPACE_TWO, Token.BACKSPACE_THREE,
}

# Top 3000 English words by frequency (Google corpus), plurals included; each
# word is one key on the word keyboard. Words that collide with a control-token
# name are dropped so their questions are not overwritten.
words = [
    word
    for word in Path(__file__).with_name("top-english-3000.txt").read_text().lower().split()
    if word not in _TOKEN_NAMES
]


def append_word(text: str, word: str) -> str:
    return f"{text} {word}" if text else word


def remove_last_words(text: str, n: int) -> str:
    return " ".join(text.split()[:-n])


def questions(
    only_words: list[str] | None = None, current_response: str = "", detailed: bool = False
) -> dict[str, Noul]:
    questions_dict = {}
    for word in words:
        if detailed:
            appended = append_word(current_response, word)
            instructions = (
                f"Pressing the {word!r} word key would change the text to {appended!r}."
                " Would that be a correct start of the answer that `text` asks for?"
            )
        else:
            instructions = f"Should the {word!r} word key be pressed next?"
        questions_dict[word] = Noul(instructions=instructions)

    with_dot = current_response + "."
    questions_dict[Token.DOT] = Noul(
        instructions=(
            f"Pressing the period key ('.') would change the text to {with_dot!r}."
            " Would that be a correct start of the answer that `text` asks for?"
        )
    )
    with_question_mark = current_response + "?"
    questions_dict[Token.QUESTION_MARK] = Noul(
        instructions=(
            f"Pressing the question-mark key ('?') would change the text to {with_question_mark!r}."
            " Would that be a correct start of the answer that `text` asks for?"
        )
    )
    questions_dict[Token.FINISHED] = Noul(
        instructions=(
            f"The text field shows {current_response!r}."
            " Should the enter key be pressed now, submitting this text as the final answer to `text`?"
        ),
        criteria={
            "true": "The text is the complete, correct answer to `text`; nothing more should be typed.",
            "false": "The text is empty or unfinished, or still contains a wrong word that must be fixed first.",
        },
    )
    after_one = remove_last_words(current_response, 1)
    questions_dict[Token.BACKSPACE] = Noul(
        instructions=(
            f"The text field shows {current_response!r}."
            f" Pressing word-backspace once would delete the last word, leaving {after_one!r}."
            " Should word-backspace be pressed exactly once?"
        ),
        criteria={
            "true": "Only the last word is wrong; deleting just it leaves a correct start of the intended answer.",
            "false": "The text as it stands is a correct start of the intended answer, or it is empty.",
        },
    )
    after_two = remove_last_words(current_response, 2)
    questions_dict[Token.BACKSPACE_TWO] = Noul(
        instructions=(
            f"The text field shows {current_response!r}."
            f" Pressing word-backspace twice would delete the last two words, leaving {after_two!r}."
            " Should word-backspace be pressed exactly twice?"
        ),
        criteria={
            "true": "The last two words are both wrong; deleting both leaves a correct start of the intended answer.",
            "false": (
                "The text as it stands is a correct start of the intended answer,"
                " the second word-backspace press would delete a correct word,"
                " or the text has fewer than two words."
            ),
        },
    )
    after_three = remove_last_words(current_response, 3)
    questions_dict[Token.BACKSPACE_THREE] = Noul(
        instructions=(
            f"The text field shows {current_response!r}."
            f" Pressing word-backspace three times would delete the last three words, leaving {after_three!r}."
            " Should word-backspace be pressed exactly three times?"
        ),
        criteria={
            "true": "The last three words are all wrong; deleting all three leaves a correct start of the intended answer.",
            "false": (
                "The text as it stands is a correct start of the intended answer,"
                " the extra word-backspace presses would delete a correct word,"
                " or the text has fewer than three words."
            ),
        },
    )

    if only_words is not None:
        questions_dict = {k: v for k, v in questions_dict.items() if k in only_words}

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
        print(f"\n[x] Determining next word. Current response so far: '{current_response}'")

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
                            " This is a second, verification pass on the same key-press decision."
                            " `previous_analysis` maps the previously top-rated key presses to their estimated"
                            " probability of being the correct one. Use it as a prior: confirm it where it"
                            " holds up and overrule it where it looks wrong."
                        ),
                        "text": input_text,
                        "response": current_response,
                        "last_actions": self.last_actions,
                        "previous_analysis": previous_analysis,
                    },
                    questions=questions(
                        only_words=[l for l, _ in top],
                        current_response=current_response,
                        detailed=True,
                    ),
                )
                top = top_letters(response.nouls, 10)
                print(f"Probabilities[{correction_loop + 1}]: {top}")
                next_letter = best_letter(response.nouls)

        match next_letter:
            case Token.DOT:
                next_letter = "."
            case Token.QUESTION_MARK:
                next_letter = "?"
            case Token.FINISHED:
                self.finished = True
                next_letter = ""
            case Token.BACKSPACE:
                self.response = remove_last_words(current_response, 1)
                next_letter = ""
            case Token.BACKSPACE_TWO:
                self.response = remove_last_words(current_response, 2)
                next_letter = ""
            case Token.BACKSPACE_THREE:
                self.response = remove_last_words(current_response, 3)
                next_letter = ""
            case word:
                next_letter = f" {word}" if current_response else word

        if next_letter:
            action = f"pressed {next_letter.strip()!r}, the text field now shows {current_response + next_letter!r}"
        elif self.finished:
            action = f"pressed enter, submitting {current_response!r}"
        else:
            deleted = len(current_response.split()) - len(self.response.split())
            action = f"pressed word-backspace {deleted} time(s), the text field now shows {self.response!r}"
        self.last_actions = (self.last_actions + [action])[-N_LAST_ACTIONS:]

        return next_letter


def main() -> None:
    client_instance = Client()
    response = client_instance.complete(
        system_prompt=(
            "You are typing an answer on a word keyboard into a text field, one key press at a time,"
            " so that the submitted text carries out the instruction in `text` exactly. Each word key"
            " types one whole word; a space is inserted automatically between words. The text typed so"
            " far is in `response`; the backspace and enter questions show the exact text they would"
            " leave or submit."
            " Each question asks about one possible next key press: a word key, the period or"
            " question-mark key (these attach to the last word, with no space), word-backspace"
            " (deleting the last whole word — pressed once, twice, or three times), or enter to submit"
            " the answer. Judge every proposed key press independently on whether it is the correct"
            " next press."
            " `last_actions` lists the most recent key presses, oldest first. Use it to stay consistent"
            " and to avoid loops: keep extending the wording already typed; if a word was typed and then"
            " backspaced away, it was wrong — do not press it again; and if the text keeps returning to"
            " the same state, favor a different key than the one tried before."
            f" Possible words: {' '.join(words)}"
        ),
        input_text="What is the most successful scam in history?",
        letters_count=100,
        correction_loops=5,
    )
    print(response)


if __name__ == "__main__":
    main()
