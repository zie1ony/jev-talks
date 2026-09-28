"""Two-phase composition with Jev: scout, then arrange.

Phase 1 (scouting) asks, for every vocabulary word, whether it could be one of
the next LOOKAHEAD words of the answer. Phase 2 (arranging) takes the TOP_WORDS
best scouts, builds every arrangement of 1..LENGTH of them, and judges each as
the possible next move — or removes the last word, skips the round, or declares
the answer complete.

Reads TYPESAFE_API_KEY from .env. Usage: uv run main.py
"""

from itertools import permutations
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, NoulAnswer

load_dotenv()

GATEWAY_URL = "https://api.typesafe.ai"

N_LAST_ACTIONS = 10

# User-defined game parameters.
LOOKAHEAD = 5   # scouting horizon: could the word be one of the next LOOKAHEAD words
TOP_WORDS = 20  # how many scouted words advance to the arranging phase
LENGTH = 2      # max words per placed arrangement (raising to 3 adds P(20,3)=6840 candidates)


class Token:
    FINISHED = "end_of_text"
    REMOVE_WORD = "remove_word"
    NONE_FITS = "none_of_them_fits"


# Basic English 850 word list; the vocabulary being scouted.
all_words = Path(__file__).with_name("basic-english-850.txt").read_text().lower().split()


def append_words(text: str, phrase: str) -> str:
    return f"{text} {phrase}" if text else phrase


def remove_last_words(text: str, n: int) -> str:
    return " ".join(text.split()[:-n])


def scout_questions() -> dict[str, Noul]:
    return {
        word: Noul(instructions=f"Could {word!r} be one of the next {LOOKAHEAD} words of the answer?")
        for word in all_words
    }


def arrangement_candidates(scouted: list[str]) -> list[str]:
    phrases = []
    for length in range(1, LENGTH + 1):
        phrases.extend(" ".join(combo) for combo in permutations(scouted, length))
    return phrases


def arrangement_questions(
    phrases: list[str],
    current_response: str,
    only: list[str] | None = None,
) -> dict[str, Noul]:
    questions_dict = {}
    for phrase in phrases:
        extended = append_words(current_response, phrase)
        questions_dict[phrase] = Noul(
            instructions=(
                f"Placing {phrase!r} would change the answer to {extended!r}."
                " Would that be a good continuation of the answer that `text` asks for?"
            )
        )

    after_one = remove_last_words(current_response, 1)
    questions_dict[Token.REMOVE_WORD] = Noul(
        instructions=(
            f"The answer so far is {current_response!r}."
            f" Should the last word be taken off, leaving {after_one!r}?"
        ),
        criteria={
            "true": "The last word is wrong; removing it improves the answer.",
            "false": "The answer as it stands is a good start, or it is empty.",
        },
    )
    questions_dict[Token.NONE_FITS] = Noul(
        instructions=(
            "Do none of the candidate sequences in `candidates` fit as a good"
            " continuation of `response`? If so, nothing is placed this round"
            " and the vocabulary is scouted again."
        ),
        criteria={
            "true": "Every candidate would make the answer worse; better to place nothing this round.",
            "false": "At least one candidate in `candidates` is a genuinely good continuation.",
        },
    )
    questions_dict[Token.FINISHED] = Noul(
        instructions=(
            f"The answer so far is {current_response!r}."
            " Is it already a complete, good answer to `text`, so composing should stop?"
        ),
        criteria={
            "true": "The answer is complete and good; placing more words would make it worse.",
            "false": "The answer is empty, unfinished, or still contains a wrong word.",
        },
    )

    if only is not None:
        questions_dict = {k: v for k, v in questions_dict.items() if k in only}

    return questions_dict


def top_answers(response: dict[str, NoulAnswer], n: int) -> list[tuple[str, float]]:
    ranked = sorted(response, key=lambda k: response[k].noul, reverse=True)[:n]
    return [(key, response[key].noul) for key in ranked]


def best_answer(response: dict[str, NoulAnswer]) -> str:
    return max(response, key=lambda k: response[k].noul)


class Client:
    def __init__(self):
        self.client = TypeSafeClient(base_url=GATEWAY_URL)
        self.response = ""
        self.finished = False
        self.last_actions = []

    def complete(self, system_prompt: str, input_text: str, steps_count: int, correction_loops: int) -> str:
        for _ in range(steps_count):
            placed = self.next_move(system_prompt, input_text, self.response, correction_loops)
            if self.finished:
                print("Answer declared complete.")
                break
            self.response += placed
            print(f"Current response: {self.response}")
        return self.response

    def next_move(self, system_prompt: str, input_text: str, current_response: str, correction_loops: int) -> str:
        print(f"\n[x] Determining next move. Answer so far: '{current_response}'")

        scout_response = self.client.system_one(
            state={
                "system_prompt": system_prompt,
                "phase": "scouting: rate each word's chance to appear among the next words",
                "text": input_text,
                "response": current_response,
                "last_actions": self.last_actions,
            },
            questions=scout_questions(),
        )
        scouted_with_p = top_answers(scout_response.nouls, TOP_WORDS)
        scouted = [word for word, _ in scouted_with_p]
        print(f"Scouted words: {scouted_with_p}")

        phrases = arrangement_candidates(scouted)
        state = {
            "system_prompt": system_prompt,
            "phase": "arranging: choose the best next move",
            "text": input_text,
            "response": current_response,
            "scouted_words": scouted,
            "candidates": phrases,
            "other_options": [
                "remove the last word",
                "none of them fits (place nothing this round)",
                "the answer is complete",
            ],
            "last_actions": self.last_actions,
        }

        response = self.client.system_one(
            state=state,
            questions=arrangement_questions(phrases, current_response),
        )

        top = top_answers(response.nouls, 10)
        print(f"Probabilities[0]: {top}")

        choice = top[0][0]
        choice_probability = top[0][1]

        if choice_probability <= 0.9:
            for correction_loop in range(correction_loops):
                previous_analysis = {}
                for key, probability in top:
                    previous_analysis[key] = probability

                response = self.client.system_one(
                    state={
                        **state,
                        "system_prompt": system_prompt + (
                            " This is a second, verification pass on the same move decision."
                            " `previous_analysis` maps the previously top-rated moves to their estimated"
                            " probability of being the correct one. Use it as a prior: confirm it where it"
                            " holds up and overrule it where it looks wrong."
                        ),
                        "previous_analysis": previous_analysis,
                    },
                    questions=arrangement_questions(
                        phrases,
                        current_response,
                        only=[key for key, _ in top],
                    ),
                )
                top = top_answers(response.nouls, 10)
                print(f"Probabilities[{correction_loop + 1}]: {top}")
                choice = best_answer(response.nouls)

        match choice:
            case Token.FINISHED:
                self.finished = True
                placed = ""
            case Token.REMOVE_WORD:
                self.response = remove_last_words(current_response, 1)
                placed = ""
            case Token.NONE_FITS:
                placed = ""
            case phrase:
                placed = f" {phrase}" if current_response else phrase

        if placed:
            action = f"placed {placed.strip()!r}, the answer now reads {current_response + placed!r}"
        elif self.finished:
            action = f"declared the answer complete: {current_response!r}"
        elif choice == Token.NONE_FITS:
            action = "placed nothing — none of the scouted arrangements fit"
        else:
            action = f"took off the last word, the answer now reads {self.response!r}"
        self.last_actions = (self.last_actions + [action])[-N_LAST_ACTIONS:]

        return placed


def main() -> None:
    client_instance = Client()
    response = client_instance.complete(
        system_prompt=(
            "You are composing an answer to `text` word by word, in two repeating phases."
            " In the scouting phase, every vocabulary word is rated: could it be one of the next"
            f" {LOOKAHEAD} words of the answer? In the arranging phase, the {TOP_WORDS} best-rated"
            " words (`scouted_words`) are combined into candidate sequences (`candidates`), and each"
            " is judged as a possible next move."
            " `response` is the answer built so far; a space is added automatically when placing."
            " The moves: place one candidate sequence at the end of `response`, take the last word"
            " off, place nothing this round because none of the candidates fits, or declare the"
            " answer complete. Judge every question independently: is it the best next move toward a"
            " short, good answer to `text`?"
            " The candidates come from words scouted as likely, so usually one of them fits; place"
            " nothing only when every arrangement genuinely reads wrong."
            " `last_actions` lists the most recent moves, oldest first. Use it to stay consistent and"
            " avoid loops: keep extending the sentence already built; if words were placed and then"
            " removed, do not place them again; if the last round placed nothing, favor different"
            " words now."
        ),
        input_text="What is the most successful scam in history?",
        steps_count=30,
        correction_loops=1,
    )
    print(response)


if __name__ == "__main__":
    main()
