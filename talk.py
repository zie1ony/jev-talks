"""Make a model that cannot write talk, one judgment at a time.

Jev, TypeSafe's System One model, does not generate text. It takes some state
plus typed questions and returns calibrated probabilities. This script composes
an answer anyway, as a game played in rounds:

  1. scout    ask, for each of 3,000 words, "could this be one of the next few
              words of the answer?" and keep the 30 most likely
  2. arrange  turn those into ~900 one- and two-word phrases and ask, for each,
              "is appending this a good next move?", next to three other moves:
              take the last word back, pass, or finish
  3. apply    the single most probable move, then play the next round

Usage:  uv run talk.py "Why is the sky blue?" [-v] [--vocab FILE]
Needs TYPESAFE_API_KEY, from the environment or a .env file.
"""

import argparse
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, permutations
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

# Settings --------------------------------------------------------------------

VOCABULARY = Path(__file__).with_name("top-english-3000.txt")
LOOKAHEAD = 3  # scouting asks: could the word be one of the next LOOKAHEAD words?
TOP_WORDS = 30  # scouted words that advance to arranging
PHRASE = 2  # longest phrase placed in one move; 30 words make 30 + 870 phrases
TAIL = 10  # words of the answer quoted in each phrase question, to bound the request size
BATCH = 1000  # words per scouting request, to fit Jev's 64k-token window
MEMORY = 10  # recent moves the model gets to see, so it does not loop
MAX_ROUNDS = 50

PHRASES = sum(math.perm(TOP_WORDS, n) for n in range(1, PHRASE + 1))  # candidate phrases per round
REMOVE, PASS, FINISH = "<take the last word back>", "<pass>", "<finish>"
OTHER_MOVES = ["take the last word back", "pass: place nothing this round", "finish: the answer is complete"]

RULES = (
    "You are composing an answer to `question` word by word, in rounds of two phases."
    f" Scouting: every vocabulary word is rated on whether it could be one of the next {LOOKAHEAD} words."
    f" Arranging: the {TOP_WORDS} best-rated words (`scouted_words`) are combined into short phrases"
    " (`candidates`) and each is judged as a possible next move, next to three other moves: take the"
    " last word back, pass this round, or finish."
    " `answer` is the answer so far; a space is added automatically when a phrase is placed."
    " Judge every question on its own: is this the best next move toward a short, good answer?"
    " The candidates were scouted as likely, so usually one of them fits; pass only when all read wrong."
    " `recent_moves` lists the latest moves, oldest first. Stay consistent and avoid loops: keep"
    " extending the sentence already built, do not place words that were taken back, and after a"
    " pass favor different words."
)

VERBOSE = False


# The game --------------------------------------------------------------------


class Composer:
    def __init__(self, jev: "Jev", vocabulary: list[str], question: str):
        self.jev, self.vocabulary, self.question = jev, vocabulary, question
        self.answer: list[str] = []
        self.history: list[str] = []

    def run(self) -> str:
        for n in range(1, MAX_ROUNDS + 1):
            log(f"\nround {n}, answer so far: {self.text!r}")
            with Spinner(self.text, f"scouting {len(self.vocabulary):,} words") as spinner:
                words = self.scout()
                spinner.status = f"judging {PHRASES:,} phrases"
                move = self.arrange(words)
            if move == FINISH:
                break
            self.apply(move)
        return self.text

    def scout(self) -> list[str]:
        """Rate every vocabulary word, in parallel batches, and keep the TOP_WORDS best."""
        state = self.state(phase="scouting: rate each word's chance to appear among the next words")
        batches = [self.vocabulary[i : i + BATCH] for i in range(0, len(self.vocabulary), BATCH)]
        with ThreadPoolExecutor() as pool:
            results = pool.map(lambda words: self.jev(state, scouting_questions(words)), batches)
        probabilities = {word: p for result in results for word, p in result.items()}
        words = top(probabilities, TOP_WORDS)
        log("  scouted:", show(probabilities, words[:10]), "...")
        return words

    def arrange(self, words: list[str]) -> str:
        """Judge every phrase made of the scouted words, and the three other moves; pick the best."""
        phrases = [" ".join(p) for n in range(1, PHRASE + 1) for p in permutations(words, n)]
        state = self.state(
            phase="arranging: choose the best next move", scouted_words=words, candidates=phrases, other_moves=OTHER_MOVES
        )
        probabilities = self.jev(state, arranging_questions(self.answer, phrases))
        moves = top(probabilities, 5)
        log("  moves:  ", show(probabilities, moves))
        return moves[0]

    def apply(self, move: str) -> None:
        if move == REMOVE:
            taken = self.answer.pop()
            self.remember(f"took back {taken!r}; the answer now reads {self.text!r}")
        elif move == PASS:
            self.remember("passed: none of the candidates fit")
        else:
            self.answer += move.split()
            self.remember(f"placed {move!r}; the answer now reads {self.text!r}")

    def remember(self, what: str) -> None:
        log("  ->", what)
        self.history = (self.history + [what])[-MEMORY:]

    def state(self, **extra) -> dict:
        return {"rules": RULES, "question": self.question, "answer": self.text, "recent_moves": self.history, **extra}

    @property
    def text(self) -> str:
        return " ".join(self.answer)


# The questions ---------------------------------------------------------------


def scouting_questions(words: list[str]) -> dict[str, Noul]:
    return {word: Noul(instructions=f"Could {word!r} be one of the next {LOOKAHEAD} words of the answer?") for word in words}


def arranging_questions(answer: list[str], phrases: list[str]) -> dict[str, Noul]:
    def outcome(phrase: str) -> str:
        if len(answer) <= TAIL:
            return f"would change the answer to {' '.join(answer + [phrase])!r}"
        return f"would make the answer end with {'... ' + ' '.join(answer[-TAIL:] + [phrase])!r}"

    questions = {
        phrase: Noul(
            instructions=f"Placing {phrase!r} {outcome(phrase)}."
            " Would that be a good continuation of the answer that `question` asks for?"
        )
        for phrase in phrases
    }
    questions[PASS] = Noul(
        instructions="Does none of the phrases in `candidates` fit as a good continuation of `answer`,"
        " so that nothing should be placed this round?",
        criteria={
            "true": "Every candidate would make the answer worse; better to place nothing this round.",
            "false": "At least one candidate is a genuinely good continuation.",
        },
    )
    if answer:
        text, shorter = " ".join(answer), " ".join(answer[:-1])
        questions[REMOVE] = Noul(
            instructions=f"The answer so far is {text!r}. Should the last word be taken back, leaving {shorter!r}?",
            criteria={
                "true": "The last word is wrong; removing it improves the answer.",
                "false": "The answer so far is a good start.",
            },
        )
        questions[FINISH] = Noul(
            instructions=f"The answer so far is {text!r}. Is it already a complete, good answer to `question`,"
            " so composing should stop?",
            criteria={
                "true": "The answer is complete and good; more words would make it worse.",
                "false": "The answer is unfinished or still contains a wrong word.",
            },
        )
    return questions


def top(probabilities: dict[str, float], n: int) -> list[str]:
    return sorted(probabilities, key=probabilities.get, reverse=True)[:n]


def show(probabilities: dict[str, float], keys: list[str]) -> str:
    return ", ".join(f"{key} {probabilities[key]:.2f}" for key in keys)


def log(*parts: object) -> None:
    if VERBOSE:
        print(*parts, file=sys.stderr, flush=True)


def animated() -> bool:
    """Redraw the answer in place only on a real terminal that is not showing the verbose trace."""
    return sys.stdout.isatty() and not VERBOSE


# The plumbing ----------------------------------------------------------------


class Jev:
    """The TypeSafe client, reduced to one call that returns each question's probability of yes."""

    def __init__(self):
        self.client = TypeSafeClient()
        self.lock = threading.Lock()
        self.requests = self.tokens = 0

    def __call__(self, state: dict, questions: dict[str, Noul]) -> dict[str, float]:
        response = self.client.system_one(state=state, questions=questions)
        with self.lock:
            self.requests += 1
            self.tokens += response.usage.input_tokens or 0
        return {name: answer.noul for name, answer in response.nouls.items()}


class Spinner:
    """Keep the answer so far on the screen, with a spinning glyph and a status, while Jev thinks."""

    def __init__(self, text: str, status: str):
        self.text, self.status = text, status
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.spin, daemon=True)

    def spin(self) -> None:
        for glyph in cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self.stop.is_set():
                return
            sys.stdout.write(f"\r\x1b[2K{self.text}{' ' if self.text else ''}\x1b[2m{glyph} {self.status}\x1b[0m")
            sys.stdout.flush()
            self.stop.wait(0.1)

    def __enter__(self) -> "Spinner":
        if animated():
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()
            sys.stdout.write(f"\r\x1b[2K{self.text}")
            sys.stdout.flush()


def main() -> None:
    global VERBOSE
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog="Needs TYPESAFE_API_KEY, from the environment or a .env file."
    )
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every round's judgments and the token usage")
    parser.add_argument("--vocab", type=Path, default=VOCABULARY, help="word list to compose from, one word per line")
    args = parser.parse_args()
    VERBOSE = args.verbose

    load_dotenv()
    try:
        jev = Jev()
        composer = Composer(jev, args.vocab.read_text().lower().split(), args.question)
        answer = composer.run()
    except KeyboardInterrupt:
        sys.exit("\n(stopped)")
    except (TypeSafeError, OSError) as error:
        sys.exit(f"error: {error}")
    print(("\r\x1b[2K" if animated() else "") + (answer or "(no answer)"))
    log(f"\n{jev.requests} requests, {jev.tokens:,} input tokens")


if __name__ == "__main__":
    main()
