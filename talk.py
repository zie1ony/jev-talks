"""Make a model that cannot write talk, one judgment at a time.

Jev, TypeSafe's System One model, does not generate text. It takes some state
plus typed questions and returns calibrated probabilities. This script composes
an answer anyway, as a game played in rounds:

  1. scout    show the vocabulary in groups of 250 and ask each group "which of
              these could be one of the next few words of the answer?"; keep
              the 30 most probable words
  2. arrange  build 240 one- and two-word phrases from them and ask, for each,
              "placed at the end of the answer, would this read as natural,
              correct English and lead to a good answer?", next to two more
              yes/no questions: take the last word back, or finish
  3. apply    the most probable move, then play the next round

Scouting for the next round runs in the background while the current round is
judged, so a round is one request of about 10,000 tokens and half a second.
Three answers are composed in parallel, each from a differently grouped
vocabulary, and Jev grades them on a rubric to keep the best one.

Usage:  uv run talk.py "Why is the sky blue?" [-v] [--tries N] [--vocab FILE]
Needs TYPESAFE_API_KEY, from the environment or a .env file.
"""

import argparse
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, permutations
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Question, Score, TypeSafeClient, TypeSafeError

# Settings --------------------------------------------------------------------

VOCABULARY = Path(__file__).with_name("top-english-3000.txt")
LOOKAHEAD = 3  # scouting asks which word could be one of the next LOOKAHEAD words
GROUP = 250  # words per scouting question; a Choice takes at most 255 options
TOP_WORDS = 30  # scouted words that advance to arranging
PAIRS_FROM = 15  # two-word phrases are built from the best PAIRS_FROM of them: 30 + 15 * 14 = 240 phrases
TAIL = 10  # words of the answer quoted in each phrase question, to bound the request size
STALE = 2  # rounds a scouting result may be reused before waiting for a fresh one
MEMORY = 10  # recent moves the model gets to see, so it does not loop
LONG = 15  # from this many words on, the state notes that the answer is long and should be finished
MAX_ROUNDS = 30  # every run that went past 30 rounds in testing was garbage
TRIES = 3  # answers composed in parallel; the best by Jev's grade is kept

PHRASES = TOP_WORDS + PAIRS_FROM * (PAIRS_FROM - 1)
REMOVE, FINISH = "<take the last word back>", "<finish>"

RULES = (
    "You are composing an answer to `question` word by word, in rounds of two phases."
    " Scouting: the vocabulary is shown in groups, and each group is asked which of its words could be one"
    f" of the next {LOOKAHEAD} words of the answer."
    f" Arranging: the {TOP_WORDS} best-rated words are combined into short phrases, and each phrase is judged"
    " as a possible next move, next to two other moves: take the last word back, or finish."
    " `answer` is the answer so far; a space is added automatically when a phrase is placed."
    " Judge every question on its own: is this the best next move toward a short, good answer?"
    " Answer in one complete sentence of several words; a single word or a fragment is never a finished answer."
    " `recent_moves` lists the latest moves, oldest first. Stay consistent and avoid loops: keep extending"
    " the sentence already built, and do not place words that were taken back."
)

VERBOSE = False


# The game --------------------------------------------------------------------


class Composer:
    def __init__(self, jev: "Jev", vocabulary: list[str], question: str, label: str = ""):
        self.jev, self.vocabulary, self.question, self.label = jev, vocabulary, question, label
        self.answer: list[str] = []
        self.history: list[str] = []
        self.rejected: set[str] = set()  # answers that were taken back from; never retried

    def run(self, spinner: "Spinner | None" = None) -> str:
        """Play rounds until the model finishes. Scouting overlaps the previous round's arranging."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            scouting = pool.submit(self.scout, self.text)
            candidates, age = [], STALE
            for n in range(1, MAX_ROUNDS + 1):
                self.log(("" if self.label else "\n") + f"round {n}, answer so far: {self.text!r}")
                if scouting and (scouting.done() or age >= STALE):
                    self.tell(spinner, f"scouting {len(self.vocabulary):,} words")
                    candidates, age, scouting = scouting.result(), 0, None
                self.tell(spinner, f"judging {PHRASES} phrases")
                move = self.arrange(candidates)
                if move == FINISH:
                    break
                self.apply(move)
                self.tell(spinner, None)
                age += 1
                if scouting is None:
                    scouting = pool.submit(self.scout, self.text)
        return self.text

    def scout(self, answer: str) -> list[str]:
        """One request: a multiple-choice question per group of words. Keep the TOP_WORDS most probable."""
        state = self.state(answer, phase="scouting: which words could appear among the next words")
        groups = [self.vocabulary[i : i + GROUP] for i in range(0, len(self.vocabulary), GROUP)]
        answers = self.jev(state, {f"group{i}": scouting_question(group) for i, group in enumerate(groups)})
        probabilities = {word: p for answer in answers.values() for word, p in answer.probabilities.items()}
        words = top(probabilities, TOP_WORDS)
        self.log("  scouted:", show(probabilities, words[:10]), "...")
        return words

    def arrange(self, words: list[str]) -> str:
        """One request: a yes/no question per phrase, plus taking back and finishing. The most probable wins."""
        phrases = words + [" ".join(pair) for pair in permutations(words[:PAIRS_FROM], 2)]
        phrases = [p for p in phrases if f"{self.text} {p.split()[0]}".strip() not in self.rejected]
        state = self.state(self.text, phase="arranging: choose the best next move")
        answers = self.jev(state, arranging_questions(self.answer, phrases))
        probabilities = {name: answer.noul for name, answer in answers.items()}
        moves = top(probabilities, 5)
        self.log("  moves:  ", show(probabilities, moves))
        return moves[0]

    def apply(self, move: str) -> None:
        if move == REMOVE:
            self.rejected.add(self.text)
            taken = self.answer.pop()
            self.remember(f"took back {taken!r}; the answer now reads {self.text!r}")
        else:
            self.answer += move.split()
            self.remember(f"placed {move!r}; the answer now reads {self.text!r}")

    def tell(self, spinner: "Spinner | None", status: str | None) -> None:
        """Update the terminal animation, when there is one, with the answer so far and what Jev is doing."""
        if spinner:
            spinner.text = self.text
            if status:
                spinner.status = status

    def log(self, *parts: object) -> None:
        log(*((f"[{self.label}]",) if self.label else ()), *parts)

    def remember(self, what: str) -> None:
        self.log("  ->", what)
        self.history = (self.history + [what])[-MEMORY:]

    def state(self, answer: str, **extra) -> dict:
        state = {"rules": RULES, "question": self.question, "answer": answer, "recent_moves": self.history, **extra}
        if len(answer.split()) >= LONG:  # Jev cannot count, but it reads a plain statement literally
            state["length"] = (
                f"The answer already has {len(answer.split())} words, which is long. Finish it as soon as it is"
                " a complete sentence; add a word only if the sentence cannot end without it."
            )
        return state

    @property
    def text(self) -> str:
        return " ".join(self.answer)


# The questions ---------------------------------------------------------------


def scouting_question(words: list[str]) -> Choice:
    return Choice(
        instructions=f"Which of these words is the most likely to be one of the next {LOOKAHEAD} words of the answer?"
        " Every option is a vocabulary word to be placed literally, not a meta answer.",
        criteria={word: None for word in words},
    )


def arranging_questions(answer: list[str], phrases: list[str]) -> dict[str, Question]:
    def outcome(phrase: str) -> str:
        if len(answer) <= TAIL:
            return f"would change the answer to {' '.join(answer + [phrase])!r}"
        return f"would make the answer end with {'... ' + ' '.join(answer[-TAIL:] + [phrase])!r}"

    part = "a good continuation of" if answer else "a good beginning of"
    questions: dict[str, Question] = {
        phrase: Noul(
            instructions=f"Placing {phrase!r} {outcome(phrase)}. Would the answer then read as natural, correct"
            f" English, and be {part} a complete, informative answer to `question`?"
        )
        for phrase in phrases
    }
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
                "true": "The answer is a complete sentence of several words that fully answers the question;"
                " more words would make it worse.",
                "false": "The answer is a single word, a fragment, unfinished, or still contains a wrong word.",
            },
        )
    return questions


def grading_question() -> Score:
    return Score(
        instructions="Grade `answer` as an answer to `question`. It is lowercase and has no punctuation by"
        " construction; ignore that and judge the words.",
        criteria=[
            "not an answer: empty, nonsense, repetitive, or unrelated to the question",
            "poor: a fragment that stops mid-thought, or a wrong or contradictory claim",
            "fair: on topic and understandable, but incomplete, awkward, or too vague to be useful",
            "good: a complete, natural sentence that answers the question sensibly",
            "excellent: a complete, natural, informative sentence a thoughtful person might have written",
        ],
    )


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
    """The TypeSafe client, reduced to one call that returns the answers by question name."""

    def __init__(self):
        self.client = TypeSafeClient()
        self.lock = threading.Lock()
        self.requests = self.tokens = 0

    def __call__(self, state: dict, questions: dict[str, Question]) -> dict:
        response = self.client.system_one(state=state, questions=questions)
        with self.lock:
            self.requests += 1
            self.tokens += response.usage.input_tokens or 0
        return response.answers


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


def compose(jev: Jev, words: list[str], question: str, tries: int) -> str:
    """Compose `tries` answers in parallel, each from a differently grouped vocabulary, and keep the best."""
    def one(seed: int, spinner: Spinner | None) -> str:
        vocabulary = list(words)
        random.Random(seed).shuffle(vocabulary)  # so every scouting group mixes common and rare words
        label = "ABCDEFGH"[seed % 8] if tries > 1 else ""
        return Composer(jev, vocabulary, question, label).run(spinner)

    with Spinner("", "starting") as spinner:
        with ThreadPoolExecutor() as pool:  # the animation follows the first answer
            answers = list(pool.map(lambda seed: one(seed, spinner if seed == 0 else None), range(tries)))
        if tries == 1:
            return answers[0]
        spinner.status = f"grading {tries} answers"
        grades = {}
        for answer in dict.fromkeys(answers):
            grades[answer] = jev({"question": question, "answer": answer}, {"grade": grading_question()})["grade"].score
            log(f"\ngrade {grades[answer]:.2f}  {answer!r}")
        spinner.text = max(grades, key=grades.get)
    return spinner.text


def main() -> None:
    global VERBOSE
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog="Needs TYPESAFE_API_KEY, from the environment or a .env file."
    )
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every round's judgments and the token usage")
    parser.add_argument("--tries", type=int, default=TRIES, help="compose this many answers in parallel and keep the best")
    parser.add_argument("--vocab", type=Path, default=VOCABULARY, help="word list to compose from, one word per line")
    args = parser.parse_args()
    VERBOSE = args.verbose

    load_dotenv()
    try:
        jev = Jev()
        answer = compose(jev, args.vocab.read_text().lower().split(), args.question, args.tries)
    except KeyboardInterrupt:
        sys.exit("\n(stopped)")
    except (TypeSafeError, OSError) as error:
        sys.exit(f"error: {error}")
    print(("\r\x1b[2K" if animated() else "") + (answer or "(no answer)"))
    log(f"\n{jev.requests} requests, {jev.tokens:,} input tokens")


if __name__ == "__main__":
    main()
