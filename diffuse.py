"""Make a model that cannot write talk, by filling in the blanks.

Jev, TypeSafe's System One model, does not generate text. It takes some state
plus typed questions and returns calibrated probabilities. talk.py composes an
answer left to right anyway; this script borrows the loop of a masked diffusion
language model instead. The answer starts as a row of blanks, every blank is
judged in every step, and words land wherever the model is surest:

  1. scout    once: show the vocabulary in groups of 250 and ask each group
              "which of these is likely to appear in the answer?"; the 100 most
              probable words, the 60 most frequent ones and the question's own
              words become the pool every blank chooses from
  2. propose  ask, for every blank, "which word of the pool belongs here, or
              none at all?"; for every placed word, "is this word wrong here?";
              and once, "is the answer already complete?"
  3. verify   ask, for the 3 best words of every blank, "with this word in
              place, is the draft still on its way to a good sentence?"
  4. apply    the most probable move: place a word, or several when the model
              is sure; drop a blank; blank a doubted word again; or finish

A step is two requests, about 10,000 tokens and half a second together. While
its placements are verified, the proposals for the canvas that the favourite
one leads to are requested ahead, which saves a request in four steps out of
ten. Three canvases are denoised in parallel, each from a differently grouped
vocabulary, and Jev grades them on a rubric to keep the best one.

Usage:  uv run diffuse.py "Why is the sky blue?" [-v] [--tries N] [--vocab FILE]
Needs TYPESAFE_API_KEY, from the environment or a .env file.
"""

import argparse
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Question, Score, TypeSafeClient, TypeSafeError

# Settings --------------------------------------------------------------------

VOCABULARY = Path(__file__).with_name("top-english-3000.txt")
GROUP = 250  # words per scouting question; a Choice takes at most 255 options
BLANKS = 10  # blanks the canvas starts with; surplus blanks are dropped, a missing one is added at the end
SCOUTED = 100  # scouted words that join the pool
COMMON = 60  # words from the top of the vocabulary file, its most frequent ones, that are always in the pool
CANDIDATES = 3  # words per blank that advance from the multiple-choice question to the yes/no check
PRIOR = 0.5  # a placement ranks by its yes/no probability times its multiple-choice probability to this power
SURE = 0.75  # besides the best placement, every other one at least this probable lands in the same step
DOUBT = 0.5  # a placed word at least this doubted is blanked again
FLOOR = 0.3  # a placement less probable than this is never made
MAX_WORDS = 12  # a full canvas that is not a finished sentence grows by one blank, up to this many words
MAX_REBLANKS = 6  # words blanked again per canvas; past that the canvas keeps what it has
MAX_STEPS = 16  # hard stop; a canvas that needs more is going in circles
TRIES = 3  # canvases denoised in parallel; the best by Jev's grade is kept

NOTHING, FINISH = "<no word>", "<finish>"

RULES = (
    "An answer to `question` is being written as one short, complete sentence. It is written like a crossword,"
    " not left to right: `draft` shows the whole sentence with ___ in place of every word that is still missing,"
    " and the blanks are filled in any order. Each ___ takes exactly one word."
    " The sentence ends at the period; words are lowercase and there is no other punctuation."
)

VERBOSE = False
STOP = threading.Event()  # set on Ctrl-C or an error, so that every canvas stops after its current request


# The canvas ------------------------------------------------------------------


@dataclass
class Cell:
    """One position of the canvas: a placed word, or a blank and the model's current guess for it."""

    word: str | None = None
    guess: str = "___"  # only shown in the animation


class Diffusion:
    def __init__(self, jev: "Jev", vocabulary: list[str], common: list[str], question: str, label: str = ""):
        self.jev, self.vocabulary, self.common, self.question, self.label = jev, vocabulary, common, question, label
        self.cells = [Cell() for _ in range(BLANKS)]
        self.rejected: set[str] = set()  # drafts a word was blanked out of; never rebuilt
        self.extended: set[str] = set()  # full drafts that got one more blank; never extended twice
        self.reblanks = 0

    def run(self, spinner: "Spinner | None" = None) -> str:
        """Denoise the canvas step by step until the model finishes. A step is two requests, propose and verify."""
        self.tell(spinner, f"scouting {len(self.vocabulary):,} words")
        pool = self.scout()
        with ThreadPoolExecutor(max_workers=1) as ahead:
            expected, early = None, None  # the canvas the last step expected to lead to, and its proposals
            for n in range(1, MAX_STEPS + 1):
                if STOP.is_set():
                    break
                words = self.words
                self.log(("" if self.label else "\n") + f"step {n}, draft: {self.draft!r}")
                self.tell(spinner, f"filling {count(words.count(None), 'blank')}" if None in words else "checking")
                answers = early.result() if words == expected else self.propose(words, pool)
                guesses, doubts, finished = self.read(answers)
                candidates = [(i, word) for i, word in guesses if render(after(words, i, word)) not in self.rejected]
                expected = None
                if candidates and finished < 0.5:  # while these are verified, propose for where the favourite leads
                    expected = after(words, *max(candidates, key=guesses.get))
                    early = ahead.submit(self.propose, expected, pool)
                self.tell(spinner, f"judging {count(len(candidates), 'placement')}")
                verdicts = self.verify(words, candidates)
                if not self.apply(guesses, verdicts, doubts, finished):
                    break
        self.cells = [cell for cell in self.cells if cell.word]  # the blanks that were never filled
        self.tell(spinner, "finished")
        return self.text

    def scout(self) -> list[str]:
        """One request: a multiple-choice question per group of words. Returns the pool for the whole answer."""
        n = -(-len(self.vocabulary) // GROUP)  # as few groups as will fit, all of the same size
        groups = [self.vocabulary[i::n] for i in range(n)]
        questions = {f"group{i}": scouting_question(group) for i, group in enumerate(groups)}
        answers = self.jev({"question": self.question}, questions)
        probabilities = {word: p for answer in answers.values() for word, p in answer.probabilities.items()}
        scouted = top(probabilities, SCOUTED)
        asked = [word for word in re.findall(r"[a-z']+", self.question.lower()) if word in probabilities]
        self.log("  scouted:", show(probabilities, scouted[:10]), "...")
        return list(dict.fromkeys(asked + scouted + self.common))[:254]  # plus NOTHING: 255 options at most

    def propose(self, words: list[str | None], pool: list[str]) -> dict:
        """One request: which word belongs in each blank, is any placed word wrong, is the answer complete?"""
        questions = proposing_questions(words, pool)
        return self.jev(self.state(words), questions) if questions else {}

    def read(self, answers: dict) -> tuple[dict, dict, float]:
        """Sort the proposals into guesses for the blanks, doubts about the placed words, and finishing."""
        guesses = {}  # (position, word) -> the word's share of its blank's probability
        for i, cell in enumerate(self.cells):
            if cell.word is None:
                probabilities = answers[f"blank{i}"].probabilities
                best = [word for word in top(probabilities, CANDIDATES) if probabilities[word]]
                guesses |= {(i, word): probabilities[word] for word in best}
                cell.guess = best[0] if best and best[0] != NOTHING else "___"
        doubts = {i: answers[f"word{i}"].noul for i, cell in enumerate(self.cells) if cell.word}
        return guesses, doubts, answers[FINISH].noul if FINISH in answers else 0.0

    def verify(self, words: list[str | None], candidates: list[tuple[int, str]]) -> dict:
        """One request: a yes/no question per proposed placement."""
        questions = {f"{i} {word}": verifying_question(words, i, word) for i, word in candidates}
        answers = self.jev(self.state(words), questions) if questions else {}
        return {(i, word): answers[f"{i} {word}"].noul for i, word in candidates}

    def apply(self, guesses: dict, verdicts: dict, doubts: dict, finished: float) -> bool:
        """Make the most probable move. Returns False once the answer is finished."""
        sound = {candidate: p for candidate, p in verdicts.items() if p >= FLOOR}
        ranked = sorted(sound, key=lambda candidate: sound[candidate] * guesses[candidate] ** PRIOR, reverse=True)
        best = max(sound.values(), default=0.0)
        doubted = max(doubts, key=doubts.get, default=None)
        # logged as word@position, then its yes/no probability and its share of the blank's multiple choice
        moves = ", ".join(f"{word}@{i + 1} {sound[i, word]:.2f}/{guesses[i, word]:.2f}" for i, word in ranked[:5])
        doubt = f"| doubt {self.cells[doubted].word!r} {doubts[doubted]:.2f}" if doubts else ""
        self.log("  moves:  ", moves or "none", f"| finish {finished:.2f}", doubt)
        if doubted is not None and (doubts[doubted] < DOUBT or self.reblanks >= MAX_REBLANKS):
            doubted = None

        if doubted is not None and doubts[doubted] >= max(best, finished):
            return self.blank(doubted)
        if None not in self.words:  # a full canvas: stop, or add a blank if the sentence is not finished
            if finished >= 0.5 or len(self.cells) >= MAX_WORDS or self.draft in self.extended:
                return False
            self.extended.add(self.draft)
            self.cells.append(Cell())
            self.note("added a blank at the end")
            return True
        if finished >= best:  # also when no placement is sound
            self.log("  -> finished; the blanks that are left are dropped")
            return False
        chosen = ranked[:1]
        for candidate in ranked[1:]:
            if sound[candidate] >= SURE and all(together(candidate, other) for other in chosen):
                chosen.append(candidate)
        for i, word in sorted(chosen, reverse=True):  # from the right, so that dropping a blank moves nothing
            if word == NOTHING:
                del self.cells[i]
            else:
                self.cells[i].word = word
        made = ("dropped a blank" if word == NOTHING else f"placed {word!r}" for _, word in sorted(chosen))
        self.note(", ".join(made))
        return True

    def blank(self, i: int) -> bool:
        self.rejected.add(self.draft)
        self.reblanks += 1
        taken, self.cells[i] = self.cells[i].word, Cell()
        self.note(f"blanked {taken!r} again")
        return True

    def tell(self, spinner: "Spinner | None", status: str | None) -> None:
        """Update the terminal animation, when there is one: placed words as they are, blanks as dim guesses."""
        if spinner:
            spinner.text = " ".join(cell.word or f"\x1b[2m{cell.guess}\x1b[0m" for cell in self.cells)
            if status:
                spinner.status = status

    def log(self, *parts: object) -> None:
        log(*((f"[{self.label}]",) if self.label else ()), *parts)

    def note(self, what: str) -> None:
        self.log(f"  -> {what}; the draft now reads {self.draft!r}")

    def state(self, words: list[str | None]) -> dict:
        return {"rules": RULES, "question": self.question, "draft": render(words)}

    @property
    def words(self) -> list[str | None]:
        return [cell.word for cell in self.cells]

    @property
    def draft(self) -> str:
        return render(self.words)

    @property
    def text(self) -> str:
        return " ".join(cell.word for cell in self.cells if cell.word)


def render(words: list[str | None], marked: int | None = None) -> str:
    """The canvas as Jev reads it: ___ for a blank, a period at the end, and one cell in brackets if marked."""
    shown = [word or "___" for word in words]
    if marked is not None:
        shown[marked] = f"[{words[marked] or '?'}]"
    return " ".join(shown) + " ."


def after(words: list[str | None], i: int, word: str) -> list[str | None]:
    """The canvas once blank i is filled with a word, or dropped."""
    return words[:i] + ([] if word == NOTHING else [word]) + words[i + 1 :]


def together(one: tuple[int, str], other: tuple[int, str]) -> bool:
    """Whether two placements may land in the same step: never the same word, never two neighbouring blanks."""
    (i, word), (j, second) = one, other
    if NOTHING in (word, second):
        return i != j
    return word != second and abs(i - j) > 1


# The questions ---------------------------------------------------------------


def scouting_question(words: list[str]) -> Choice:
    return Choice(
        instructions="Which of these words is the most likely to appear in a good one-sentence answer to `question`?"
        " Every option is a vocabulary word to be used literally, not a meta answer.",
        criteria={word: None for word in words},
    )


def proposing_questions(words: list[str | None], pool: list[str]) -> dict[str, Question]:
    options: dict[str, str | None] = {word: None for word in pool}
    options[NOTHING] = "No word is needed here: the sentence is better with this blank simply removed."
    questions: dict[str, Question] = {}
    for i, word in enumerate(words):
        if word is None:
            questions[f"blank{i}"] = Choice(
                instructions=f"Here is the draft again with one blank marked: '{render(words, i)}'. Which word"
                " should replace [?] so that the finished sentence reads as natural, correct English and answers"
                " `question` well? Every option is a word to be placed literally.",
                criteria=options,
            )
        else:
            questions[f"word{i}"] = Noul(
                instructions=f"Here is the draft again with one word marked in brackets: '{render(words, i)}'."
                " Every ___ is a blank that will be filled with one word later. Is the marked word wrong here: out of"
                " place, ungrammatical, repeated, or making the answer incorrect?"
            )
    placed = " ".join(word for word in words if word)
    if placed:
        questions[FINISH] = Noul(
            instructions=f"Ignoring the blanks, the draft reads {placed!r}. Is that already a complete, good answer"
            " to `question`, so writing should stop?",
            criteria={
                "true": "The words form a complete sentence of several words that fully answers the question;"
                " more words would make it worse.",
                "false": "It is a single word, a fragment, unfinished, or still contains a wrong word.",
            },
        )
    return questions


def verifying_question(words: list[str | None], i: int, word: str) -> Noul:
    move = "Removing a blank" if word == NOTHING else f"Filling a blank with {word!r}"
    return Noul(
        instructions=f"{move} would change the draft to {render(after(words, i, word))!r}. Every ___ is a blank that"
        " will be filled with one word later. Would the draft then be a good step toward a natural, correct English"
        " sentence that gives a complete, informative answer to `question`?",
        criteria={
            "true": "Every word fits its neighbours and the answer; a good sentence can be completed from this draft.",
            "false": "A word is out of place, ungrammatical next to its neighbours, repeated, or leads the answer"
            " astray.",
        },
    )


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


def count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def log(*parts: object) -> None:
    if VERBOSE:
        print(*parts, file=sys.stderr, flush=True)


def animated() -> bool:
    """Redraw the canvas in place only on a real terminal that is not showing the verbose trace."""
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
    """Keep the canvas on the screen, with a spinning glyph and a status, while Jev thinks."""

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
    """Denoise `tries` canvases in parallel, each from a differently grouped vocabulary, and keep the best."""
    def one(seed: int, spinner: Spinner | None) -> str:
        vocabulary = list(words)
        random.Random(seed).shuffle(vocabulary)  # so every scouting group mixes common and rare words
        label = "ABCDEFGH"[seed % 8] if tries > 1 else ""
        return Diffusion(jev, vocabulary, words[:COMMON], question, label).run(spinner)

    with Spinner("", "starting") as spinner:
        with ThreadPoolExecutor() as pool:  # the animation follows the first canvas
            try:
                answers = list(pool.map(lambda seed: one(seed, spinner if seed == 0 else None), range(tries)))
            except BaseException:  # Ctrl-C, or an error in one canvas: the others should not run on
                STOP.set()
                raise
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
    parser.add_argument("-v", "--verbose", action="store_true", help="print every step's judgments and the token usage")
    parser.add_argument("--tries", type=int, default=TRIES, help="canvases to denoise in parallel; the best one is kept")
    parser.add_argument("--vocab", type=Path, default=VOCABULARY, help="word list to compose from, one word per line")
    args = parser.parse_args()
    if args.tries < 1:
        parser.error("--tries must be at least 1")
    VERBOSE = args.verbose

    load_dotenv()
    try:
        words = list(dict.fromkeys(args.vocab.read_text().lower().split()))
        if not words:
            sys.exit(f"error: {args.vocab} has no words")
        jev = Jev()
        answer = compose(jev, words, args.question, args.tries)
    except KeyboardInterrupt:
        sys.exit("\n(stopped)")
    except (TypeSafeError, OSError) as error:
        sys.exit(f"error: {error}")
    print(("\r\x1b[2K" if animated() else "") + (answer or "(no answer)"))
    log(f"\n{jev.requests} requests, {jev.tokens:,} input tokens")


if __name__ == "__main__":
    main()
