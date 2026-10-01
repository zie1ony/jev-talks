"""Make a model that cannot write talk, by rewriting a noisy draft.

Jev, TypeSafe's System One model, does not generate text. It takes some state
plus typed questions and returns calibrated probabilities. talk.py composes an
answer left to right anyway; this script borrows the loop of a diffusion
language model instead. A whole draft exists from the first pass on, every
pass judges all of it at once and changes a few words, and the loop ends when
Jev calls the answer good:

  1. scout   once: show the vocabulary in groups of 250 and ask each group
             "which of these is likely to appear in the answer?"; the 100 most
             probable words, the 60 most frequent ones and the question's own
             words become the pool every position chooses from
  2. write   the first pass: ask every position of an empty row "which word of
             the pool belongs here?" and fill them all at once, no word twice;
             what comes out is noise, "the is what un of est in called paris"
  3. judge   every later pass, in one request: ask, for every word, "is it
             wrong here?" and, with the word hidden, "which word should stand
             here, or none?"; for every gap between words, "which word, if
             any, is missing here?"; and about the whole draft, "is this a
             complete, good answer?"
  4. change  up to 3 things: the most doubted words give way to the word
             their place prefers, or to nothing, and the most missed words
             are inserted
  5. stop    once the draft is at least 0.6 probably a good answer, when
             nothing is left to change, or after 12 passes; the best draft
             that was judged is the answer

A pass is one request, about 17,000 tokens and half a second. Three rows are
rewritten in parallel, each from a differently grouped vocabulary, and Jev
grades them on a rubric to keep the best one.

Usage:  uv run diffuse.py "Why is the sky blue?" [-v] [-n WORDS] [--tries N] [--vocab FILE]
Needs TYPESAFE_API_KEY, from the environment or a .env file.
"""

import argparse
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Question, Score, TypeSafeClient, TypeSafeError

# Settings --------------------------------------------------------------------

VOCABULARY = Path(__file__).with_name("top-english-3000.txt")
GROUP = 250  # words per scouting question; a Choice takes at most 255 options
BLANKS = 10  # blanks the first pass fills, all at once; this is the size of the response (also -n)
LARGEST = 24  # the largest size; a pass over a longer row would not fit into one request of 64,000 tokens
SCOUTED = 100  # scouted words that join the pool
COMMON = 60  # words from the top of the vocabulary file, its most frequent ones, that are always in the pool
CHANGES = 3  # words a pass may replace, remove or insert
DOUBT = 0.5  # a word is replaced or removed only if it is at least this doubted
INSERT = 0.3  # a gap gets a word only if that word beats "nothing is missing" by at least this much
GOOD = 0.6  # the loop stops once the draft is at least this probably a good answer
MAX_PASSES = 12  # hard stop
EXTRA = 2  # words the later passes may insert beyond the size of the response
TRIES = 3  # rows rewritten in parallel; the best by Jev's grade is kept

NOTHING, QUALITY = "<no word>", "<quality>"

WRITING = (
    "An answer to `question` is being written as one short, complete sentence. `draft` shows the whole sentence"
    " with ___ in place of every word that is still missing. Each ___ takes exactly one word."
    " The sentence ends at the period; words are lowercase and there is no other punctuation."
)
REWRITING = (
    "An answer to `question` is being written as one short, complete sentence. `draft` is the current attempt;"
    " it may still contain wrong, misplaced, repeated or missing words. It is improved in passes: every word is"
    " judged in its place, and a few of them are replaced in each pass."
    " The sentence ends at the period; words are lowercase and there is no other punctuation."
)

VERBOSE = False
STOP = threading.Event()  # set on Ctrl-C or an error, so that every row stops after its current request


# The row ---------------------------------------------------------------------


@dataclass
class Word:
    """One word of the row, with what the next pass and the animation need to know about it."""

    text: str
    tried: set[str] = field(default_factory=set)  # the words this place already held, itself included; none returns
    doubt: float = 0.0  # how wrong the latest pass found it
    fresh: bool = True  # whether the latest pass put it there


class Diffusion:
    def __init__(self, jev: "Jev", vocabulary: list[str], common: list[str], question: str, size: int, label: str = ""):
        self.jev, self.vocabulary, self.common, self.question, self.label = jev, vocabulary, common, question, label
        self.size = size  # how many words the first pass writes
        self.row: list[Word] = []
        self.seen: set[str] = set()  # drafts that were already judged; never rebuilt

    def run(self, spinner: "Spinner | None" = None) -> str:
        """Write a noisy draft, then rewrite a few words per pass until Jev calls it good. Returns the best draft."""
        self.tell(spinner, f"scouting {len(self.vocabulary):,} words")
        pool = self.scout()
        self.tell(spinner, "pass 1")
        self.write(pool)
        best, quality = self.text, -1.0
        for n in range(2, MAX_PASSES + 1):
            if STOP.is_set():
                break
            self.tell(spinner, f"pass {n}")
            answers = self.judge(pool)
            good = answers[QUALITY].noul if QUALITY in answers else 0.0
            if good > quality:
                best, quality = self.text, good
            doubts = sorted(self.row, key=lambda word: word.doubt, reverse=True)[:5]
            self.log(("" if self.label else "\n") + f"pass {n}, good {good:.2f}, draft: {draft(self.words)!r}")
            self.log("  doubts: ", ", ".join(f"{word.text} {word.doubt:.2f}" for word in doubts))
            self.tell(spinner, f"pass {n}, good {good:.2f}")
            changes = [] if good >= GOOD or n == MAX_PASSES else self.choose(answers)
            if not changes:
                break
            self.change(changes)
        if spinner:
            spinner.text, spinner.status = best, "finished"
        return best

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

    def write(self, pool: list[str]) -> None:
        """The first pass, one request: every blank gets a word at once, and no word is used twice."""
        blanks = ["___"] * self.size
        answers = self.jev(self.state(WRITING, blanks), writing_questions(blanks, pool))
        offers = [(p, i, word) for i in range(self.size) for word, p in answers[f"blank{i}"].probabilities.items() if p]
        filled: dict[int, str] = {}
        for _, i, word in sorted(offers, reverse=True):  # the surest pairs of blank and word first
            if i not in filled and (word == NOTHING or word not in filled.values()):
                filled[i] = word
        self.row = [Word(filled[i], {filled[i]}) for i in sorted(filled) if filled[i] != NOTHING]
        self.log(("" if self.label else "\n") + f"pass 1, draft: {draft(self.words)!r}")

    def judge(self, pool: list[str]) -> dict:
        """One request about the whole draft: every word, every gap between words, and how good it all is."""
        answers = self.jev(self.state(REWRITING, self.words), judging_questions(self.words, pool, self.size + EXTRA))
        for i, word in enumerate(self.row):
            word.doubt = answers[f"doubt{i}"].noul
        return answers

    def choose(self, answers: dict) -> list[tuple[float, str]]:
        """Pick this pass's few changes, as (where, new word): a word's index, or k - 0.5 for the gap before word k."""
        agreed, doubted = [], []  # (urgency, where, new word); doubted ones lack the backing of their place's preference
        for i, word in enumerate(self.row):
            probabilities = answers[f"word{i}"].probabilities
            untried = (w for w in top(probabilities, len(probabilities)) if w not in word.tried and probabilities[w])
            new = next(untried, None)
            if new is not None and word.doubt >= DOUBT:
                backed = probabilities[new] > probabilities.get(word.text, 0.0)
                (agreed if backed else doubted).append((word.doubt, i, new))
        for k in range(len(self.row) + 1):
            if f"gap{k}" in answers:
                probabilities = answers[f"gap{k}"].probabilities
                new = next(w for w in top(probabilities, 2) if w != NOTHING)
                if probabilities[new] - probabilities.get(NOTHING, 0.0) >= INSERT:
                    agreed.append((probabilities[new] - probabilities.get(NOTHING, 0.0), k - 0.5, new))
        changes: list[tuple[float, str]] = []
        for _, where, new in sorted(agreed or doubted, reverse=True):  # never two neighbours, never one word twice
            apart = all(abs(where - other) > 1 and (new != word or new == NOTHING) for other, word in changes)
            if len(changes) < CHANGES and apart:
                changes.append((where, new))
        while changes and text(self.after(changes)) in self.seen:
            changes.pop()
        return changes

    def change(self, changes: list[tuple[float, str]]) -> None:
        self.seen.add(self.text)
        for word in self.row:
            word.fresh = False
        made = []
        for where, new in sorted(changes, reverse=True):  # from the right, so that the indices stay valid
            i = int(where + 0.5)
            if i != where:
                self.row.insert(i, Word(new, {new}))
                made.append(f"added {new!r}")
            elif new == NOTHING:
                made.append(f"removed {self.row.pop(i).text!r}")
            else:
                made.append(f"{self.row[i].text!r} became {new!r}")
                self.row[i].tried.add(new)
                self.row[i].text, self.row[i].doubt, self.row[i].fresh = new, 0.0, True
        self.log("  ->", ", ".join(reversed(made)))

    def after(self, changes: list[tuple[float, str]]) -> list[str]:
        """The words of the row once the changes are made."""
        words = self.words
        for where, new in sorted(changes, reverse=True):
            i = int(where + 0.5)
            if i != where:
                words.insert(i, new)
            elif new == NOTHING:
                del words[i]
            else:
                words[i] = new
        return words

    def tell(self, spinner: "Spinner | None", status: str) -> None:
        """Update the terminal animation, when there is one: just-changed words bold, doubted words dim."""
        if spinner:
            spinner.text = " ".join(map(styled, self.row)) or "\x1b[2m" + " ".join(["___"] * self.size) + "\x1b[0m"
            spinner.status = status

    def log(self, *parts: object) -> None:
        log(*((f"[{self.label}]",) if self.label else ()), *parts)

    def state(self, rules: str, words: list[str]) -> dict:
        return {"rules": rules, "question": self.question, "draft": draft(words)}

    @property
    def words(self) -> list[str]:
        return [word.text for word in self.row]

    @property
    def text(self) -> str:
        return text(self.words)


def styled(word: Word) -> str:
    """A word for the terminal: dim while it is doubted, bold right after it was put there."""
    return ("\x1b[2m" if word.doubt >= DOUBT else "\x1b[1m" if word.fresh else "") + word.text + "\x1b[0m"


def text(words: list[str]) -> str:
    return " ".join(words)


def draft(words: list[str], marked: float | None = None, reveal: bool = False) -> str:
    """The row as Jev reads it: its words in order and a period at the end.

    One place can be marked: a word's index hides the word as [?], or shows it as [word] with `reveal`;
    k - 0.5 puts [?] into the gap before word k.
    """
    shown = [(f"[{word}]" if reveal else "[?]") if i == marked else word for i, word in enumerate(words)]
    if marked is not None and marked != int(marked):
        shown.insert(int(marked + 0.5), "[?]")
    return " ".join(shown) + " ."


# The questions ---------------------------------------------------------------


def scouting_question(words: list[str]) -> Choice:
    return Choice(
        instructions="Which of these words is the most likely to appear in a good one-sentence answer to `question`?"
        " Every option is a vocabulary word to be used literally, not a meta answer.",
        criteria={word: None for word in words},
    )


def writing_questions(blanks: list[str], pool: list[str]) -> dict[str, Question]:
    options: dict[str, str | None] = {word: None for word in pool}
    options[NOTHING] = "No word is needed here: the sentence is better with this blank simply removed."
    return {
        f"blank{i}": Choice(
            instructions=f"Here is the draft again with one blank marked: '{draft(blanks, i)}'. Which word should"
            " replace [?] so that the finished sentence reads as natural, correct English and answers `question`"
            " well? Every option is a word to be placed literally.",
            criteria=options,
        )
        for i in range(len(blanks))
    }


def judging_questions(words: list[str], pool: list[str], limit: int) -> dict[str, Question]:
    """What one pass asks: per word a doubt and its place's preference, per gap what is missing, and the quality.

    A row of `limit` words or more gets no gap questions, so that nothing more is inserted.
    """
    kept: dict[str, str | None] = {word: None for word in pool}
    kept[NOTHING] = "No word is needed at [?]: the sentence is better without one there."
    missing: dict[str, str | None] = {word: None for word in pool}
    missing[NOTHING] = (
        "Nothing is missing at [?]: the words around it already connect, or the sentence starts or ends there."
    )
    questions: dict[str, Question] = {}
    for i in range(len(words)):
        questions[f"word{i}"] = Choice(
            instructions=f"Here is the draft again with one place marked: '{draft(words, i)}'. Which word should"
            " stand at [?] so that the sentence reads as natural, correct English and answers `question` well?"
            " Every option is a word to be placed literally.",
            criteria=kept,
        )
        questions[f"doubt{i}"] = Noul(
            instructions=f"Here is the draft again with one word marked in brackets: '{draft(words, i, reveal=True)}'."
            " Is the marked word wrong here: out of place, ungrammatical, repeated, or making the answer incorrect?"
        )
    for k in range(len(words) + 1 if len(words) < limit else 0):
        questions[f"gap{k}"] = Choice(
            instructions="Here is the draft again with one place between its words marked:"
            f" '{draft(words, k - 0.5)}'. Which word, if any, should be inserted at [?] so that the sentence reads as"
            " natural, correct English and answers `question` well? Every option is a word to be placed literally.",
            criteria=missing,
        )
    if words:
        questions[QUALITY] = Noul(
            instructions="Is `draft` a complete, good answer to `question`, so that rewriting should stop?",
            criteria={
                "true": "The words form a complete, natural sentence of several words that fully answers the question.",
                "false": "It is a fragment, unfinished, ungrammatical, repetitive, or still contains a wrong word.",
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
    """Redraw the row in place only on a real terminal that is not showing the verbose trace."""
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
    """Keep the row on the screen, with a spinning glyph and a status, while Jev thinks."""

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


def compose(jev: Jev, words: list[str], question: str, tries: int, size: int) -> str:
    """Rewrite `tries` rows in parallel, each from a differently grouped vocabulary, and keep the best."""
    def one(seed: int, spinner: Spinner | None) -> str:
        vocabulary = list(words)
        random.Random(seed).shuffle(vocabulary)  # so every scouting group mixes common and rare words
        label = "ABCDEFGH"[seed % 8] if tries > 1 else ""
        return Diffusion(jev, vocabulary, words[:COMMON], question, size, label).run(spinner)

    with Spinner("", "starting") as spinner:
        with ThreadPoolExecutor() as pool:  # the animation follows the first row
            try:
                answers = list(pool.map(lambda seed: one(seed, spinner if seed == 0 else None), range(tries)))
            except BaseException:  # Ctrl-C, or an error in one row: the others should not run on
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
    parser.add_argument("-v", "--verbose", action="store_true", help="print every pass's judgments and the token usage")
    parser.add_argument(
        "-n", dest="size", type=int, default=BLANKS, metavar="WORDS", help="how large the response is, in words"
    )
    parser.add_argument("--tries", type=int, default=TRIES, help="rows to rewrite in parallel; the best one is kept")
    parser.add_argument("--vocab", type=Path, default=VOCABULARY, help="word list to compose from, one word per line")
    args = parser.parse_args()
    if args.tries < 1:
        parser.error("--tries must be at least 1")
    if not 1 <= args.size <= LARGEST:
        parser.error(f"-n must be between 1 and {LARGEST}")
    VERBOSE = args.verbose

    load_dotenv()
    try:
        words = list(dict.fromkeys(args.vocab.read_text().lower().split()))
        if not words:
            sys.exit(f"error: {args.vocab} has no words")
        jev = Jev()
        answer = compose(jev, words, args.question, args.tries, args.size)
    except KeyboardInterrupt:
        sys.exit("\n(stopped)")
    except (TypeSafeError, OSError) as error:
        sys.exit(f"error: {error}")
    print(("\r\x1b[2K" if animated() else "") + (answer or "(no answer)"))
    log(f"\n{jev.requests} requests, {jev.tokens:,} input tokens")


if __name__ == "__main__":
    main()
