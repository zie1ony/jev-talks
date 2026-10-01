from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Question, Score, TypeSafeClient, TypeSafeError
from pathlib import Path
from dataclasses import dataclass, replace
from math import sqrt
import random


words = Path(__file__).with_name("basic-english-850.txt").read_text().lower().split()

SCOUT_SIZE = 50 # Number of words to scout from the list
GROUP = 50 # Words per scouting question; a Choice gives a real probability to a few options only, so the groups are small
POSITION_SIZE = 3 # Best words of every position that get reevaluated
SURE = 0.7 # Besides the best match, a word reevaluated at least this probable is fixed in the same round
EMPTY = "<empty>" # The word that removes its position from the sentence
OPEN = "___" # How Jev sees a position that is not fixed yet

RULES = (
    "A one-sentence response to `input` is being written into a row of positions, one word per position."
    f" `draft` is the row so far. {OPEN} is a position that is still open: it will get one word or be removed."
    " The finished response must read as natural, correct English, and it must answer the input, not repeat it."
)

@dataclass
class Option:
    """A word that could be fixed at a position."""
    position: int
    word: str
    share: float # Probability of the word among all scouted words, asked for this position
    sure: float = 0.0 # Probability that the draft is still a good sentence with the word in place

    @property
    def match(self) -> float:
        """How well the word matches the position. The share lets a word the position clearly prefers beat one that fits anywhere."""
        return self.sure * sqrt(self.share)

class Diffusion:
    def __init__(self, input: str, output_size: int):
        self.client = TypeSafeClient()
        self.output = [""] * output_size # The word drawn for every position; it can change until the position is fixed
        self.fixed_positions = [False] * output_size
        self.input = input

    def run(self) -> str:
        while not all(self.fixed_positions):
            words = self.scout()
            positional_options = self.diffusion_step(words)
            positional_options, complete = self.reevaluate_positional_options(positional_options)
            for option in sorted(positional_options, key=lambda option: option.match):
                self.output[option.position] = option.word # Every open position draws its best match again
            for option in self.decide_which_words_to_fix(positional_options, complete):
                self.output[option.position] = option.word
                self.fixed_positions[option.position] = True
            print(self.progress())
        return " ".join(word for word in self.output if word != EMPTY)

    def scout(self) -> list[str]:
        """Draw the words that could fill the open positions of the draft."""
        vocabulary = random.sample(words, len(words)) # Shuffled, so that every group mixes common and rare words
        groups = [vocabulary[i : i + GROUP] for i in range(0, len(vocabulary), GROUP)]
        questions = {
            f"group{i}": Choice(
                instructions="Which of these words is the most likely to be one of the words that are still missing in `draft`?"
                " Every option is a vocabulary word to be placed literally, not a meta answer.",
                criteria={word: None for word in group},
            )
            for i, group in enumerate(groups)
        }
        response = self.client.system_one(state=self.state(), questions=questions)
        probabilities = {word: p for answer in response.answers.values() for word, p in answer.probabilities.items()}
        return sorted(probabilities, key=probabilities.get, reverse=True)[:SCOUT_SIZE]

    def diffusion_step(self, posible_words: list[str]) -> list[Option]:
        """Ask every open position which word belongs there. Return its best words, and <empty>, with their probabilities."""
        criteria = {word: None for word in posible_words}
        criteria[EMPTY] = "No word belongs here: the draft reads better with this position removed."
        open_positions = [i for i, fixed in enumerate(self.fixed_positions) if not fixed]
        questions = {
            f"position{i}": Choice(instructions=f"Which word belongs at [?] in {self.draft(i)!r}?", criteria=criteria)
            for i in open_positions
        }
        response = self.client.system_one(state=self.state(), questions=questions)
        positional_options = []
        for i in open_positions:
            probabilities = response.answers[f"position{i}"].probabilities
            best = sorted(probabilities, key=probabilities.get, reverse=True)[:POSITION_SIZE]
            positional_options += [Option(i, word, probabilities[word]) for word in dict.fromkeys(best + [EMPTY])]
        return positional_options

    def reevaluate_positional_options(self, positional_options: list[Option]) -> tuple[list[Option], float]:
        """Ask a yes/no question about every option, spelling out the draft it leads to, and one more: is the draft complete?"""
        questions = {}
        for i, option in enumerate(positional_options):
            change = "Removing [?] from" if option.word == EMPTY else f"Writing {option.word!r} at [?] in"
            questions[f"option{i}"] = Noul(
                instructions=f"{change} {self.draft(option.position)!r} gives {self.draft(option.position, option.word)!r}."
                " Is that a good step toward a natural, correct English sentence that answers `input`?"
            )
        fixed_words = " ".join(word for word, fixed in zip(self.output, self.fixed_positions) if fixed and word != EMPTY)
        questions["complete"] = Noul(
            instructions=f"Without its open positions the draft reads {fixed_words!r}. Is that already a complete, good"
            " response to `input`, so that the open positions should be removed?",
            criteria={
                "true": "It is a complete sentence of several words that answers the input; more words would make it worse.",
                "false": "It is empty, a single word, a fragment, unfinished, or still misses a word.",
            },
        )
        answers = self.client.system_one(state=self.state(), questions=questions).answers
        reevaluated = [replace(option, sure=answers[f"option{i}"].noul) for i, option in enumerate(positional_options)]
        return reevaluated, answers["complete"].noul

    def decide_which_words_to_fix(self, positional_options: list[Option], complete: float) -> list[Option]:
        """Fix the best match, and every other word that is SURE. A complete sentence has its open positions removed instead."""
        ranked = sorted(positional_options, key=lambda option: option.match, reverse=True)
        if complete > max(0.5, ranked[0].sure):
            return [option for option in ranked if option.word == EMPTY]
        to_fix = ranked[:1]
        for option in ranked[1:]:
            # Neighbours were judged without each other, and so was a word that is written twice
            clashes = any(option.word == other.word or self.neighbours(option.position, other.position) for other in to_fix)
            if option.sure >= SURE and not clashes:
                to_fix.append(option)
        return to_fix

    def neighbours(self, a: int, b: int) -> bool:
        """Two positions are neighbours when every position between them has been removed."""
        return all(self.fixed_positions[i] and self.output[i] == EMPTY for i in range(min(a, b) + 1, max(a, b)))

    def state(self) -> dict:
        return {"input": self.input, "rules": RULES, "draft": self.draft()}

    def draft(self, position: int | None = None, word: str = "[?]") -> str:
        """The row as Jev sees it: only the fixed words, optionally with `word` written at `position`."""
        row = [fixed_word if fixed else OPEN for fixed_word, fixed in zip(self.output, self.fixed_positions)]
        if position is not None:
            row[position] = word
        return " ".join(shown for shown in row if shown != EMPTY) + " ." # The period shows Jev where the sentence has to end

    def progress(self) -> str:
        """The row for the terminal: fixed words as they are, words that can still be drawn again in parentheses."""
        row = [word if fixed else f"({word})" for word, fixed in zip(self.output, self.fixed_positions)]
        return " ".join(shown for shown in row if shown != EMPTY)


def main() -> None:
    load_dotenv()

    diffusion = Diffusion(input="Tell me a joke", output_size=10)
    diffusion.run()

if __name__ == "__main__":
    main()
