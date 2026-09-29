"""Two-phase composition with Jev: scout, then arrange.

Phase 1 (scouting) asks, for every vocabulary word, whether it could be one of
the next LOOKAHEAD words of the answer. Phase 2 (arranging) takes the TOP_WORDS
best scouts, builds every arrangement of 1..LENGTH of them, and judges each as
the possible next move — or removes the last word, skips the round, or declares
the answer complete.

Reads TYPESAFE_API_KEY from .env. Usage: uv run main.py
"""

import _thread
import sys
import threading
from itertools import cycle, permutations
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient, NoulAnswer

load_dotenv()

GATEWAY_URL = "https://api.typesafe.ai"

N_LAST_ACTIONS = 10

# Verbose logging; enabled by the --debug flag. When off, only the answer is printed.
DEBUG = False


def debug(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)


# User-defined game parameters.
LOOKAHEAD = 3     # scouting horizon: could the word be one of the next LOOKAHEAD words
TOP_WORDS = 30    # how many scouted words advance to the arranging phase
LENGTH = 2        # max words per placed arrangement (raising to 3 adds P(20,3)=6840 candidates)
SCOUT_BATCH = 1000  # words scouted per request; large vocabularies are split to stay under the token limit


class Token:
    FINISHED = "end_of_text"
    REMOVE_WORD = "remove_word"
    NONE_FITS = "none_of_them_fits"


# The vocabulary being scouted.
all_words = Path(__file__).with_name("top-english-3000.txt").read_text().lower().split()


def chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def append_words(text: str, phrase: str) -> str:
    return f"{text} {phrase}" if text else phrase


def remove_last_words(text: str, n: int) -> str:
    return " ".join(text.split()[:-n])


def scout_questions(word_batch: list[str]) -> dict[str, Noul]:
    return {
        word: Noul(instructions=f"Could {word!r} be one of the next {LOOKAHEAD} words of the answer?")
        for word in word_batch
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
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def system_one(self, **kwargs):
        """Wrap the SDK call so every request's `.usage` is accumulated."""
        response = self.client.system_one(**kwargs)
        self.total_input_tokens += response.usage.input_tokens or 0
        self.total_output_tokens += response.usage.output_tokens or 0
        self.total_calls += 1
        return response

    def print_usage(self) -> None:
        total = self.total_input_tokens + self.total_output_tokens
        debug(
            f"\nToken usage over {self.total_calls} calls: "
            f"{self.total_input_tokens} input + {self.total_output_tokens} output = {total} total"
        )

    def complete(self, system_prompt: str, input_text: str, steps_count: int, correction_loops: int) -> str:
        for _ in range(steps_count):
            if DEBUG:
                placed = self.next_move(system_prompt, input_text, self.response, correction_loops)
            else:
                # Spin a loader after the answer-so-far while the (blocking) move is computed.
                with Spinner(self.response):
                    placed = self.next_move(system_prompt, input_text, self.response, correction_loops)
            if self.finished:
                debug("Answer declared complete.")
                break
            self.response += placed
            if DEBUG:
                debug(f"Current response: {self.response}")
            else:
                # Redraw the answer-so-far in place. Clearing the line first lets a removed
                # word shrink it, instead of leaving stale characters behind.
                render_progress(self.response)
        if not DEBUG and self.response.strip():
            # The finishing move also ran under a Spinner, whose normal exit cleared the line.
            # Redraw the final answer before ending the line, so a completed run shows its
            # answer instead of a blank line.
            render_progress(self.response)
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.print_usage()
        return self.response

    def next_move(self, system_prompt: str, input_text: str, current_response: str, correction_loops: int) -> str:
        debug(f"\n[x] Determining next move. Answer so far: '{current_response}'")

        scout_state = {
            "system_prompt": system_prompt,
            "phase": "scouting: rate each word's chance to appear among the next words",
            "text": input_text,
            "response": current_response,
            "last_actions": self.last_actions,
        }
        # Scout the whole vocabulary in batches so no single request exceeds the token limit,
        # then merge the per-batch judgments and rank across all of them together.
        batches = chunked(all_words, SCOUT_BATCH)
        scout_nouls: dict[str, NoulAnswer] = {}
        for batch_index, batch in enumerate(batches, start=1):
            debug(f"Scouting batch {batch_index}/{len(batches)} ({len(batch)} words)")
            scout_response = self.system_one(state=scout_state, questions=scout_questions(batch))
            scout_nouls.update(scout_response.nouls)

        scouted_with_p = top_answers(scout_nouls, TOP_WORDS)
        scouted = [word for word, _ in scouted_with_p]
        debug(f"Scouted words: {scouted_with_p}")

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

        response = self.system_one(
            state=state,
            questions=arrangement_questions(phrases, current_response),
        )

        top = top_answers(response.nouls, 10)
        debug(f"Probabilities[0]: {top}")

        choice = top[0][0]
        choice_probability = top[0][1]

        if choice_probability <= 0.9:
            for correction_loop in range(correction_loops):
                previous_analysis = {}
                for key, probability in top:
                    previous_analysis[key] = probability

                response = self.system_one(
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
                debug(f"Probabilities[{correction_loop + 1}]: {top}")
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


SYSTEM_PROMPT = (
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
)

USAGE = (
    'Usage: uv run main.py [--debug] "<your question>"\n\n'
    "Composes an answer to the question one word at a time using Jev.\n"
    "  --debug   print the full scouting/arranging trace and token usage\n"
    'Example: uv run main.py "What is the capital of France?"'
)


def render_progress(text: str) -> None:
    """Redraw the answer-so-far on one line, clearing it first so removals shrink it."""
    sys.stdout.write("\r\x1b[2K" + text)
    sys.stdout.flush()


class Spinner:
    """Animate a loader after `prefix` on one line until the enclosed work finishes.

    Runs in a daemon thread; the main thread does the blocking work inside the
    `with` block. On a normal exit the line is cleared so the caller can redraw it;
    if the block is interrupted, the loader is dropped but `prefix` is left on the line.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, prefix: str, interval: float = 0.1):
        self.prefix = prefix
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        sep = " " if self.prefix else ""
        for frame in cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r\x1b[2K{self.prefix}{sep}{frame}")
            sys.stdout.flush()
            self._stop.wait(self.interval)

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        self._thread.join()
        if exc_type is None:
            # Normal exit: clear the line so the caller can redraw the updated answer.
            sys.stdout.write("\r\x1b[2K")
        else:
            # Interrupted: drop the loader but keep the answer-so-far on the line.
            sys.stdout.write("\r\x1b[2K" + self.prefix)
        sys.stdout.flush()


def watch_for_eof(stop: threading.Event) -> None:
    """Ctrl-D at an interactive terminal sends EOF on stdin; interrupt the main thread.

    Typed lines are ignored (this program takes its question from argv); only EOF ends
    the run. Started only when stdin is a TTY, so piped/background runs are unaffected.
    """
    try:
        while sys.stdin.readline():
            if stop.is_set():
                return
    except Exception:
        return
    if not stop.is_set():
        _thread.interrupt_main()


def main() -> None:
    global DEBUG
    args = sys.argv[1:]
    DEBUG = "--debug" in args
    positional = [arg for arg in args if arg != "--debug"]
    question = positional[0].strip() if positional else ""
    if not question:
        print(USAGE)
        return

    # Ctrl-D (stdin EOF) ends the run at an interactive terminal; Ctrl-C always does.
    stop_watch = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=watch_for_eof, args=(stop_watch,), daemon=True).start()

    client_instance = Client()
    try:
        response = client_instance.complete(
            system_prompt=SYSTEM_PROMPT,
            input_text=question,
            steps_count=50,
            correction_loops=1,
        )
    except (KeyboardInterrupt, EOFError):
        # Spinner.__exit__ has dropped the loader and left the answer-so-far on the line.
        # End that line (only default mode leaves it unterminated), then note the stop.
        if not DEBUG and client_instance.response.strip():
            sys.stdout.write("\n")
            sys.stdout.flush()
        print("(stopped by the user)")
        return
    finally:
        stop_watch.set()

    # Default mode already rendered the answer live during composition; here we only finalize
    # the empty case and, in debug mode, label the final answer beneath the trace.
    text = response.strip()
    if DEBUG:
        debug(f"\nQ: {question}")
        print(f"A: {text}" if text else "A: (no answer composed)")
    elif not text:
        print("(no answer composed)")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        # Backstop for an interrupt that escapes main() (e.g. during final output).
        print("\n(stopped by the user)")
        sys.exit(130)
