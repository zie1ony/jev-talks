"""Make a model that cannot write talk, by filling in sentence templates.

Jev, TypeSafe's System One model, does not generate text. It takes some state
plus typed questions and returns calibrated probabilities. diffuse.py rewrites
a row of loose words until it reads like a sentence; this script keeps that
loop and gives it a skeleton to work on. Every sentence starts as a template
with blanks, such as "_ is _ because _.", and a blank takes a word, or a
clause template with blanks of its own:

  1. plan    before every sentence, one request asks which sentence
             template it should follow. <END> is one of the options; when
             it is the most probable one, the response is over. With -n the
             response has exactly that many sentences, and <END> is not an
             option. Before the first sentence the same request scouts the
             vocabulary in groups of 250 for the words the response could
             use; the words of the question join the vocabulary first. The
             100 likeliest words, the 60 most frequent ones and the words
             of the question are the pool for every blank of the response.
  2. write   one request asks every blank "which word belongs here?" and
             "which sentence pattern would a clause here give?". A blank
             that prefers a clause becomes that clause, and the next pass
             asks about its blanks. Otherwise every blank puts up its best
             words, alone and in pairs, a second request asks a yes/no
             question about each sentence they can make, and the most
             probable sentence is written.
  3. judge   one request about the finished sentence: for every word in a
             blank, "is it wrong here?" and "is it a name?", and about the
             whole, "is this a good sentence for the answer?". A name gets
             a <CAP> tag in front of it.
  4. change  up to 3 of the most doubted words are blanked again, and step
             2 fills them anew, with the old word still in the running
  5. stop    once the sentence is at least 0.6 probably good, when no word
             is doubted, when rewriting leads back to a sentence that was
             judged before, or after 8 requests; the best sentence that was
             judged is kept

The three most probable templates are filled in parallel, and the sentence
that ranks highest goes into the response. The rank is how probably good Jev
finds the sentence, times the fourth root of the template's probability, so
of two about equally good sentences the more probable template wins. Without
-n, a sentence after the first is kept only if it is good; otherwise the
response ends there.

The response is a row of tokens: lowercase words, the marks . , ? ! and two
tags. <CAP> makes the next word capital, and every template starts with one.
<END> closes the response. "<CAP> paris is the capital of <CAP> france . <END>"
is printed as "Paris is the capital of France."

Every request is a state plus questions. The state holds what the questions
share: the rules, the question, the response so far and the draft. While the
blanks are filled the draft shows them numbered, so the question about a blank
is one short line. What a request costs is its options, about 6 tokens each,
which is why the vocabulary is scouted once per response, and why a blank
right after "a" or "the" is not offered the clauses that cannot follow it.

An answer averages two sentences, 4 seconds and 80,000 tokens, a third of a
cent: 23,000 for the first plan with its scouting, 3,500 for every later plan,
and three or four requests for most templates.

Usage:  uv run bumblebee.py "Why is the sky blue?" [-v | --debug] [-n SENTENCES] [--tries N] [--vocab FILE]
Needs TYPESAFE_API_KEY, from the environment or a .env file.
"""

import argparse
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import cycle, product
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Question, TypeSafeClient, TypeSafeError

# Settings --------------------------------------------------------------------

VOCABULARY = Path(__file__).with_name("top-english-3000.txt")
GROUP = 250  # words per scouting question, templates per planning question; a Choice takes at most 255 options
SIZE = 4  # sentences the response may have when -n does not say how many; <END> usually stops it earlier
PLACES = 12  # blanks a sentence may have; past that no blank becomes a clause
SCOUTED = 100  # scouted words that join the pool
COMMON = 60  # words from the top of the vocabulary file, its most frequent ones, that are always in the pool
SURE = 0.8  # a word at least this probable for its blank is written without being judged against others
VERIFIED = 120  # sentences one request may judge to settle the blanks that are not sure
CHANGES = 3  # words a pass may blank again
DOUBT = 0.5  # a word is blanked again only if it is at least this doubted
GOOD = 0.6  # a row stops once its sentence is at least this probably good; a later sentence below it is dropped
PRIOR = 0.25  # power of the template's probability in a sentence's rank; 0 ranks by Jev's verdict alone
MAX_PASSES = 8  # hard stop: requests per row
TRIES = 3  # templates filled in parallel for every sentence; the sentence that ranks highest is kept

CAP, END = "<CAP>", "<END>"  # tags among the words: the next word is capital, the response is over
MARKS = ".,?!"
BLANK = "___"  # how Jev and the terminal see a blank; the templates below spell it _
NOTHING, QUALITY = "<no word>", "<quality>"
# After one of these words only a word of its own phrase can come, so a blank there is not offered the clauses that
# start with a fixed word, such as "a because ___ is ___". That spares Jev four fifths of the patterns to read.
HEADS = {"a", "an", "the", "my", "your", "its", "their", "our", "no", "every", "very", "more", "most"}

RULES = (
    "An answer to `question` is being written sentence by sentence. `written` holds the sentences that are"
    " finished; it is empty while the first sentence is being written. `draft` is the sentence that comes next."
    " Every ___ in the draft is a blank that is filled later."
)
WRITING = (  # the rules while blanks are filled: the draft shows them numbered, so that a question can be short
    "An answer to `question` is being written sentence by sentence. `written` holds the sentences that are"
    " finished; it is empty while the first sentence is being written. `draft` is the sentence that comes next."
    " Every number in brackets in it is a blank that takes one word. A word question asks which word should stand"
    " at one of the blanks so that the finished sentence {aim}. Every option of it is a word to be placed"
    " literally; <no word> means the sentence is better without that blank."
)

# The templates ---------------------------------------------------------------

# A sentence template is written as it reads. _ is a blank: it takes a word of the pool, sometimes two, or one of
# the clause templates further down, whose blanks are filled the same way. A capital letter becomes a <CAP> tag in
# front of the word, and every sentence starts with one. "a" turns into "an" by the word that ends up after it.
SENTENCES = """
It is _.
It is a _.
It is the _.
It is a _ of _.
It is the _ of _.
It is a _ that _.
It is _ and _.
It is _ because _.
It is _ to _.
It is not _.
It is not a _.
It is also _.
It is like _.
It is one of the _.
It is when _.
It is what _.
It was _.
It has _.
It can _.
It can be _.
It may be _.
It will _.
It means _.
It means that _.
It depends.
It depends on _.
It takes _.
It comes from _.
It happens when _.
It is said that _.
It is hard to _.
It seems that _.
It is in _.
It is about _.
It is called _.
It is made of _.
It is used to _.
It is used for _.
It is caused by _.
It works by _.
It has a _.
It helps you _.
This is _.
This is a _.
This is why _.
This means that _.
That is _.
That is why _.
That is because _.
There is _.
There is a _.
There is no _.
There are _.
There are many _.
They are _.
They are _ and _.
They _.
They _ _.
They _ because _.
They _ when _.
They do it to _.
They can _.
They have _.
_ is _.
_ is a _.
_ is the _.
_ is a _ of _.
_ is the _ of _.
_ is a _ that _.
_ is a _ where _.
_ is _ and _.
_ is _ because _.
_ is not _.
_ is _ than _.
_ is when _.
_ is what _.
_ is to _.
_ is like _.
_ is one of the _.
_ is the most _ _.
_ are _.
_ are _ because _.
_ are not _.
_ was _.
_ were _.
_ has _.
_ have _.
_ can _.
_ cannot _.
_ will _.
_ would _.
_ should _.
_ must _.
_ may _.
_ means _.
_ makes _ _.
_ comes from _.
_ happens when _.
_ depends on _.
_.
_ _.
_ _ _.
_ _ _ _.
_ _ _ _ _.
_ _ because _.
_ _ when _.
_ _ to _.
_ _ and _.
The _ is _.
The _ is a _.
The _ is the _.
The _ is _ because _.
The _ is to _.
The _ is that _.
The _ of _ is _.
The _ of _ is to _.
The _ of the _ is _.
The _ are _.
The _ _.
The _ _ _.
The _ _ is _.
The _ _ the _.
The _ has _.
The _ can _.
The most _ _ is _.
The most _ _ in _ is _.
The best _ is _.
The best way is to _.
The best way to _ is to _.
The reason is _.
The reason is that _.
The answer is _.
A _ is a _.
A _ is a _ of _.
A _ is a _ that _.
A _ is _.
A _ is not a _.
A _ _ is a _.
A _ _ is a _ that _.
A _ _ is not a _.
A _ can _.
Some _ are _.
Some _ _.
Some people _.
Some people believe that _.
Some say _.
Many _ are _.
Many people _.
Many people believe that _.
Most _ are _.
Most people _.
All _ are _.
Every _ is _.
People _.
People _ because _.
Nobody knows.
Nobody knows _.
No one knows _.
I am _.
I am a _.
I am the _.
I am _ and _.
I am a _ that _.
I am a _ _.
I am not _.
I am not a _.
I am not sure.
I am here to _.
I can _.
I can _ you _.
I cannot _.
I do not _.
I do not know.
I do not know _.
I do not have _.
I have _.
I have no _.
I think _.
I think that _.
I think so.
I believe that _.
I like _.
I want to _.
I will _.
I would say _.
I was made to _.
My _ is _.
You are _.
You can _.
You can _ by _.
You can _ and _.
You cannot _.
You should _.
You should _ and _.
You must _.
You need _.
You need to _.
You have to _.
You will _.
We are _.
We can _.
We do not know.
We do not know _.
We _ _.
Yes.
Yes, it is.
Yes, it is _.
Yes, it is a _.
Yes, I am.
Yes, I am _.
Yes, I can.
Yes, _.
Yes, _ is _.
Yes, but _.
Yes, because _.
No.
No, it is not.
No, it is not _.
No, it is a _.
No, I am not.
No, I am not _.
No, I am a _.
No, I cannot.
No, _.
No, _ is not _.
No, but _.
No, because _.
Maybe.
Maybe _.
Not really.
Of course.
Perhaps _.
Probably _.
Because _.
Because _ is _.
Because of _.
Because of the _.
Because it is _.
Because they _.
Because they are _.
To _.
To _ and _.
To _, you must _.
To _, you need to _.
By _.
By _ and _.
If _, _.
If you _, you will _.
When _, _.
When you _, you _.
After _, _.
First _, then _.
Just _.
Try to _.
Do not _.
Never _.
Always _.
Please _.
_ and _.
_, _ and _.
_, but _.
_, so _.
Not _, but _.
Also, _.
But _.
However, _.
For example, _.
In fact, _.
In other words, _.
Then _.
What if _ is _?
What is _?
What do you _?
What about _?
Who knows?
Why not?
Do you _?
Are you _?
Can you _?
Hello!
Hello, I am _.
Thank you!
Thank you for _.
Sorry, _.
What a _!
_!
_ _!
_ _ _?
""".split("\n")[1:-1]

# A clause template can stand wherever a blank is; it has no capital and no end of its own.
CLAUSES = """
the _
a _
the _ _
a _ _
the _ of _
a _ of _
the _ of the _
_ and _
_ or _
_, _ and _
_ of _
_ _
_ _ _
_ _ _ _
my _
your _
its _
their _
our _
some _
many _
all _
no _
more _
very _
not _
to _
to _ _
to _ the _
to be _
to _ and _
_ the _
_ a _
_ it
_ them
_ you
in _
in the _
in a _
on the _
at the _
of the _
from the _
from _
with _
with a _
with the _
without _
for _
for the _
by _
by the _
about _
like a _
as a _
because _ is _
because of _
because _ _ _
because it is _
because they are _
that _
that _ _
that _ is _
that is _
that are _
that can _
that you _
which is _
which _ _
who _
who is _
who _ _
what _ _
what _ is
what you _
when _ _
when you _
when it is _
when they are _
where _ _
where _ is
why _ _
how _ _
how to _
if _ _
if you _
if it is _
so that _ _
so _ that _
as _ as _
while _ _
after _ _
before _ _
until _ _
than _
_ is _
_ are _
_ can _
it is _
it _
it _ _
they are _
they _
they _ _
you _
you _ _
you are _
you can _
we _
we _ _
we are _
there is _
there are _
is _
are _
can _
will _
do not _
does not _
_ that _
_ who _
_ to _
_ of the _
_ in the _
_ for _
_ with _
_ from _
""".split("\n")[1:-1]

VERBOSE = 0  # 1 with -v: every pass of every row; 2 with --debug: every proposal and every verdict as well
STOP = threading.Event()  # set on Ctrl-C or an error, so that every row stops after its current request


# The row ---------------------------------------------------------------------


@dataclass(eq=False)
class Word:
    """One blank of the row: the word it holds, and what the next pass and the animation need to know about it."""

    text: str = ""  # empty while the blank is open
    tried: list[str] = field(default_factory=list)  # the words this place already held, itself last; none returns
    doubt: float = 0.0  # how wrong the latest pass found it
    fresh: bool = True  # whether the latest pass put it there


Row = list["str | Word"]  # the tokens of a sentence in order: fixed words, marks and tags, and a Word for every blank


class Diffusion:
    """One sentence: a template whose blanks are filled, judged and blanked again in passes."""

    def __init__(self, jev: "Jev", pool: list[str], question: str, written: str, template: str, prior: float, label: str):
        self.jev, self.pool, self.question, self.written, self.label = jev, pool, question, written, label
        self.template, self.prior = template, prior  # the prior is how probable Jev found the template
        self.row = pattern(template, sentence=True)
        self.passes = 0  # requests made so far
        self.seen: set[str] = set()  # sentences that were already judged
        self.spinner: Spinner | None = None

    def run(self, spinner: "Spinner | None" = None) -> tuple[list[str], float]:
        """Fill the blanks, then blank a few doubted words again per pass until Jev calls the sentence good.

        Returns the tokens of the best sentence that was judged, and how good Jev found it.
        """
        self.spinner = spinner
        best, quality = [], -1.0
        while self.passes < MAX_PASSES and not STOP.is_set():
            if self.blanks:
                self.write()
                if draft(self.row).lower() in self.seen:  # rewriting led back to a sentence that was judged
                    break
                continue
            good = self.judge()
            if good > quality:
                best, quality = spelled(self.row), good
            self.log(f"pass {self.passes}, good {good:.2f}: {draft(self.row)!r}")
            self.log("  doubts:", ", ".join(f"{place.text} {place.doubt:.2f}" for place in self.places) or "none")
            self.tell(f"pass {self.passes}, good {good:.2f}")
            doubted = [] if good >= GOOD or MAX_PASSES - self.passes < 3 else self.choose()
            if not doubted:
                break
            self.change(doubted)
        if self.spinner and best:  # the row ends on the best sentence it had, not on its last attempt
            self.spinner.text = (self.written + " " if self.written else "") + draft(best)
            self.spinner.status = f"good {quality:.2f}"
        return best, quality

    def write(self) -> None:
        """Fill the blanks: one request asks every blank which word belongs there and which clause would.

        Its state shows the draft with the blanks numbered, so the question about a blank's word is one line.
        A blank that prefers a clause becomes that clause, and its blanks are asked about in the next pass.
        Otherwise every blank puts up its best words, alone and in pairs; a second request judges the sentences
        they can make, and the best one is written.
        """
        blanks = self.blanks
        room = MAX_PASSES - self.passes > 3 and len(self.places) < PLACES
        aim = aim_of(self.written)
        questions: dict[str, Question] = {}
        for i, blank in enumerate(blanks):
            questions[f"word{i}"] = word_question(i + 1, self.pool)
            if room:
                questions[f"frame{i}"] = frame_question(self.row, blank, aim)
        answers = self.ask(questions, WRITING.format(aim=aim), numbered=True)
        offers = [self.offers(answers, i, blank) for i, blank in enumerate(blanks)]
        numbered = draft(self.row, numbered=True)
        for i in range(len(blanks)):
            self.log(f"pass {self.passes}, [{i + 1}] of {numbered!r}:", show(offers[i], top(offers[i], 8)), level=2)
        for place in self.places:
            place.fresh = False
        clauses = {i: best for i, offer in enumerate(offers) for best in top(offer, 1) if best in CLAUSES}
        if clauses:
            for i, clause in clauses.items():
                if len(self.places) < PLACES:
                    self.row = swap(self.row, blanks[i], pattern(clause))
            self.log(f"pass {self.passes}, clauses: {draft(self.row)!r}")
            return
        unsure = sum(1 for i, blank in enumerate(blanks) if blank.tried or max(offers[i].values(), default=0) < SURE)
        count = min(10, max(2, int(VERIFIED ** (1 / max(unsure, 1)))))  # so that the sentences stay within VERIFIED
        options = [candidates(offers[i], blank, count) for i, blank in enumerate(blanks)]
        combos = [combo for combo in product(*options) if distinct(combo)][:VERIFIED] or [first_distinct(options)]
        if len(combos) > 1:
            sentences = [draft(self.filled(blanks, combo)) for combo in combos]
            verdicts = self.ask({f"sentence{k}": good_question(text, self.written) for k, text in enumerate(sentences)})
            ranked = sorted(range(len(combos)), key=lambda k: verdicts[f"sentence{k}"].noul, reverse=True)
            shown = ranked if VERBOSE > 1 else ranked[:4]
            verdict = ", ".join(f"{sentences[k]!r} {verdicts[f'sentence{k}'].noul:.2f}" for k in shown)
            self.log(f"pass {self.passes}, {len(combos)} sentences: {verdict}")
            combos = [combos[ranked[0]]]
        self.row = self.filled(blanks, combos[0])
        self.log(f"pass {self.passes}, wrote: {draft(self.row)!r}")

    def offers(self, answers: dict, i: int, blank: Word) -> dict[str, float]:
        """What could stand at a blank, by probability: every word of the pool, no word, and every clause.

        A clause counts by how probable the sentence pattern with it is; a word shares the probability of the
        pattern staying as it is. What the blank held before is left out.
        """
        found = dict(answers[f"word{i}"].probabilities)
        if f"frame{i}" in answers:
            patterns = answers[f"frame{i}"].probabilities
            clauses = frames(self.row, blank)
            stay = next(patterns[frame] for frame, clause in clauses.items() if not clause)
            found = {word: stay * p for word, p in found.items()}
            found.update({clause: patterns[frame] for frame, clause in clauses.items() if clause})
        return {new: p for new, p in found.items() if p and new not in blank.tried}

    def filled(self, blanks: list[Word], combo: tuple[tuple[str, ...], ...]) -> Row:
        """The row with words in its blanks: for every blank no word, one word or two."""
        row: Row = []
        for item in self.row:
            words = next((words for blank, words in zip(blanks, combo, strict=True) if blank is item), None)
            if words is None:
                row.append(item)
            elif len(words) == 1:
                row.append(Word(words[0], [old for old in item.tried if old != words[0]] + [words[0]]))
            else:
                row += [Word(word, [word]) for word in words]
        return row

    def judge(self) -> float:
        """One request about the whole sentence: is each word wrong, is it a name, and how good is it all."""
        places = self.places
        questions: dict[str, Question] = {QUALITY: good_question(draft(self.row), self.written)}
        for i, place in enumerate(places):
            questions[f"doubt{i}"] = doubt_question(place.text)
            questions[f"capital{i}"] = capital_question(place.text)
        self.seen.add(draft(self.row).lower())
        answers = self.ask(questions)
        for i, place in enumerate(places):
            place.doubt = answers[f"doubt{i}"].noul
            self.capitalize(place, answers[f"capital{i}"].noul >= 0.5)
        names = ", ".join(f"{place.text} {answers[f'capital{i}'].noul:.2f}" for i, place in enumerate(places))
        self.log(f"pass {self.passes}, names:", names or "no word to ask about", level=2)
        return answers[QUALITY].noul

    def capitalize(self, place: Word, capital: bool) -> None:
        """Put a <CAP> tag in front of a word, or take it away. The tag that starts the sentence stays."""
        i = self.index(place)
        tagged = i > 0 and self.row[i - 1] == CAP
        if capital and not tagged:
            self.row.insert(i, CAP)
        elif tagged and not capital and i > 1:
            del self.row[i - 1]

    def choose(self) -> list[Word]:
        """Pick the few words this pass blanks again: the most doubted ones, never two neighbours."""
        places = self.places
        doubted: list[int] = []
        for i in sorted(range(len(places)), key=lambda i: places[i].doubt, reverse=True):
            if places[i].doubt >= DOUBT and len(doubted) < CHANGES and all(abs(i - other) > 1 for other in doubted):
                doubted.append(i)
        return [places[i] for i in doubted]

    def change(self, doubted: list[Word]) -> None:
        """Blank the doubted words again. A blank remembers what it held."""
        for place in doubted:
            self.capitalize(place, False)
            place.text = ""
        self.log(f"  -> blanked again: {draft(self.row)!r}")

    def ask(self, questions: dict[str, Question], rules: str = RULES, numbered: bool = False) -> dict:
        """One pass: a request whose state holds the rules, the question, the response so far and the sentence."""
        self.passes += 1
        self.tell(f"pass {self.passes}")
        sentence = draft(self.row, numbered=numbered)
        return self.jev({"rules": rules, "question": self.question, "written": self.written, "draft": sentence}, questions)

    def index(self, place: Word) -> int:
        return next(i for i, item in enumerate(self.row) if item is place)

    def tell(self, status: str) -> None:
        """Update the terminal animation, when there is one: just-placed words bold, doubted words and blanks dim."""
        if self.spinner:
            self.spinner.text = (self.written + " " if self.written else "") + draft(self.row, styled=True)
            self.spinner.status = status

    def log(self, *parts: object, level: int = 1) -> None:
        log(*((f"[{self.label}]",) if self.label else ()), *parts, level=level)

    @property
    def places(self) -> list[Word]:
        return [item for item in self.row if isinstance(item, Word)]

    @property
    def blanks(self) -> list[Word]:
        return [place for place in self.places if not place.text]


def candidates(offers: dict[str, float], blank: Word, count: int) -> list[tuple[str, ...]]:
    """What a blank puts up for the sentences that get judged: no word, one word or two, the likeliest first.

    A sure word stands alone. A blank that held a word before puts it up again, alone and next to every new word,
    since a doubted word may just have lacked company.
    """
    words = [new for new in top(offers, len(offers)) if new not in CLAUSES][:4]
    old = blank.tried[-1:]  # the word a blank held before it was blanked again
    if words and not old and offers[words[0]] >= SURE:
        return [as_words(words[0])]
    found: list[tuple[str, ...]] = [tuple(old)] if old or not words else []
    for k, word in enumerate(words):
        found.append(as_words(word))
        for other in old or words[:k]:
            if NOTHING not in (word, other):
                found += [(other, word), (word, other)]
    return found[:count]


def as_words(new: str) -> tuple[str, ...]:
    return () if new == NOTHING else (new,)


def distinct(combo: tuple[tuple[str, ...], ...]) -> bool:
    """No word twice among the blanks of one sentence."""
    words = [word for words in combo for word in words]
    return len(set(words)) == len(words)


def first_distinct(options: list[list[tuple[str, ...]]]) -> tuple[tuple[str, ...], ...]:
    """Every blank's likeliest candidate that repeats no word of the blanks before it, or no word at all."""
    used: set[str] = set()
    combo = []
    for found in options:
        combo.append(next((words for words in found if not used & set(words)), ()))
        used.update(combo[-1])
    return tuple(combo)


def pattern(template: str, sentence: bool = False) -> Row:
    """A template as a row: its words and marks as tokens, a <CAP> tag for a capital letter, a Word for every _."""
    row: Row = []
    for token in re.findall(r"_|[^\W_]+(?:['’-][^\W_]+)*|[.,?!]", template):
        if token == "_":
            row.append(Word())
        elif token[0].isupper():
            row += [CAP, token.lower()]
        else:
            row.append(token)
    if sentence and row[0] != CAP:
        row.insert(0, CAP)
    return row


AN = re.compile(r"(?!uni|use|usu|uti|eu|one|once)[aeiou]|hour|honest|honor|heir")  # words that take "an"


def spelled(row: Row, hidden: Word | None = None) -> list[str]:
    """The row as plain tokens: the word of every blank, "" for an open one, and "a" or "an" by the word after it.

    The article in front of a `hidden` place stays as it is, so that it does not give the word away.
    """
    tokens = [item.text if isinstance(item, Word) else item for item in row]
    for k, token in enumerate(tokens):
        after = next((k + j for j, other in enumerate(tokens[k + 1 :], 1) if other != CAP), None)
        if token in ("a", "an") and after is not None and tokens[after][:1].isalpha() and row[after] is not hidden:
            tokens[k] = "an" if AN.match(tokens[after]) else "a"
    return tokens


def draft(row: Row, marked: Word | None = None, reveal: bool = False, styled: bool = False, numbered: bool = False) -> str:
    """Tokens as text, the way Jev and the terminal read them.

    A <CAP> tag makes the next word capital, a mark hugs the word before it, an open blank is ___ and <END> ends
    the text. One place can be marked: it is hidden as [?], or shown as [word] with `reveal`. `styled` dims
    doubted words and blanks and makes just-placed words bold. `numbered` shows the open blanks as [1], [2], ...
    """
    parts: list[str] = []
    capital, blanks = False, 0
    for item, word in zip(row, spelled(row, None if reveal else marked), strict=True):
        if item == CAP:
            capital = True
            continue
        if item == END:
            break
        blanks += isinstance(item, Word) and not word
        shown = word or (f"[{blanks}]" if numbered else BLANK)
        if capital and word:
            shown = shown[0].upper() + shown[1:]
        capital = False
        if item is marked:
            shown = f"[{shown}]" if reveal else "[?]"
        elif styled and isinstance(item, Word):
            shown = ("\x1b[2m" if item.doubt >= DOUBT or not word else "\x1b[1m" if item.fresh else "") + shown + "\x1b[0m"
        if shown in MARKS and parts:
            parts[-1] += shown
        else:
            parts.append(shown)
    return " ".join(parts)


def swap(row: Row, place: Word, items: Row) -> Row:
    """The row with other tokens where a place was."""
    i = next(k for k, item in enumerate(row) if item is place)
    return row[:i] + items + row[i + 1 :]


def frames(row: Row, place: Word) -> dict[str, str]:
    """The sentence patterns a place can lead to: one blank there, or a clause template there.

    Right after one of the HEADS, only the clauses that start with a blank are among them.
    """
    before = [item.text if isinstance(item, Word) else item for item in row[: row.index(place)] if item != CAP]
    bound = bool(before) and before[-1] in HEADS
    options = {draft(swap(row, place, [Word()])): ""}
    for clause in CLAUSES:
        if not bound or clause.startswith("_"):
            options.setdefault(draft(swap(row, place, pattern(clause))), clause)
    return options


# The questions ---------------------------------------------------------------


def aim_of(written: str) -> str:
    if written:
        return "reads as natural, correct English and is a good next sentence after `written` in the answer to `question`"
    return "reads as natural, correct English and answers `question` well"


def scouting_question(words: list[str]) -> Choice:
    return Choice(
        instructions="Which of these words is the most likely to appear in a good answer to `question`?"
        " Every option is a vocabulary word to be used literally, not a meta answer.",
        criteria={word: None for word in words},
    )


def planning_question(templates: list[str], written: str, ending: bool) -> Choice:
    options: dict[str, str | None] = {draft(pattern(template, sentence=True)): None for template in templates}
    if written:
        text = (
            "A good answer to `question` is being written sentence by sentence; `written` holds the sentences so far."
            " Which pattern should the next sentence follow, the one that comes right after `written`?"
        )
        if ending:
            options[END] = "No more sentences: `written` is already a complete answer to `question`."
    else:
        text = "A good answer to `question` is being written. Which pattern should its first sentence follow?"
    return Choice(
        instructions=text + " Every ___ in a pattern stands for a word or a short phrase that is filled in afterwards;"
        " the other words of the pattern are used literally.",
        criteria=options,
    )


def word_question(number: int, pool: list[str]) -> Choice:
    """The question about one blank. It is short: the draft with its numbered blanks and what to aim for are state."""
    return Choice(
        instructions=f"Which word should stand at [{number}]?", criteria={word: None for word in [*pool, NOTHING]}
    )


def frame_question(row: Row, place: Word, aim: str) -> Choice:
    options = frames(row, place)
    blank = Word()
    return Choice(
        instructions=f"The sentence will follow the pattern '{next(iter(options))}'. Here it is again with one blank"
        f" marked: '{draft(swap(row, place, [blank]), blank)}'. Every ___ takes exactly one word. If one"
        " word is enough at [?], the pattern stays as it is; if a phrase belongs there, the pattern needs more blanks"
        " and words at that place. Which of these patterns should the sentence follow so that, once every ___ is"
        f" filled with one word, it {aim}?",
        criteria={frame: None for frame in options},
    )


def doubt_question(word: str) -> Noul:
    return Noul(
        instructions=f"Is the word {word!r} wrong in `draft`: out of place, ungrammatical, repeated, or making the"
        " answer incorrect?"
    )


def capital_question(word: str) -> Noul:
    return Noul(
        instructions=f"Does correct English write the word {word!r} of `draft` with a capital first letter wherever"
        " it stands in a sentence: is it the name of a person, a place, a people, a language, a day or a month, or"
        " the word I?"
    )


def good_question(sentence: str, written: str) -> Noul:
    if written:
        return Noul(
            instructions=f"Is {sentence!r} a complete, natural and correct sentence, and a good next sentence after"
            " `written` in the answer to `question`: one that adds something and does not repeat it?"
        )
    return Noul(instructions=f"Is {sentence!r} a complete, natural and correct answer to `question`?")


def top(probabilities: dict[str, float], n: int) -> list[str]:
    return sorted(probabilities, key=probabilities.get, reverse=True)[:n]


def show(probabilities: dict[str, float], keys: list[str]) -> str:
    return ", ".join(f"{key} {probabilities[key]:.2f}" for key in keys)


def log(*parts: object, level: int = 1) -> None:
    if VERBOSE >= level:
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


def plan(
    jev: Jev, vocabulary: list[str], basics: list[str], question: str, written: str, ending: bool, pool: list[str]
) -> tuple[list[str], dict[str, float], float]:
    """One request before every sentence: which template it should follow.

    Before the first sentence there is no `pool` yet, and the same request scouts the vocabulary for one: the
    words the whole response could use. That is 3,000 options, most of the tokens of the request, so it is asked
    once. <END> is among the templates only if the response may end here, which is what `ending` says.
    Returns the pool, the templates by probability, and how probable <END> is.
    """
    m = -(-len(SENTENCES) // GROUP)  # as few groups as will fit, all of the same size
    questions: dict[str, Question] = {f"plan{i}": planning_question(SENTENCES[i::m], written, ending) for i in range(m)}
    words = list(vocabulary)
    random.Random(0).shuffle(words)  # so every scouting group mixes common and rare words
    n = 0 if pool else -(-len(words) // GROUP)
    questions.update({f"group{i}": scouting_question(words[i::n]) for i in range(n)})
    answers = jev({"question": question, "written": written}, questions)
    shown = {draft(pattern(template, sentence=True)): template for template in SENTENCES}
    templates = {shown[p]: v for i in range(m) for p, v in answers[f"plan{i}"].probabilities.items() if p != END}
    end = min(answers[f"plan{i}"].probabilities.get(END, 0.0) for i in range(m))
    if not pool:
        scouted = {word: p for i in range(n) for word, p in answers[f"group{i}"].probabilities.items()}
        pool = list(dict.fromkeys(basics + top(scouted, SCOUTED)))[:254]  # plus NOTHING: 255 options at most
        log("\n  scouted:", show(scouted, top(scouted, 10)), "...")
        log("  pool:", " ".join(pool), level=2)
    best = top(templates, 20 if VERBOSE > 1 else 5)
    log(("" if n else "\n") + "  templates:", show(templates, best), f"... {END} {end:.2f}")
    return pool, templates, end


def fill(rows: list[Diffusion], spinner: Spinner) -> list[tuple[list[str], float]]:
    """Run the rows of one sentence in parallel. The animation follows the first row."""
    with ThreadPoolExecutor() as executor:
        try:
            return list(executor.map(lambda row: row.run(spinner if row is rows[0] else None), rows))
        except BaseException:  # Ctrl-C, or an error in one row: the others should not run on
            STOP.set()
            raise


def keep(rows: list[Diffusion], results: list[tuple[list[str], float]], written: str) -> int | None:
    """Which row's sentence goes into the response: the one that ranks highest.

    The rank is how probably good Jev found the sentence, times a root of how probable it found the template. So
    of two sentences that are about equally good, the one from the more probable template is kept; a sentence from
    a less probable template has to be clearly better. A sentence that the response already has is never kept.
    """
    ranks = [good * row.prior**PRIOR for row, (_, good) in zip(rows, results, strict=True)]
    fresh = [k for k, (tokens, _) in enumerate(results) if tokens and draft(tokens).lower() not in written.lower()]
    kept = max(fresh, key=lambda k: ranks[k], default=None)
    for k, (row, (tokens, good)) in enumerate(zip(rows, results, strict=True)):
        mark = "kept" if k == kept else "    "
        log(f"{mark} {row.label or '-'} rank {ranks[k]:.2f}, good {good:.2f}  {draft(tokens)!r}  from {row.template!r}")
    return kept


def compose(jev: Jev, words: list[str], question: str, tries: int, size: int, forced: bool = False) -> str:
    """Write the response sentence by sentence: for each, fill the `tries` most probable templates and keep one.

    The response has up to `size` sentences and ends sooner when <END> is the most probable template or the next
    sentence is not good. When it is `forced`, it has exactly `size` sentences.
    """
    asked = [word.lower() for word in re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", question)]
    vocabulary = list(dict.fromkeys(words + asked))  # every word of the question can be used, whatever the word list has
    basics = list(dict.fromkeys(asked + words[:COMMON]))  # these are in the pool of every sentence
    response: list[str] = []
    used: set[str] = set()  # the templates of the sentences so far; none is used twice
    pool: list[str] = []  # the words every blank chooses from; scouted before the first sentence, kept for the rest
    note = ""
    with Spinner("", "starting") as spinner:
        for n in range(size):
            written = draft(response)
            spinner.text, spinner.status = written, f"{note}planning sentence {n + 1}"
            pool, templates, end = plan(jev, vocabulary, basics, question, written, not forced, pool)
            ranked = [template for template in top(templates, len(templates)) if template not in used]
            if response and not forced and end >= max((templates[template] for template in ranked), default=0.0):
                log(f"{END} is the most probable template")
                break
            kept = None
            while kept is None and ranked:  # a forced sentence goes on to the next templates until one is written
                labels = "ABCDEFGH" if tries > 1 else [""]
                rows = [
                    Diffusion(jev, pool, question, written, template, templates[template], labels[k % len(labels)])
                    for k, template in enumerate(ranked[:tries])
                ]
                results = fill(rows, spinner)
                kept = keep(rows, results, written)
                ranked = ranked[tries:] if forced else []
            if kept is None or (response and not forced and results[kept][1] < GOOD):
                log("no sentence came out of it" if kept is None else "it is not good enough to keep")
                break
            response += results[kept][0]
            used.add(rows[kept].template)
            spinner.text = draft(response)
            first, good = results[0][1], results[kept][1]  # the animation followed the first row
            note = f"another template did better, {good:.2f} against {max(first, 0):.2f}; " if good > first else ""
        response.append(END)
        spinner.text = draft(response)  # a sentence that was not kept leaves the screen
        log("\n" + " ".join(response))
    return draft(response)


def main() -> None:
    global VERBOSE
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog="Needs TYPESAFE_API_KEY, from the environment or a .env file."
    )
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every pass's judgments and the token usage")
    parser.add_argument("--debug", action="store_true", help="print every proposal and verdict as well; no animation")
    parser.add_argument("-n", dest="size", type=int, metavar="SENTENCES", help="write exactly this many sentences")
    parser.add_argument("--tries", type=int, default=TRIES, help="templates to fill in parallel for every sentence")
    parser.add_argument("--vocab", type=Path, default=VOCABULARY, help="word list to compose from, one word per line")
    args = parser.parse_args()
    if args.tries < 1:
        parser.error("--tries must be at least 1")
    if args.size is not None and args.size < 1:
        parser.error("-n must be at least 1")
    VERBOSE = 2 if args.debug else int(args.verbose)

    load_dotenv()
    try:
        words = list(dict.fromkeys(args.vocab.read_text().lower().split()))
        if not words:
            sys.exit(f"error: {args.vocab} has no words")
        jev = Jev()
        answer = compose(jev, words, args.question, args.tries, args.size or SIZE, forced=args.size is not None)
    except KeyboardInterrupt:
        sys.exit("\n(stopped)")
    except (TypeSafeError, OSError) as error:
        sys.exit(f"error: {error}")
    print(("\r\x1b[2K" if animated() else "") + (answer or "(no answer)"))
    log(f"\n{jev.requests} requests, {jev.tokens:,} input tokens")


if __name__ == "__main__":
    main()
