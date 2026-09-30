# jev-talks

**A model that cannot write, made to talk.**

![jev-talks composing an answer word by word](demo.gif)

[Jev](https://typesafe.ai) is a *System One* model. You hand it some state and a set of typed yes/no questions, and it returns a calibrated probability for each one. That is all it does. It has no decoder, it never emits a token of text, and its own docs list "generation" as a known failure mode with the advice to use a different model.

This repo makes it talk anyway. Text generation becomes a game of moves, and every move is decided by a few thousand yes/no questions. One Python file, no other model involved.

## How it works

```
                 question · answer so far · last 10 moves
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │  one round                 ▼                            │
       │   ┌──────────────────────────────────────────────┐      │
       │   │ 1. scout     3,000 common English words      │      │
       │   │    "could ‹word› be one of the next 3 words  │      │
       │   │     of the answer?"                          │      │
       │   │    3 parallel requests × 1,000 questions     │      │
       │   └───────────────────┬──────────────────────────┘      │
       │                       │  the 30 most probable words     │
       │   ┌───────────────────▼──────────────────────────┐      │
       │   │ 2. arrange   900 phrases of one or two words │      │
       │   │    "placing ‹phrase› would change the answer │      │
       │   │     to ‹answer + phrase›. good continuation?"│      │
       │   │    + take the last word back / pass / finish │      │
       │   │    1 request × 903 questions                 │      │
       │   └───────────────────┬──────────────────────────┘      │
       │                       │  the single most probable move  │
       │   ┌───────────────────▼──────────────────────────┐      │
       │   │ 3. apply     append the phrase, or pop the   │      │
       │   │              last word, or do nothing        │      │
       │   └───────────────────┬──────────────────────────┘      │
       └───────────────────────┼─────────────────────────────────┘
                               │  until "finish" wins, or 50 rounds
                               ▼
                             answer
```

Every question is a [Noul](https://docs.typesafe.ai/primitives/noul): a statement Jev rates with the probability that it is true. Questions in one request are judged independently and in parallel, so a round is four HTTP requests and about two seconds.

1. **Scout.** For each of the 3,000 words in the vocabulary, ask whether it could be one of the next three words of the answer. Keep the 30 most probable.
2. **Arrange.** Build every one- and two-word phrase from those 30 words, 900 in total. For each, ask whether appending it would be a good continuation. Three more questions offer the other moves: take the last word back, pass this round, or declare the answer finished.
3. **Apply** whichever single question scored highest, and go again.

The state sent with every question holds the rules of the game, the question being answered, the answer so far, and the last ten moves, so the model can keep extending the sentence it started instead of looping.

## Run it

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and a TypeSafe API key from [console.typesafe.ai](https://console.typesafe.ai).

```sh
cp .env.example .env            # then paste your key after TYPESAFE_API_KEY=
uv run talk.py "Why is the sky blue?"
```

Flags:

```sh
uv run talk.py -v "Why is the sky blue?"                            # print every round's judgments and the token count
uv run talk.py --vocab basic-english-850.txt "Why is the sky blue?"  # speak Ogden's Basic English instead
```

Stop a run with Ctrl-C. Redirect stdout and you get just the answer, no animation.

## What it says

Real answers, unedited, from the runs used to test this version:

| Question | Answer |
| --- | --- |
| What is the capital of France? | paris |
| Who are you? | i am an assistant |
| Are you conscious? | no |
| Is a hot dog a sandwich? | generally yes |
| Why is the sky blue? | because the air is a filter that allows light through only blue colors |
| What is the meaning of life? | is nothing but process of continuing forward |
| What is love? | is a feeling of connection between people |
| What happens after we die? | is unknown |
| How do I become rich? | start by working hard then save enough money for investment to grow your portfolio |

Ask the same question twice and you may get a different answer; on another run the sky was blue "because the air contains certain components responsible for making it blue". It is also capable of a sixty-word run-on sentence about a "peter flat" when asked about the most successful scam in history. "Finish" competes with 900 phrases every round and sometimes keeps losing by a hair.

## Cost and speed

A round is 4 requests and roughly 110,000 input tokens, which is about half a cent at Jev's list price, and takes about two seconds. Rounds place one or two words, so a twenty-word answer is around 13 rounds, 30 seconds, and 7 cents. Output tokens are free.

## Knobs

All at the top of `talk.py`:

| Setting | Default | What it does |
| --- | --- | --- |
| `LOOKAHEAD` | 3 | scouting asks whether a word could be among the next *n* words |
| `TOP_WORDS` | 30 | scouted words that advance to arranging |
| `PHRASE` | 2 | longest phrase placed in one move; 3 would mean 25,260 phrases per round |
| `TAIL` | 10 | words of the answer quoted in each phrase question, keeping requests inside Jev's 64k-token window |
| `BATCH` | 1000 | words per scouting request |
| `MEMORY` | 10 | recent moves the model gets to see |
| `MAX_ROUNDS` | 50 | hard stop |

## Caveats

- This is a toy. It exists to find out what a decision-only model does when you corner it into generating, not to compete with a language model that is a few hundred times cheaper per word.
- The vocabulary is a web-frequency list of 3,000 words, so it knows "php", "llc" and "faq" but not "purr". Swap in any file with one word per line.
- The vocabulary is lowercase and has no punctuation, so the answers read like telegrams.
- The take-back and pass moves are legal but Jev never picked either in the test runs for this version. Every answer above was written strictly left to right, and the last word of a long answer usually loses to "finish" only by a hair.

## Acknowledgements

Built on [TypeSafe](https://typesafe.ai) and its [Python SDK](https://docs.typesafe.ai/sdk/python). The 850-word list is Ogden's Basic English.
