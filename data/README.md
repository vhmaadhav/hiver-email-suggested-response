# Data

This folder holds the dataset CSV. **The CSV is gitignored** - fetch it instead of committing it.

```bash
uv run python scripts/fetch_data.py
```

## Source

[`oanannv/enron-email-reply-dataset`](https://www.kaggle.com/datasets/oanannv/enron-email-reply-dataset)
on Kaggle - approximately 15,377 cleaned email/reply pairs derived from the
public **Enron Email Dataset** (released into the public domain by FERC after
the 2002 investigation, and distributed for research by CMU at
<https://www.cs.cmu.edu/~enron/>).

Please check the licence stated on the Kaggle dataset page before redistributing.
This repository redistributes none of the data - only code that downloads it.

## Expected file

`EnronEmailReplyPairsWithContext.csv` (~11 MB), columns:

| Column | Meaning |
|---|---|
| `EmailSend` | the incoming email body |
| `EmailReply` | the reply a human actually sent |
| `SubjectSend` / `SubjectReply` | subjects of each |
| `From` / `To` | participants |
| `DateSend` / `DateReply` | timestamps |
| `Context` | additional thread context supplied by the dataset author |

Any CSV in this folder is picked up automatically (largest wins), and column
names are resolved against a candidate list in `src/hiver_email/data.py`, so a
slightly different dump still loads.

## Ethics note

These are real emails from real people, made public through litigation. They are
used here strictly as a research corpus for evaluating reply generation.
