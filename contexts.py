"""The privileged information c: what the teacher reads, and the student does not.

Each function takes one training example and the response that the student has just written for it, and returns
the whole message that the teacher reads: the prompt x followed by c. The teacher then scores that same response. The
student always reads only example["prompt"]. Pick one with --context <name>; in Part 3 you write your own, mine() at
the bottom (TODO 6). The four given here do not use the student's response.

An example is a dict with these fields (see hw3/data.py):

    "prompt"          x: the documentation of the API, the answer format and, at the end after "Question:",
                      the user's question, as one string
    "demo"            the expert's response, e.g. 'I need to ... \nAction: getStoreInformation\nAction Input: {...}'
    "answer"          the correct tool calls, a list of {"Action": tool name, "Action_Input": arguments as JSON text}
    "other_question"  another question about the same API ...
    "other_demo"      ... and the expert's response to it
    "api", "id"       the API's name and the example's id
"""


def demo(example: dict, response: str) -> str:
    """The expert's response to this question: the reasoning and the answer. This is SDFT's teacher prompt."""
    return (
        f"{example['prompt']}\n\n"
        f"This is an example for a response to the question:\n{example['demo']}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )


def answer(example: dict, response: str) -> str:
    """Only the correct tool calls, without the reasoning."""
    calls = "\n".join(f"Action: {a['Action']}\nAction Input: {a['Action_Input']}" for a in example["answer"])
    return (
        f"{example['prompt']}\n\n"
        f"These are the correct tool calls for the question:\n{calls}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )


def other(example: dict, response: str) -> str:
    """The expert's response to a different question about the same API: the format, but not this answer."""
    return (
        f"{example['prompt']}\n\n"
        f"This is an example for a response to a different question about the same tools.\n"
        f"Question: {example['other_question']}\n{example['other_demo']}\n\n"
        "Now answer the original question with a response of your own, including the thinking process.\n"
    )


def none(example: dict, response: str) -> str:
    """Nothing: the teacher reads what the student reads. The control."""
    return example["prompt"]


def mine(example: dict, response: str) -> str:
    """TODO 6 (Part 3). Your own privileged information.

    Return the teacher's whole message, like the functions above: start with example["prompt"], then add your c.
    You can compute c in code from any field of the example and from the student's response: for example a hint
    derived from example["answer"], a rule, part of the answer, or feedback on the student's attempt, as the
    environment would give it. hw3.evaluate has two helpers for feedback: correct(response, example["answer"]) says
    whether the attempt is right, and parse(response) returns the tools it called and the arguments it passed. Think
    about what the teacher should know that the student does not, and what you want the student to learn from it.

    Args:
        example: one training example (the fields are listed at the top of this file).
        response: the response that the student wrote for it, as text; the teacher will score this response.
    Returns:
        The teacher's message, as one string."""
    raise NotImplementedError("TODO 6")


CONTEXTS = {"demo": demo, "answer": answer, "other": other, "none": none, "mine": mine}
