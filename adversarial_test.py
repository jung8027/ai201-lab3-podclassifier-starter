"""
Adversarial Description Test
=============================
Tests descriptions intentionally crafted to sit on taxonomy boundaries.
Each has a "surface label" (what it looks like) and a "true label" (what it is).

Run with:
    python adversarial_test.py
"""
from classifier import load_labeled_examples, classify_episode

ADVERSARIAL = [
    {
        "title": "The Week Everything Changed",
        "description": (
            "Five years ago I sat in a hospital waiting room with nothing but a notebook. "
            "What followed was the strangest week of my life. I take you through it hour by "
            "hour — what I was thinking, who I met in that waiting room, what I overheard, "
            "and how the arc of that week still shapes how I think about time. This is a "
            "story with a beginning, a crisis, and an ending. I am the only voice."
        ),
        "true_label": "solo",
        "surface_label": "narrative",
        "why_hard": "Clear story arc, multiple 'characters,' past-tense narration — reads "
                    "like a narrative but the host is the sole source, speaking from memory.",
    },
    {
        "title": "Webb vs. Chen: The Nuclear Debate",
        "description": (
            "Dr. Marcus Webb and I don't agree on much about nuclear energy's future, and "
            "today we prove it. For ninety minutes he and I argue — his bullish case, my "
            "skepticism, neither of us holding back. We treat each other as equals. "
            "Both of us change our minds on at least one point. One guest, one host, "
            "zero moderation."
        ),
        "true_label": "interview",
        "surface_label": "panel",
        "why_hard": "Language of debate and equality ('treat each other as equals') mimics "
                    "panel format, but it is still one host + one guest — the structural "
                    "definition of interview.",
    },
    {
        "title": "What I Found in the Archive",
        "description": (
            "I spent six months in the municipal archive. This episode is my account: the "
            "boxes I opened, the handwriting I learned to read, the officials who helped me "
            "and the ones who didn't. Every voice you hear gave me permission to be there. "
            "I assembled this story from documents, recordings, and interviews — "
            "but I narrate the whole thing myself in the first person."
        ),
        "true_label": "narrative",
        "surface_label": "solo",
        "why_hard": "Heavy first-person framing sounds like a solo essay, but the episode "
                    "draws on external archival material, recordings, and interviews — the "
                    "assembly is what makes it narrative.",
    },
    {
        "title": "Four Researchers Respond to the Same Paper",
        "description": (
            "We sent a landmark 2019 paper on sleep and cognition to four researchers and "
            "asked each to record a five-minute response. None of them heard the others "
            "before recording. The host stitches together their replies with brief "
            "connective narration. You hear four voices, but no one is talking to anyone else."
        ),
        "true_label": "narrative",
        "surface_label": "panel",
        "why_hard": "Four expert voices, one topic — looks like a panel, but there is no "
                    "discussion. The voices were assembled by the host into a narrative "
                    "structure; they never interacted.",
    },
]


def main():
    labeled = load_labeled_examples()

    print("\nAdversarial Description Test")
    print("=" * 60)

    correct = 0
    for ex in ADVERSARIAL:
        result = classify_episode(ex["description"], labeled)
        predicted = result["label"]
        confidence = result.get("confidence")
        match = "✓" if predicted == ex["true_label"] else "✗"
        conf_str = f"  conf={confidence}/10" if confidence is not None else ""

        print(f"\n{match} '{ex['title']}'")
        print(f"   true={ex['true_label']}  surface={ex['surface_label']}  "
              f"predicted={predicted}{conf_str}")
        print(f"   Why hard: {ex['why_hard']}")
        if predicted != ex["true_label"]:
            print(f"   Reasoning: {result['reasoning']}")
        if predicted == ex["true_label"]:
            correct += 1

    print(f"\n{correct}/{len(ADVERSARIAL)} correct on adversarial set")
    print("\nNote: errors here are informative — they reveal which taxonomy")
    print("boundaries are underspecified in your few-shot examples.")


if __name__ == "__main__":
    main()
