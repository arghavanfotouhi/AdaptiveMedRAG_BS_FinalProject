import numpy as np
import json
import re


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_options(options):
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}: {v}" for k, v in options.items())
    if isinstance(options, list):
        return "\n".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))
    return str(options)


def _option_keys(options):
    if not options:
        return []
    if isinstance(options, dict):
        return list(options.keys())
    if isinstance(options, list):
        return [chr(65 + i) for i in range(len(options))]
    return []


def _format_snippets(snippets):
    """Number snippets 1..N in retrieval order. Numbering must match what
    the prompt shows, since supporting_documents refers to these numbers."""
    text = ""
    for i, snippet in enumerate(snippets, start=1):
        if isinstance(snippet, dict):
            title = snippet.get("title", "")
            content = snippet.get("content", snippet)
        else:
            title = ""
            content = str(snippet)
        header = f"[Document {i}]"
        if title:
            header += f" Title: {title}"
        text += f"{header}\n{content}\n\n"
    return text.strip()


def _extract_json(raw_response):
    if raw_response is None:
        raise ValueError("Empty response from LLM.")
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(match.group(0))



JUDGE_SYSTEM_PROMPT = r"""
You are a strict evidence-sufficiency auditor for adaptive medical retrieval.

Your task is NOT to answer the medical question directly.
Your task is to determine whether the retrieved evidence contains enough
specific, relevant information to safely distinguish the answer choices.

You must separate three things:

1. INFORMATION REQUIREMENTS
   What factual/clinical information must be established from the literature?

2. EVIDENCE ASSESSMENT
   Do the retrieved documents actually establish those requirements?

3. OPTION DISCRIMINATION
   After assessing the evidence, does the evidence support one answer option?

============================================================
REQUIREMENT GENERATION — STRICT RULES
============================================================

A requirement is an OPTION-INDEPENDENT factual or clinical information need
that must be established by retrieved evidence to determine the answer.

A requirement describes WHAT MUST BE KNOWN.
It must NOT describe WHAT THE ANSWER IS.

CRITICAL: Requirements must NOT leak an answer choice.

A requirement MUST NOT:
- mention an option letter (A, B, C, etc.);
- quote an answer option;
- paraphrase an answer option;
- use a synonym that uniquely identifies an answer option;
- name a candidate diagnosis, treatment, cause, mechanism, organism,
  medication, exposure, or other answer candidate when that candidate comes
  specifically from the answer choices;
- assert or imply that one candidate is more likely than another;
- contain a conclusion that would effectively answer the question;
- be written as "determine whether [specific option]...";
- be written as "establish whether [specific option] caused/explains...";
- contain the expected answer in disguised form.

For example, if an answer choice is "scorpion envenomation":

BAD:
"Determine whether scorpion envenomation caused the pancreatitis."

BAD:
"Determine whether the patient's arthropod venom exposure caused the
pancreatitis."

BAD:
"Establish whether the recent sting is responsible for the pancreatitis."

These all leak the candidate answer.

BETTER:
"Establish whether a documented recent exposure provides a causal association
with the patient's clinical condition."

The evidence must subsequently determine WHICH exposure, if any, is supported.

============================================================
REQUIREMENT CONTENT
============================================================

Requirements should be:

- factual rather than conclusory;
- concise;
- independently assessable;
- grounded in the clinical question;
- specific enough that a retrieved passage can clearly support or fail
  to support them;
- limited to information that materially affects answer determination.

Do NOT generate requirements merely for completeness.

Generate the MINIMUM number of independent requirements needed to determine
the answer. Usually 1-3 requirements are sufficient. More requirements are
allowed only when genuinely necessary.

Do not turn one clinical issue into multiple overlapping requirements.

Do not write a requirement as a mini-answer, explanation, diagnosis, or
treatment recommendation.

============================================================
OPTION-INDEPENDENCE TEST
============================================================

Before returning each requirement, apply this test:

"If I replaced the current answer choices with a different set of plausible
answer choices, would this requirement still make sense for the same clinical
question?"

If NO:
    rewrite the requirement to remove option-specific information.

A second test:

"Does this requirement tell the retrieval system what evidence is needed,
or does it tell the system what answer I expect?"

If it tells the expected answer, rewrite it.

============================================================
CRITICALITY
============================================================

critical=true ONLY when failure to establish that factual requirement would
leave the answer materially underdetermined.

Do NOT make a requirement critical merely because:
- it is medically important;
- it is interesting;
- it is related to an answer option;
- evidence supporting it would favor one option;
- it helps explain the case.

The question is:

"If this requirement remained unresolved, would the available evidence still
be sufficient to safely distinguish the answer?"

If YES -> critical=false.
If NO  -> critical=true.

Most questions should have only 1-2 critical requirements.

CRITICALITY COUNTERFACTUAL:
Temporarily assume that this requirement can never be resolved.
If the remaining evidence could still safely establish which answer is correct,
mark critical=false.
If the answer would remain materially underdetermined, mark critical=true.

This is a test of INFORMATION NECESSITY, not OPTION PREFERENCE.
Do not mark a requirement critical merely because it strongly supports or rules
out one particular answer choice.

============================================================
EVIDENCE EVALUATION
============================================================

A document supports a requirement only if it explicitly and specifically
addresses the factual information described by that requirement.

Do not count:
- broad topical similarity;
- generic medical knowledge;
- indirect associations;
- evidence about a different disease, exposure, population, or mechanism;
- information that merely happens to make an answer option seem plausible.

Do not use outside medical knowledge.

Every RESOLVED or PARTIALLY_RESOLVED requirement must have valid supporting
document numbers.

If you are uncertain whether a passage establishes the requirement, mark it
UNRESOLVED.

============================================================
OPTION DISCRIMINATION
============================================================

Only AFTER independently evaluating the requirements and evidence, assess the
answer choices.

The supported_option must be:
- one option letter, ONLY when the retrieved evidence explicitly and
  specifically supports that option AND distinguishes it from competing
  options;
- null when the evidence does not clearly distinguish one option.

Evidence that merely supports a fact compatible with an option is NOT enough
if the same evidence could also be compatible with another option.

Do not choose an option merely because:
- it matches a requirement you wrote;
- it is medically plausible;
- it is the most common diagnosis;
- it is consistent with outside knowledge;
- the requirement wording appears to favor it.

The requirement generation process must never determine the supported option.

The evidence must determine the supported option.

============================================================
INCREMENTAL ROUNDS
============================================================

On later retrieval rounds:

- preserve existing requirement IDs;
- normally preserve existing requirement wording;
- re-evaluate every requirement against the FULL evidence set;
- do not create new requirements merely because more documents were added;
- add a requirement only if a genuinely necessary information need was
  completely missed;
- a requirement may be rewritten if and ONLY if necessary to remove
  option-specific leakage or make the information need precise;
- do not make a previously non-critical requirement critical merely because
  new evidence makes an option appear more attractive.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Each requirement must contain:

- id
- text
- critical
- status
- supporting_documents
- justification
- option_leakage (true if the requirement contains option-specific or
  answer-bearing wording; false otherwise)

Use only these statuses:

RESOLVED
PARTIALLY_RESOLVED
UNRESOLVED

The supported_option must be an option letter or null.

Do not include a top-level decision field.
The calling Python code determines STOP/CONTINUE.

Return this schema:

{
  "requirements": [
    {
      "id": "R1",
      "text": "...",
      "critical": false,
      "status": "UNRESOLVED",
      "supporting_documents": [],
      "justification": "...",
      "option_leakage": false
    }
  ],
  "supported_option": null,
  "option_reasoning": "...",
  "reason": "..."
}
"""

def _build_user_prompt(question, options_text, requirements, snippets_text,
                        is_incremental, new_count, total_count):
    options_block = f"\nANSWER OPTIONS:\n{options_text}\n" if options_text else ""

    if requirements is None:
        requirements_block = """No requirements have been identified yet — this is the first check.

1. List the MINIMUM number of concrete, option-independent information requirements
needed to determine the answer. Usually 1-3 requirements are sufficient. Each
requirement must describe WHAT MUST BE KNOWN, not what answer is expected.
2. Apply the OPTION-INDEPENDENCE TEST and CRITICALITY COUNTERFACTUAL from the
system instructions before returning each requirement.
3. Evaluate the CURRENT evidence set against each requirement."""
    else:
        req_lines = "\n".join(
            f"- id={r['id']} | critical={r['critical']} | text={r['text']}"
            for r in requirements
        )

        requirements_block = f"""
These requirements were identified in a previous round.

EXISTING REQUIREMENTS:
{req_lines}

Re-evaluate these same information requirements against the FULL evidence set.

IMPORTANT:

The requirements represent INFORMATION NEEDS, not candidate answers.

Do not reinterpret a requirement toward whichever answer option currently
appears most plausible.

Normally preserve the same IDs and wording.

You may rewrite a requirement ONLY if:
1. the original wording contains option-specific information;
2. the original wording contains a conclusion rather than an information need;
3. the original wording is materially ambiguous; or
4. the original requirement is redundant with another requirement.

If rewriting is necessary, preserve the underlying information need while
removing answer leakage.

Do not create a new requirement simply because a newly retrieved document
supports an existing requirement.

Add a new requirement only when a genuinely necessary information need was
completely absent from the previous requirement set.

Re-evaluate whether each requirement is actually supported by the documents.
Pay particular attention to whether newly retrieved documents establish the
specific factual gap rather than merely discussing the same general topic.
"""

    mode_note = (
        f"This round adds {new_count} newly retrieved document(s). The evidence set "
        f"below contains all {total_count} documents retrieved so far, numbered in "
        f"retrieval order (new documents are the last {new_count})."
        if is_incremental else
        f"This is the first evidence check, over the first {total_count} retrieved "
        f"document(s), numbered in retrieval order."
    )

    return f"""QUESTION:
{question}
{options_block}
{mode_note}

{requirements_block}

EVIDENCE SET ({total_count} passages):
{snippets_text}

TASK:
1. Produce/update the requirement list. Each item needs: id, text, critical
(true/false), status ("RESOLVED" / "PARTIALLY_RESOLVED" / "UNRESOLVED"),
supporting_documents (document numbers that actually ground a RESOLVED or
PARTIALLY_RESOLVED status — empty list if UNRESOLVED), a one-sentence
justification, and option_leakage (true/false).
2. Before returning each requirement, verify that it is option-independent and
does not contain a candidate answer, disguised candidate, or conclusion. If it
does, rewrite it before returning it.
3. Mark critical=true ONLY under the criticality counterfactual: if that factual
requirement stayed unresolved, the answer would remain materially underdetermined.
Do not mark it critical merely because it favors an option.
4. ONLY AFTER independently evaluating the requirements against the evidence,
set supported_option. Set it to a single option letter ONLY when the documents
distinguish that option from competing choices; otherwise set null.
5. Do NOT include a top-level "decision" field — the calling system computes it.

Return ONLY this JSON schema:
{{
  "requirements": [
    {{
      "id": "R1",
      "text": "...",
      "critical": true,
      "status": "RESOLVED",
      "supporting_documents": [1, 3],
      "justification": "..."
    }}
  ],
  "supported_option": "B",
  "option_reasoning": "...",
  "reason": "one or two sentence overall summary"
}}"""



# ---------------------------------------------------------------------------
# Core judge (single prompt/schema for round 1 and every incremental round)
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED"}


def _requirement_snapshot(normalized):
    """A hashable snapshot of everything that matters for detecting whether
    a round actually changed anything: per-requirement status + which docs
    ground it. Used for the stagnation/plateau checks."""
    return tuple(sorted(
        (r["id"], r["status"], tuple(sorted(r["supporting_documents"])))
        for r in normalized
    ))


def judge_evidence_sufficiency(
    medrag,
    question,
    snippets,
    options=None,
    previous_requirements=None,
    new_snippet_count=None,
    verbose=False,
):
    """Evaluate whether `snippets` (all documents retrieved so far) are
    sufficient to answer `question`.

    Pass previous_requirements=None for the very first check. For every
    later round, pass the `requirements` list returned by the previous
    call so the same requirement ids get re-evaluated instead of
    regenerated.

    Returns a dict with:
        decision            "STOP" or "CONTINUE" (computed in Python, not
                             taken from the LLM)
        stop_reason          "clean" | "plateau" | None — WHY it stopped,
                             for logging/debugging
        requirements         updated, grounded requirement list
        missing_requirements requirements not yet RESOLVED (for the next
                              round's `previous_requirements`)
        supported_option      option letter or None
        snapshot              opaque tuple for stagnation detection by the
                              caller (see adaptive_retrieve)
        reason
        raw_response
    """
    options_text = _format_options(options)
    total_count = len(snippets)
    is_incremental = previous_requirements is not None
    new_count = new_snippet_count if new_snippet_count is not None else total_count
    snippets_text = _format_snippets(snippets)

    user_prompt = _build_user_prompt(
        question=question,
        options_text=options_text,
        requirements=previous_requirements,
        snippets_text=snippets_text,
        is_incremental=is_incremental,
        new_count=new_count,
        total_count=total_count,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = None
    parse_error = None
    result = None

    # One retry on parse failure — LLMs occasionally wrap JSON in prose.
    for attempt in range(2):
        try:
            raw_response = medrag.generate(messages)
            result = _extract_json(raw_response)
            break
        except Exception as e:  # noqa: BLE001
            parse_error = e
            if attempt == 0:
                messages = messages + [
                    {"role": "user", "content": "That was not valid JSON. Return ONLY the JSON object, nothing else."}
                ]
                continue

    if result is None:
        # Fail safe: never STOP on a parse failure. Keep prior requirements
        # so the schedule can still try again with more evidence.
        fallback_requirements = previous_requirements or []
        return {
            "decision": "CONTINUE",
            "stop_reason": None,
            "requirements": fallback_requirements,
            "missing_requirements": fallback_requirements,
            "supported_option": None,
            "fully_unresolved_critical": [
                            r["id"] for r in fallback_requirements
                            if r.get("critical") and r.get("status") == "UNRESOLVED"
                        ],
            "partial_critical": [
                r["id"] for r in fallback_requirements
                if r.get("critical") and r.get("status") == "PARTIALLY_RESOLVED"
            ],
            "snapshot": _requirement_snapshot([
                {"id": r["id"], "status": r["status"], "supporting_documents": r["supporting_documents"]}
                for r in fallback_requirements
            ]) if fallback_requirements else tuple(),
            "reason": f"Judge response could not be parsed: {parse_error}",
            "raw_response": raw_response,
        }

    # --- Normalize + ground requirements --------------------------------
    normalized = []
    for i, r in enumerate(result.get("requirements", []) or []):
        if not isinstance(r, dict):
            continue
        status = str(r.get("status", "UNRESOLVED")).upper().strip()
        if status not in _VALID_STATUSES:
            status = "UNRESOLVED"

        supporting = r.get("supporting_documents", []) or []
        # Keep only integer doc numbers that are actually in range —
        # this is what forces the hypertension/UTI-style false positive
        # back to UNRESOLVED even if the model tried to claim it as RESOLVED.
        valid_supporting = [
            d for d in supporting
            if isinstance(d, int) and 1 <= d <= total_count
        ]
        if status in ("RESOLVED", "PARTIALLY_RESOLVED") and not valid_supporting:
            status = "UNRESOLVED"

        normalized.append({
            "id": r.get("id") or f"R{i + 1}",
            "text": r.get("text", "").strip(),
            "critical": bool(r.get("critical", True)),
            "status": status,
            "supporting_documents": valid_supporting,
            "justification": r.get("justification", ""),
            "option_leakage": bool(r.get("option_leakage", False)),
        })

    supported_option = result.get("supported_option")
    if isinstance(supported_option, str):
        supported_option = supported_option.strip().upper() or None
    else:
        supported_option = None

    option_ok = True
    if options:
        valid_keys = _option_keys(options)
        option_ok = supported_option is not None and supported_option in valid_keys

    # --- Deterministic decision (never trust a self-declared "decision") -
    # Split critical requirements by how bad their status is. UNRESOLVED
    # (no grounding at all) is a hard blocker. PARTIALLY_RESOLVED (some
    # grounding, not complete) is a *soft* blocker — see plateau path below.
    critical = [r for r in normalized if r["critical"]]
    fully_unresolved_critical = [r for r in critical if r["status"] == "UNRESOLVED"]
    partial_critical = [r for r in critical if r["status"] == "PARTIALLY_RESOLVED"]

    clean_stop = (not fully_unresolved_critical) and (not partial_critical) and option_ok

    decision = "STOP" if clean_stop else "CONTINUE"
    stop_reason = "clean" if clean_stop else None

    missing_requirements = [r for r in normalized if r["status"] != "RESOLVED"]

    structured_output = {
        "decision": decision,
        "stop_reason": stop_reason,
        "requirements": normalized,
        "missing_requirements": missing_requirements,
        "supported_option": supported_option,
        "option_ok": option_ok,
        "fully_unresolved_critical": [r["id"] for r in fully_unresolved_critical],
        "partial_critical": [r["id"] for r in partial_critical],
        "snapshot": _requirement_snapshot(normalized),
        "option_reasoning": result.get("option_reasoning", ""),
        "reason": result.get("reason", "No reason provided."),
        "raw_response": raw_response,
    }

    if verbose:
        print("\n" + "=" * 60)
        print(f"[JUDGE] decision={decision}  supported_option={supported_option}")
        for r in normalized:
            print(f"  [{r['status']:>18}] {r['id']}: {r['text']}  (critical={r['critical']}, docs={r['supporting_documents']})")
        print(f"  reason: {structured_output['reason']}")
        print("=" * 60 + "\n")

    return structured_output


# ---------------------------------------------------------------------------
# Adaptive retrieval driver
# ---------------------------------------------------------------------------

def adaptive_retrieve(
    medrag,
    question,
    options=None,
    rrf_k=100,
    k_schedule=(4, 8, 16, 32),
    verbose=True,
    retrieved=None
):
    """Adaptive retrieval with one initial top-max_k retrieval, evaluated
    incrementally at each k in k_schedule using a single tracked
    requirement list (see judge_evidence_sufficiency).

    Two ways the loop can end before max_k:
      1. Clean STOP: every critical requirement is fully RESOLVED and a
         grounded option is identified.
      2. Plateau STOP: no critical requirement is fully UNRESOLVED (each
         has at least partial grounding), the supported_option has been
         the same grounded value for two consecutive rounds, AND the
         evidence didn't change at all between those two rounds (i.e.
         retrieving more documents stopped producing new grounding).
    Independently of both, a round that changes NOTHING at all relative
    to the previous round (identical requirement snapshot) ends the
    schedule early, since more retrieval is provably not adding evidence.

    Returns (selected_snippets, selected_scores, selected_k, history).
    """
    if not k_schedule:
        raise ValueError("k_schedule cannot be empty.")

    cleaned_schedule = sorted(set(k_schedule))
    if cleaned_schedule[0] <= 0:
        raise ValueError("All k values in k_schedule must be positive.")

    max_k = cleaned_schedule[-1]

    if verbose:
        print("\n" + "=" * 80)
        print("ADAPTIVE RETRIEVAL")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"Retrieval schedule: {cleaned_schedule}")


    if retrieved is None:
        all_snippets, all_scores, all_ids, all_embeddings = (
            medrag.retrieval_system.retrieve_with_embeddings(
                question, k=max_k, rrf_k=rrf_k
            )
        )
    else:
        all_snippets, all_scores, all_ids, all_embeddings = retrieved


    actual_max_k = min(
    max_k,
    len(all_snippets),
    len(all_scores),
    len(all_ids),
    len(all_embeddings),
    )

    all_snippets = all_snippets[:actual_max_k]
    all_scores = all_scores[:actual_max_k]
    all_ids = all_ids[:actual_max_k]
    all_embeddings = all_embeddings[:actual_max_k]

    if actual_max_k == 0:
        if verbose:
            print(">>> NO DOCUMENTS RETRIEVED")
        return [], [], 0, []

    if len(all_embeddings) != actual_max_k:
        raise RuntimeError(
            "Embedding count does not match retrieved document count."
        )

    if len(all_ids) != actual_max_k:
        raise RuntimeError(
            "Document ID count does not match retrieved document count."
        )

    valid_schedule = [k for k in cleaned_schedule if k <= actual_max_k]
    if not valid_schedule or valid_schedule[-1] < actual_max_k:
        valid_schedule.append(actual_max_k)
    valid_schedule = sorted(set(valid_schedule))

    all_novelty = compute_document_novelty(all_embeddings)

    history = []
    requirements = None
    previous_k = 0
    selected_snippets, selected_scores = [], []

    previous_snapshot = None
    option_history = []  # supported_option seen at each round, in order
    novelty_history = []  # mean_new_novelty seen at each round, in order



    for current_k in valid_schedule:
        if current_k <= previous_k:
            continue


        current_snippets = all_snippets[:current_k]
        current_scores = all_scores[:current_k]
        current_ids = all_ids[:current_k]
        current_embeddings = all_embeddings[:current_k]
        new_count = current_k - previous_k

        if new_count > 0:
            new_novelty = all_novelty[previous_k:current_k]

            valid_new_novelty = new_novelty[
                ~np.isnan(new_novelty)
            ]

            if len(valid_new_novelty) > 0:
                mean_new_novelty = float(
                    np.mean(valid_new_novelty)
                )
                min_new_novelty = float(
                    np.min(valid_new_novelty)
                )
                max_new_novelty = float(
                    np.max(valid_new_novelty)
                )
            else:
                mean_new_novelty = None
                min_new_novelty = None
                max_new_novelty = None
        else:
            mean_new_novelty = None
            min_new_novelty = None
            max_new_novelty = None

        # Track mean novelty across rounds so the stagnation guard below
        # can look at the TREND (is novelty declining?) rather than a
        # fixed absolute cutoff. See docstring note on why an absolute
        # novelty threshold is not statistically reliable.
        novelty_history.append(mean_new_novelty)

        if verbose:
            print("\n" + "-" * 80)
            print(f"k={current_k}  (+{new_count} new, {current_k} total)")
            print("-" * 80)

        decision = judge_evidence_sufficiency(
            medrag=medrag,
            question=question,
            snippets=current_snippets,
            options=options,
            previous_requirements=requirements,
            new_snippet_count=new_count,
            verbose=verbose,
        )

        selected_snippets = current_snippets
        selected_scores = current_scores
        requirements = decision["requirements"]
        option_history.append(decision["supported_option"])

        # --- Plateau STOP: no fully-unresolved critical requirement, a
        # stable grounded option across the last two rounds, and the
        # evidence itself didn't move between those two rounds.
        if decision["decision"] != "STOP" and not decision["fully_unresolved_critical"]:
            stable_option = (
                len(option_history) >= 2
                and option_history[-1] is not None
                and option_history[-1] == option_history[-2]
            )
            evidence_plateaued = (
                previous_snapshot is not None
                and decision["snapshot"] == previous_snapshot
            )
            if stable_option and evidence_plateaued:
                decision["decision"] = "STOP"
                decision["stop_reason"] = "plateau"
                decision["reason"] = (
                    decision["reason"]
                    + " [auto-stop: remaining critical requirements stayed only "
                      "partially resolved across two rounds with no new grounding, "
                      f"and the supported option ({option_history[-1]}) was stable]"
                )
                if verbose:
                    print(f">>> PLATEAU STOP at k={current_k}: evidence and supported "
                          f"option stopped changing between rounds.")

        # --- Stagnation guard: this round changed nothing at all versus
        # the previous round, even though new documents were added. More
        # retrieval isn't buying new evidence, so stop regardless of the
        # STOP/CONTINUE verdict.
        #
        # IMPORTANT: this must NOT fire when nothing has been found yet.
        # "Zero evidence at k=4, still zero evidence at k=8" is not a
        # plateau -- it just means we haven't retrieved enough yet, and
        # with a max_k of 32 there's plenty of cheap runway left. A
        # plateau only means something once there IS grounding on a
        # critical requirement that has stopped moving. So require at
        # least one critical requirement to already have real supporting
        # documents before this guard is allowed to trigger.
        has_critical_grounding = any(
            r["supporting_documents"] for r in decision["requirements"] if r["critical"]
        )

        # Path 1 (original): the judge's structural snapshot (status +
        # supporting_documents per requirement) is byte-for-byte identical
        # to the previous round -- the strongest possible signal that
        # nothing changed.
        snapshot_stagnated = (
            previous_snapshot is not None
            and decision["snapshot"] == previous_snapshot
        )

        # Path 2 (new, embedding-level): the judge's snapshot may have
        # shifted on some non-critical detail even though retrieval itself
        # has stopped bringing new information. We detect this from the
        # TREND of marginal novelty across rounds (this round's mean new
        # novelty did not meaningfully grow versus the previous round),
        # never from a fixed absolute cutoff -- an absolute threshold was
        # tested against real judge transitions and did not separate
        # "requirement changed" vs "requirement unchanged" rounds (the
        # two distributions overlapped almost entirely), so only the
        # per-question relative trend is used here.
        novelty_saturated = (
            len(novelty_history) >= 2
            and novelty_history[-1] is not None
            and novelty_history[-2] is not None
            and novelty_history[-1] <= novelty_history[-2] * 1.05
        )

        stagnated = (
            decision["decision"] != "STOP"
            and has_critical_grounding
            and (
                snapshot_stagnated
                or (novelty_saturated and not decision["fully_unresolved_critical"])
            )
        )

        stagnation_reason = (
            "snapshot_identical" if snapshot_stagnated
            else "novelty_saturated" if (novelty_saturated and not decision["fully_unresolved_critical"])
            else None
        )


        history.append({
            "k": current_k,
            "new_documents": new_count,

            "document_ids": current_ids,

            "mean_new_novelty": mean_new_novelty,
            "min_new_novelty": min_new_novelty,
            "max_new_novelty": max_new_novelty,

            "judge_decision": decision["decision"],
            "supported_option": decision["supported_option"],

            "requirements": decision["requirements"],
            "missing_requirements": decision["missing_requirements"],

            "stop_reason": decision.get("stop_reason"),

            "reason": decision.get("reason", ""),

            "stagnated": stagnated,
            "stagnation_reason": stagnation_reason,
        })

        if decision["decision"] == "STOP":
            if verbose:
                print(f">>> STOP at k={current_k} ({decision['stop_reason']})")
            return selected_snippets, selected_scores, len(selected_snippets), history

        if stagnated:
            if verbose:
                print(f">>> STAGNATION STOP at k={current_k} ({stagnation_reason}): "
                      f"more retrieval unlikely to help.")
            return selected_snippets, selected_scores, len(selected_snippets), history

        previous_snapshot = decision["snapshot"]
        previous_k = current_k
        if current_k >= actual_max_k:
            break

    if verbose:
        print("\n" + "=" * 80)
        print(">>> MAXIMUM K REACHED without a grounded STOP")
        print(f"Returning {len(selected_snippets)} documents.")
        print("=" * 80)

    return selected_snippets, selected_scores, len(selected_snippets), history

def compute_document_novelty(embeddings):
    """
    Compute novelty of each retrieved document relative to
    documents appearing before it in the ranked retrieval list.

    novelty[i] = 1 - max cosine_similarity(
        embedding[i],
        embeddings[0:i]
    )
    """

    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=np.float32)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Normalize for cosine similarity
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / np.clip(
        norms,
        1e-12,
        None
    )

    similarity_matrix = normalized @ normalized.T

    novelty = np.ones(
        len(embeddings),
        dtype=np.float32
    )

    # First document has no previous document
    novelty[0] = np.nan

    for i in range(1, len(embeddings)):

        max_previous_similarity = np.max(
            similarity_matrix[i, :i]
        )

        novelty[i] = (
            1.0 - max_previous_similarity
        )

    return novelty


