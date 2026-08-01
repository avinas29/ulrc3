"""Lexical resources: stopwords, discourse markers, deontic cues, boilerplate.

Everything here is data, deliberately kept out of the algorithms.  Adding a
language or a domain means editing this file, not the passes.
"""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "also", "just", "really", "actually", "basically", "simply", "quite", "rather", "somewhat", "perhaps", "maybe", "things", "thing", "stuff", "lot", "lots", "kind", "sort", "well", "okay", "ok", "yes", "no", "oh", "ah", "um", "uh", "hmm"]
)

#: Words that almost never carry propositional content but inflate prose.
FILLER: frozenset[str] = frozenset(
    ["basically", "actually", "literally", "simply", "just", "really", "very", "quite", "rather", "somewhat", "fairly", "totally", "definitely", "certainly", "obviously", "clearly", "essentially", "fundamentally", "arguably", "generally", "typically", "usually", "often", "sometimes", "perhaps", "maybe", "possibly", "probably", "furthermore", "moreover", "additionally", "however", "nevertheless", "nonetheless", "therefore", "thus", "hence", "accordingly", "consequently", "indeed", "notably", "importantly", "interestingly"]
)

#: Discourse connectives -- cheap to drop, but they mark reasoning structure, so
#: we keep them when the sentence participates in a reasoning chain.
DISCOURSE: frozenset[str] = frozenset(
    ["because", "therefore", "thus", "hence", "so", "since", "although", "though", "however", "but", "whereas", "while", "if", "unless", "until", "otherwise", "then", "consequently", "accordingly", "moreover", "furthermore", "first", "second", "third", "finally", "next", "lastly", "instead", "rather", "conversely", "similarly"]
)

#: Deontic cues, tiered.  Treating every "before"/"ensure" in technical prose as
#: a hard constraint locks 60%+ of a document and destroys the compression
#: ratio, so the tiers carry different consequences:
#:
#:   STRONG -> CONSTRAINT obligation, unit LOCKED (deontic force is explicit)
#:   QUANT  -> CONSTRAINT obligation *only when a number is in the same clause*
#:             (a bound without a bound is not a bound)
#:   SOFT   -> salience boost only, no obligation
DEONTIC_STRONG = re.compile(
    r"\b(must(?:\s+not)?|shall(?:\s+not)?|may\s+not|cannot|can\s?not|never|always|"
    r"required|mandatory|forbidden|prohibited|not\s+allowed|disallow(?:ed)?|"
    r"should\s+not|do\s+not|don't|obligat\w+|entitled|liable|"
    r"shall\s+be\s+deemed|notwithstanding)\b",
    re.IGNORECASE,
)
DEONTIC_QUANT = re.compile(
    r"\b(only\s+if|only\s+when|at\s+most|at\s+least|no\s+more\s+than|no\s+fewer\s+than|"
    r"exactly|maximum|minimum|max|min|limit(?:ed)?\s+to|up\s+to|deadline|due\s+by|"
    r"expires?|timeout|threshold|quota|cap(?:ped)?\s+at|per\s+(?:second|minute|hour|day))\b",
    re.IGNORECASE,
)
DEONTIC_SOFT = re.compile(
    r"\b(require[sd]?|avoid|ensure|guarantee|subject\s+to|provided\s+that|"
    r"before|after|unless|otherwise|except)\b",
    re.IGNORECASE,
)
#: Backwards-compatible union (used by the conversation record classifier).
DEONTIC = re.compile(
    "|".join(p.pattern for p in (DEONTIC_STRONG, DEONTIC_QUANT)), re.IGNORECASE
)
_NEAR_NUMBER = re.compile(r"\d")

NEGATION = re.compile(
    r"\b(not|no|never|none|neither|nor|without|except|excluding|unless|"
    r"cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"shouldn't|wouldn't|couldn't|mustn't)\b",
    re.IGNORECASE,
)

IMPERATIVE_HEAD = re.compile(
    r"^\s*(?:please\s+)?(?:do\s+not\s+|don't\s+|never\s+|always\s+)?"
    r"(write|create|generate|produce|return|output|answer|respond|explain|summar\w+|"
    r"list|extract|classify|translate|convert|implement|fix|refactor|analyz\w+|analys\w+|"
    r"compare|evaluate|review|check|verify|ensure|use|avoid|include|exclude|format|"
    r"follow|apply|consider|ignore|assume|act\s+as|you\s+are|your\s+task|your\s+role)\b",
    re.IGNORECASE,
)

#: Security cues.  Deliberately *narrow*: generic words like "token", "role",
#: "scope", "policy" and "auth" occur constantly in LLM-adjacent text ("with
#: token cost c(u,l)") and matching them turned ordinary prose into
#: unsatisfiable security obligations.  Role prompts and system instructions are
#: protected by segment typing, which is exact, not by keyword matching.
SECURITY = re.compile(
    r"\b(api[_\- ]?key|secret[s]?\b|password|passphrase|credential|private[_\- ]?key|"
    r"bearer\s+token|access[_\- ]?token|refresh[_\- ]?token|authorization\s+header|"
    r"oauth|jwt|ssn|social\s+security\s+number|pii|phi|gdpr|hipaa|pci[_\- ]?dss|"
    r"encrypt\w*|decrypt\w*|sanitiz\w+|sql\s+injection|injection\s+attack|xss|csrf|"
    r"rbac|least\s+privilege|jailbreak|prompt\s+injection|do\s+not\s+reveal|"
    r"never\s+(?:share|disclose|reveal|log)|confidential|classified|redact\w*)\b",
    re.IGNORECASE,
)

#: Conversational noise removed by the conversation pipeline.
SMALL_TALK = re.compile(
    r"^\s*(?:"
    r"(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|greetings)\b.{0,40}|"
    r"(thanks|thank\s+you|thx|ty|cheers|appreciate\s+it)\b.{0,40}|"
    r"(you're\s+welcome|no\s+problem|np|sure\s+thing|of\s+course)\b.{0,30}|"
    r"(sorry|apologies|my\s+bad)\b.{0,40}|"
    r"(ok|okay|k|got\s+it|understood|makes\s+sense|sounds\s+good|great|nice|cool|awesome|perfect)"
    r"[\s!.]*|"
    r"(bye|goodbye|see\s+you|talk\s+later|have\s+a\s+(good|great|nice)\s+\w+)\b.{0,30}"
    r")\s*$",
    re.IGNORECASE,
)

#: LLM boilerplate openers -- pure token tax in conversation history.
ASSISTANT_BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"(certainly|sure|absolutely|of\s+course|great\s+question|happy\s+to\s+help|"
    r"i'd\s+be\s+happy\s+to|i\s+can\s+help|let\s+me\s+help|here'?s?\s+(is|are)?\s*"
    r"(a|an|the)?\s*\w*\s*(summary|overview|breakdown|explanation)?)"
    r")\b[^.!?\n]{0,80}[.!?:]?\s*",
    re.IGNORECASE,
)

#: Document boilerplate (licences, confidentiality footers, nav chrome).
BOILERPLATE_LINE = re.compile(
    r"^\s*(?:"
    r"copyright\s+\(c\)|©|all\s+rights\s+reserved|licensed\s+under|"
    r"spdx-license-identifier|this\s+file\s+is\s+part\s+of|"
    r"confidential(?:ity)?\s+notice|do\s+not\s+distribute|"
    r"page\s+\d+\s+of\s+\d+|table\s+of\s+contents|"
    r"unsubscribe|view\s+in\s+browser|follow\s+us\s+on|"
    r"generated\s+(by|on)\s+|autogenerated|do\s+not\s+edit"
    r")",
    re.IGNORECASE,
)

HEDGE = re.compile(
    r"\b(it\s+is\s+(important|worth|useful|helpful)\s+to\s+(note|mention|remember|understand)|"
    r"as\s+(previously\s+)?(mentioned|noted|discussed|stated)|"
    r"in\s+other\s+words|that\s+is\s+to\s+say|to\s+put\s+it\s+simply|"
    r"needless\s+to\s+say|as\s+you\s+(may\s+)?know|keep\s+in\s+mind\s+that|"
    r"for\s+(what\s+it'?s\s+worth|the\s+record)|at\s+the\s+end\s+of\s+the\s+day)\b",
    re.IGNORECASE,
)

EXAMPLE_LEAD = re.compile(
    r"^\s*(for\s+(example|instance)|e\.?g\.?|such\s+as|consider\s+the\s+following|"
    r"here'?s\s+an\s+example|as\s+an\s+example|imagine|suppose)\b",
    re.IGNORECASE,
)

#: Language keyword sets used by the doctype detector and the code pipeline.
CODE_KEYWORDS: dict[str, frozenset[str]] = {
    "python": frozenset(
        ["def", "class", "import", "from", "return", "yield", "lambda", "async", "await", "with", "as", "if", "elif", "else", "for", "while", "try", "except", "finally", "raise", "pass", "None", "True", "False", "self", "assert", "global", "nonlocal", "del", "in", "is", "not", "and", "or", "elif", "print"]
    ),
    "javascript": frozenset(
        ["function", "const", "let", "var", "return", "export", "import", "from", "class", "extends", "async", "await", "new", "this", "null", "undefined", "typeof", "instanceof", "interface", "type", "enum", "implements", "constructor", "super", "require", "module", "exports", "=>", "console"]
    ),
    "java": frozenset(
        ["public", "private", "protected", "class", "interface", "extends", "implements", "static", "final", "void", "new", "return", "package", "import", "throws", "try", "catch", "finally", "synchronized", "abstract", "boolean", "int", "long", "double", "String", "System"]
    ),
    "go": frozenset(
        ["func", "package", "import", "var", "const", "type", "struct", "interface", "map", "chan", "go", "defer", "return", "range", "if", "else", "for", "switch", "case", "nil", "err", "string", "int", "error", "make"]
    ),
    "rust": frozenset(
        ["fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use", "mod", "match", "if", "else", "for", "while", "loop", "return", "Some", "None", "Ok", "Err", "Result", "Option", "Vec", "String", "self", "crate", "where", "dyn"]
    ),
    "c": frozenset(
        ["include", "define", "int", "char", "void", "float", "double", "struct", "typedef", "return", "if", "else", "for", "while", "switch", "case", "break", "continue", "sizeof", "static", "const", "unsigned", "NULL", "printf"]
    ),
    "sql": frozenset(
        ["select", "from", "where", "join", "inner", "left", "right", "outer", "group", "order", "by", "having", "limit", "insert", "into", "values", "update", "set", "delete", "create", "table", "alter", "drop", "index", "primary", "key", "foreign", "references", "distinct", "union", "as", "on", "and", "or", "not", "null", "count", "sum", "avg"]
    ),
}

LEGAL_MARKERS = re.compile(
    r"\b(hereinafter|whereas|thereof|herein|hereto|hereunder|aforementioned|"
    r"pursuant\s+to|in\s+accordance\s+with|shall\s+be\s+deemed|the\s+parties|"
    r"this\s+agreement|force\s+majeure|indemnif\w+|warrant\w+|liabilit\w+|"
    r"governing\s+law|jurisdiction|termination|confidential\s+information|"
    r"section\s+\d+(\.\d+)*|article\s+[IVXLC\d]+|clause\s+\d+)\b",
    re.IGNORECASE,
)

APIDOC_MARKERS = re.compile(
    r"(\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/|"
    r"\b(200|201|204|301|400|401|403|404|409|422|429|500|502|503)\b\s*[-:( ]|"
    r"\b(request|response)\s+(body|schema|parameters?|headers?)\b|"
    r"\bcurl\s+-X\b|\bapplication/json\b|\bbearer\s+token\b|"
    r"\{[a-z_]+\}/|\b(endpoint|rate\s?limit|pagination|webhook)\b)",
    re.IGNORECASE,
)

LOG_LEVEL = re.compile(
    r"\b(TRACE|DEBUG|INFO|INFORMATION|NOTICE|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL|SEVERE|PANIC)\b"
)

TIMESTAMP = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}"
    r"|\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r"|\[\d+\.\d+\]"
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
)

ROLE_MARKER = re.compile(
    r"^\s*(?:\*\*)?(user|human|assistant|ai|system|bot|agent|customer|support|"
    r"client|rep|caller|operator|q|a)(?:\*\*)?\s*[:>\]]",
    re.IGNORECASE | re.MULTILINE,
)
