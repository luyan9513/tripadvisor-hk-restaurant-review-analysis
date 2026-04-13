POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "brilliant", "cheerful", "clean",
    "comfortable", "cozy", "crisp", "crispy", "decent", "delectable", "delicious",
    "delightful", "enjoy", "enjoyable", "enjoyed", "excellent", "exceptional",
    "fantastic", "favorite", "favourite", "flavorful", "flavourful", "fresh",
    "friendly", "good", "great", "helpful", "impeccable", "impressive",
    "incredible", "juicy", "kind", "lovely", "luscious", "memorable", "nice",
    "outstanding", "perfect", "phenomenal", "pleasant", "polite", "prompt",
    "quick", "recommend", "recommended", "satisfying", "satisfied", "scrumptious",
    "spectacular", "stellar", "superb", "tasty", "tender", "toothsome", "top",
    "unforgettable", "vibrant", "warm", "welcoming", "wonderful", "worth", "worthy",
    "wow", "yummy",
}

NEGATIVE_WORDS = {
    "annoyed", "annoying", "average", "awful", "bad", "bland", "boring", "burnt",
    "chaotic", "cold", "complaint", "concern", "concerned", "confusing", "crowded",
    "delayed", "dirty", "disappoint", "disappointed", "disappointing", "dry",
    "expensive", "frustrated", "frustrating", "greasy", "horrible", "hostile",
    "ignore", "ignored", "inconsistent", "mediocre", "messy", "mistake", "noisy",
    "overcooked", "overhyped", "overpriced", "poor", "pricey", "rude", "salty",
    "sad", "slow", "soggy", "stale", "subpar", "surprised", "tasteless", "terrible",
    "tough", "underwhelming", "unfriendly", "unhelpful", "unpleasant", "unsafe",
    "upset", "wait", "waiting", "warmish", "waste", "worried", "worst", "wrong",
}

NEGATIONS = {
    "aint", "aren't", "can't", "cannot", "didn't", "doesn't", "dont", "don't",
    "hardly", "isn't", "lack", "lacking", "neither", "never", "no", "none", "nor",
    "not", "nothing", "rarely", "scarcely", "wasn't", "weren't", "without", "won't",
}

INTENSIFIERS = {
    "absolutely", "amazingly", "deeply", "especially", "exceptionally", "extremely",
    "highly", "incredibly", "particularly", "really", "remarkably", "so", "super",
    "too", "totally", "truly", "very",
}

EMOTION_LEXICONS = {
    "happy": {
        "amazing", "cheerful", "delight", "delighted", "delightful", "enjoy",
        "enjoyable", "enjoyed", "excellent", "fantastic", "glad", "great", "happy",
        "impressed", "incredible", "joy", "joyful", "love", "loved", "lovely",
        "perfect", "pleasant", "pleased", "satisfying", "satisfied", "superb",
        "thrilled", "wonderful", "wow",
    },
    "anger": {
        "angry", "annoyed", "annoying", "appalling", "awful", "furious",
        "frustrated", "frustrating", "horrible", "hostile", "insulting", "mad",
        "outrageous", "pathetic", "rude", "terrible", "unacceptable", "upset",
        "worst",
    },
    "sad": {
        "depressing", "disappoint", "disappointed", "disappointing", "heartbroken",
        "letdown", "regret", "regretted", "sad", "sorry", "tragic", "unhappy",
        "underwhelming", "upsetting",
    },
    "fear": {
        "afraid", "alarm", "anxious", "concern", "concerned", "fear", "fearful",
        "hesitant", "nervous", "panic", "risky", "scared", "suspicious", "unsafe",
        "wary", "worried",
    },
    "surprise": {
        "astonishing", "impressed", "incredible", "shock", "shocked", "shocking",
        "startled", "stunning", "surprise", "surprised", "surprising", "unexpected",
        "unusual", "wow",
    },
}
