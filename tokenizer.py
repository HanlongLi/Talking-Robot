"""Robot communication tokenizer with 128-token vocabulary.

Vocabulary layout (128 tokens):
    0       BLANK (CTC)
    1       PAD
    2       SOS
    3       EOS
    4       SPACE
    5       UNK
    6-31    a-z  (26 letters)
    32-41   0-9  (10 digits)
    42-57   punctuation
    58-83   reserved
    84-127  robot command tokens (<STOP>, <GO>, ...)
"""

BLANK_ID = 0
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3
SPACE_ID = 4
UNK_ID = 5
NUM_SPECIAL = 6
VOCAB_SIZE = 128

LETTER_OFFSET = 6
DIGIT_OFFSET = 32
PUNCT_OFFSET = 42
RESERVED_OFFSET = 58
CMD_OFFSET = 84

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCTUATION = ".,!?-:;'\"()/@ #_+"

ROBOT_COMMANDS = {
    "<STOP>": 84, "<GO>": 85, "<FWD>": 86, "<BWD>": 87,
    "<LEFT>": 88, "<RIGHT>": 89, "<UP>": 90, "<DOWN>": 91,
    "<FAST>": 92, "<SLOW>": 93, "<HOME>": 94, "<GOTO>": 95,
    "<GRAB>": 96, "<DROP>": 97, "<LIFT>": 98, "<LOWER>": 99,
    "<PUSH>": 100, "<PULL>": 101, "<OPEN>": 102, "<CLOSE>": 103,
    "<ROTATE>": 104, "<SCAN>": 105, "<POINT>": 106, "<REACH>": 107,
    "<ACK>": 108, "<NACK>": 109, "<READY>": 110, "<WAIT>": 111,
    "<ALERT>": 112, "<EMERG>": 113, "<STATUS>": 114, "<REPORT>": 115,
    "<YES>": 116, "<NO>": 117, "<OK>": 118, "<FAIL>": 119,
    "<SYNC>": 120, "<LEAD>": 121, "<FOLLOW>": 122, "<FORM>": 123,
    "<IDLE>": 124, "<BUSY>": 125, "<CHARGE>": 126, "<ERROR>": 127,
}

ID_TO_COMMAND = {v: k for k, v in ROBOT_COMMANDS.items()}


class RobotTokenizer:
    """Character-level tokenizer with robot command token support.

    Encodes text into integer sequences using a fixed 128-token vocabulary.
    Command tokens (e.g. ``<STOP>``, ``<GO>``) are matched greedily.
    """

    def __init__(self):
        self._c2i = {}
        self._i2c = {
            BLANK_ID: "", PAD_ID: "", SOS_ID: "<sos>", EOS_ID: "<eos>",
            SPACE_ID: " ", UNK_ID: "\u00bf",
        }
        for i, ch in enumerate(LETTERS):
            self._c2i[ch] = LETTER_OFFSET + i
            self._i2c[LETTER_OFFSET + i] = ch
        for i, ch in enumerate(DIGITS):
            self._c2i[ch] = DIGIT_OFFSET + i
            self._i2c[DIGIT_OFFSET + i] = ch
        for i, ch in enumerate(PUNCTUATION):
            self._c2i[ch] = PUNCT_OFFSET + i
            self._i2c[PUNCT_OFFSET + i] = ch
        self._c2i[" "] = SPACE_ID
        for i in range(26):
            self._i2c[RESERVED_OFFSET + i] = f"<R{i}>"
        for cmd, cid in ROBOT_COMMANDS.items():
            self._i2c[cid] = cmd
        self._cmds = sorted(ROBOT_COMMANDS, key=len, reverse=True)

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def encode(self, text: str) -> list:
        """Encode a text string into a list of token IDs."""
        tokens = []
        i = 0
        while i < len(text):
            hit = False
            if text[i] == "<":
                for cmd in self._cmds:
                    if text[i : i + len(cmd)] == cmd:
                        tokens.append(ROBOT_COMMANDS[cmd])
                        i += len(cmd)
                        hit = True
                        break
            if hit:
                continue
            tokens.append(self._c2i.get(text[i].lower(), UNK_ID))
            i += 1
        return tokens

    def decode(self, ids: list, skip_special: bool = True) -> str:
        """Decode a list of token IDs back into a string."""
        parts = []
        for t in ids:
            if skip_special and t in {BLANK_ID, PAD_ID, SOS_ID, EOS_ID}:
                continue
            parts.append(self._i2c.get(t, "\u00bf"))
        return "".join(parts)
