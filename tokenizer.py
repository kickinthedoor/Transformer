import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, vocab_size=32000, char_coverage=0.9995):
        self.vocab_size = vocab_size
        self.char_coverage = char_coverage
        self.processor = spm.SentencePieceProcessor()
        self.sos_id = None
        self.eos_id = None
        self.pad_id = 0

    def train_model(self, input_file: str, model_prefix: str):
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.char_coverage,
            model_type='bpe',
            pad_id=0,
            unk_id=3,
            control_symbols=['<s>', '</s>']
        )

    def load_model(self, model_file: str):
        """Loads a .model file into an existing instance."""
        self.processor.Load(model_file)
        self.sos_id = self.processor.piece_to_id('<s>')
        self.eos_id = self.processor.piece_to_id('</s>')
        self.pad_id = 0

    @classmethod
    def from_file(cls, model_prefix_or_file: str) -> "SentencePieceTokenizer":
        """
        Factory method to instantiate and load a SentencePiece model directly from a file path.
        Handles both 'prefix' (adds .model) and direct 'file.model' paths.
        """
        instance = cls()
        model_file = model_prefix_or_file if model_prefix_or_file.endswith('.model') else f"{model_prefix_or_file}.model"
        instance.load_model(model_file)
        return instance

    def encode(self, text: str, sample: bool = False, alpha: float = 0.1) -> list[int]:
        """
        sample=True enables BPE-dropout style subword regularization: merges are
        randomly skipped (with probability related to `alpha`), so the same text
        can segment differently across calls. Use only for training data — keep
        sample=False (deterministic) for eval/inference, where you want the
        single best segmentation.
        """
        if sample:
            ids = self.processor.SampleEncodeAsIds(text, -1, alpha)
        else:
            ids = self.processor.EncodeAsIds(text)
        return [self.sos_id] + ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return self.processor.DecodeIds(ids)