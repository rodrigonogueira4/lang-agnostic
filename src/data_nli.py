import random
import string
from dataclasses import dataclass

from datasets import load_dataset, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


@dataclass
class DataDef:
    dataset_name: str
    premise_column: str
    hypothesis_column: str
    label_column: str
    num_labels: int


def data_def_name(dataset_name, subdataset_name):
    return f"{dataset_name}:{subdataset_name}"


DATA_DEFS = {
    "glue:mnli": DataDef("mnli", "premise", "hypothesis", "label", 3),
    "assin2:None": DataDef("assin2", "premise", "hypothesis", "entailment_judgment", 2),
}


def extract_seq2seq_features(
    dataset_definition, tokenizer, max_length, target_max_length, dataset,
    transliteration: bool = False,
    min_chars: int = 3,
    max_chars: int = 7,
    validation_set: str = '',
):
    hypothesis = dataset_definition.hypothesis_column
    premise = dataset_definition.premise_column
    label_column = dataset_definition.label_column

    def _preprocess_sample(example):
        sentence = f"premise: {example[premise]}. hypothesis: {example[hypothesis]}."
        original_label = example[label_column]

        encoded = tokenizer(
            sentence,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        target_encoded = tokenizer(
            str(original_label),
            max_length=target_max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        encoded["target_ids"] = target_encoded["input_ids"]
        encoded["label"] = original_label

        return encoded

    if transliteration:
        vocab = {}
        for example in dataset[validation_set]:
            text = ' '.join([example[premise], example[hypothesis]])
            for word in text.split():
                while word not in vocab:       
                    num_chars = random.randint(min_chars, max_chars)
                    new_word = ''.join(random.sample(string.ascii_lowercase, num_chars))
                    if word.istitle():
                        new_word = new_word.title()
                    if word.isupper():
                        new_word = new_word.upper()
                    if new_word in vocab: continue
                    vocab[word] = new_word
            
        def _apply_transliteration(text: str):
            return ' '.join([vocab[word] for word in text.split()])

        def _modify_example(example):
            example[premise] = _apply_transliteration(example[premise])
            example[hypothesis] = _apply_transliteration(example[hypothesis])
            return example
            
        dataset[validation_set] = dataset[validation_set].map(_modify_example, batched=False)
   
    features = dataset.map(
        _preprocess_sample, batched=False, remove_columns=[hypothesis, premise, label_column]
    )

    return features


class NLIDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        train_dataset: str,
        train_subdataset: str,
        validation_set: str,
        batch_size: int = 32,
        max_length: int = 256,
        target_max_length: int = 5,
        xlang_dataset_name: str = None,
        xlang_subdataset_name: str = None,
        xlang_validation_set: str = None,
        transliteration: bool = False,
        min_chars: int = 3,
        max_chars: int = 7,
        max_training_examples: int = None,
        max_validation_examples: int = None,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.train_dataset = train_dataset
        self.train_subdataset = train_subdataset
        self.validation_set = validation_set

        self.data_def = DATA_DEFS[data_def_name(train_dataset, train_subdataset)]

        self.xlang_dataset_name = xlang_dataset_name
        self.xlang_subdataset_name = xlang_subdataset_name
        self.xlang_validation_set = xlang_validation_set

        self.xlang_data_def = (
            DATA_DEFS[data_def_name(xlang_dataset_name, xlang_subdataset_name)]
            if xlang_dataset_name
            else None
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.target_max_length = target_max_length

        self.collate_fn = DataCollatorWithPadding(
            self.tokenizer, padding="max_length", max_length=self.max_length, return_tensors="pt")

        self.transliteration = transliteration
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_training_examples = max_training_examples
        self.max_validation_examples = max_validation_examples

    @property
    def train_size(self):
        return self.__train_size or 0

    def prepare_data(self) -> None:
        load_dataset(self.train_dataset, self.train_subdataset)

        if self.xlang_dataset_name:
            load_dataset(self.xlang_dataset_name, self.xlang_subdataset_name)

    def setup(self, stage):
        dataset = load_dataset(self.train_dataset, self.train_subdataset)

        if self.max_training_examples:
            dataset['train'] = dataset['train'].select(range(self.max_training_examples))
            
        if self.max_validation_examples:
            dataset[self.validation_set] = dataset[self.validation_set].select(
                range(self.max_validation_examples))
            
        features = extract_seq2seq_features(
            self.data_def,
            self.tokenizer,
            self.max_length,
            self.target_max_length,
            dataset,
        )

        self.__train_dataset_obj = features["train"]
        self.__valid_dataset_obj = features[self.validation_set]
        self.__train_size = len(self.__train_dataset_obj) // self.batch_size

        if self.xlang_dataset_name:
            xlang_dataset = load_dataset(
                self.xlang_dataset_name, self.xlang_subdataset_name
            )

            if self.max_training_examples:
                xlang_dataset['train'] = xlang_dataset['train'].select(range(self.max_training_examples))
                
            if self.max_validation_examples:
                xlang_dataset[self.xlang_validation_set] = xlang_dataset[self.xlang_validation_set].select(
                    range(self.max_validation_examples))

            xlang_features = extract_seq2seq_features(
                self.xlang_data_def,
                self.tokenizer,
                self.max_length,
                self.target_max_length,
                xlang_dataset,
                transliteration=self.transliteration,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
                validation_set=self.xlang_validation_set,
            )

            self.__cross_valid_dataset_obj = xlang_features[self.xlang_validation_set]

    def train_dataloader(self):
        return self._create_dataloader(self.__train_dataset_obj, True)

    def val_dataloader(self):
        val_loader1 = self._create_dataloader(self.__valid_dataset_obj)

        if not self.xlang_dataset_name:
            return val_loader1

        val_loader2 = self._create_dataloader(self.__cross_valid_dataset_obj)

        return [val_loader1, val_loader2]

    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        batch_size: int = None,
    ):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=self.collate_fn
        )
