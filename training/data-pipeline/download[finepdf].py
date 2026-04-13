from datasets import load_dataset, Dataset
from tqdm import tqdm

ds = load_dataset("HuggingFaceFW/finepdfs-edu", "por_Latn")

subsplit = ds["train"].train_test_split(
    test_size=0.2,
    seed=42
)

training_dataset = subsplit["train"]
subsplit = subsplit["test"].train_test_split(
    test_size=0.5,
    seed=42
)

test_dataset = subsplit["train"]
val_dataset = subsplit["test"]

training_dataset.save_to_disk("./dataset/untokenized/finepdf_train")
test_dataset.save_to_disk("./dataset/untokenized/finepdf_test")
val_dataset.save_to_disk("./dataset/untokenized/finepdf_val")