from datasets import DatasetDict, load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.pt")

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

training_dataset.save_to_disk("./dataset/untokenized/wikipedia_pt_train")
test_dataset.save_to_disk("./dataset/untokenized/wikipedia_pt_test")
val_dataset.save_to_disk("./dataset/untokenized/wikipedia_pt_val")