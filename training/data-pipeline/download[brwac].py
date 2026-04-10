from datasets import load_dataset

ds = load_dataset("dominguesm/brwac")

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

training_dataset.save_to_disk("./dataset/untokenized/brwac_train")
test_dataset.save_to_disk("./dataset/untokenized/brwac_test")
val_dataset.save_to_disk("./dataset/untokenized/brwac_val")