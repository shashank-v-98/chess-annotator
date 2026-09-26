"""Distillation pipeline: turn the prompted annotation pipeline into training
data for a small local model, then fine-tune (LoRA/SFT) that model to
reproduce the teacher's annotations without needing an API at inference
time.

Pipeline (run in this order):
    1. fetch_chesscom_games.py  -> examples/distill_games/*.pgn (real games)
    2. generate_dataset.py      -> distill/data/raw_dataset.jsonl (teacher-labeled)
    3. split_dataset.py         -> distill/data/{train,val,test}.jsonl
    4. train_on_kaggle.ipynb    -> LoRA adapter, trained/validated on Kaggle
    5. (back in chess_annotator) annotate.py --backend local --adapter ...
"""
