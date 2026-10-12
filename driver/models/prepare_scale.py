"""Tokenize a longer input only; no inference and no precomputed model output."""
from pathlib import Path
import struct
from tokenizers import Tokenizer
# Public export: local tokenizer and generated prompt files.
root=Path(__file__).resolve().parent
tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
question=(
    'Write a complete short story of about three hundred words using these facts. '
    'Mira is a young lighthouse keeper on a rocky island. Her friend Tomas repairs fishing boats. '
    'One evening a storm damages the lighthouse lamp while a small boat is approaching the harbor. '
    'Mira finds a spare battery in a wooden cupboard. Tomas brings a polished mirror from his workshop. '
    'The two friends must use these objects to guide the boat safely home. '
    'Tell the story in plain English, with a beginning, a problem, and a happy ending. '
    'Use six paragraphs. Return only the story, without a title or commentary.'
)
prompt='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+question+'<|im_end|>\n<|im_start|>assistant\n'
ids=tokenizer.encode(prompt,add_special_tokens=False).ids
assert len(ids) <= 176, len(ids)
(root/'prompt_scale.tokens').write_bytes(struct.pack('<I',len(ids))+struct.pack('<'+'I'*len(ids),*ids))
(root/'prompt_scale.txt').write_text(prompt)
print('INPUT_ONLY prompt=scale tokens=',len(ids))
