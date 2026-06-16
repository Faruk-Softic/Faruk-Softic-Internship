import json
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")  # closest available encoderw

with open("batch_input_50.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

total_tokens = 0
for line in lines:
    request = json.loads(line)
    # Count tokens in both instructions and input
    instructions = request["body"]["instructions"]
    input_text = request["body"]["input"]
    total_tokens += len(enc.encode(instructions + input_text))

print(f"Total input tokens: {total_tokens:,}")