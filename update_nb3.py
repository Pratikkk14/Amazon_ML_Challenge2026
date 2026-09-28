import json

with open("MLV1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read entity_resolver_cli.py content to ensure notebook also shares the exact instant checkpoint loader
with open("entity_resolver_cli.py", "r", encoding="utf-8") as f:
    code = f.read()

# Filter out shebang if needed
code_lines = [line + "\n" for line in code.split("\n")]
if code_lines and code_lines[0].startswith("#!"):
    code_lines = code_lines[1:]

nb["cells"][2]["source"] = code_lines
nb["cells"][2]["source"][-1] = nb["cells"][2]["source"][-1].rstrip("\n")

with open("MLV1.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("MLV1.ipynb updated with existing checkpoint auto-loader!")
