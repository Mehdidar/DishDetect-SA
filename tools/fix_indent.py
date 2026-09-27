import re

with open('ui.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # lines 108 to 136 (0-indexed) need +4 spaces
    # lines 138 to the end are already 12 spaces, but they need to be properly aligned
    if 108 <= i <= 136:
        if line.strip():
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    elif i > 136:
        # they are already 12 spaces, but let's just make sure they are exact
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('ui.py', 'w') as f:
    f.writelines(new_lines)
