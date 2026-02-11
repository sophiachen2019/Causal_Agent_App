
lines = []
with open('causal_agent_app.py', 'r') as f:
    lines = f.readlines()

# Find the line with "with tab_causal: # Ensure results render in the Causal Tab"
start_indent_idx = -1
for i, line in enumerate(lines):
    if "with tab_causal: # Ensure results render in the Causal Tab" in line:
        start_indent_idx = i + 1
        break

if start_indent_idx != -1:
    print(f"Indenting from line {start_indent_idx + 1}")
    for i in range(start_indent_idx, len(lines)):
        if lines[i].strip(): # Only indent non-empty lines
            lines[i] = "    " + lines[i]
    
    with open('causal_agent_app.py', 'w') as f:
        f.writelines(lines)
    print("Indentation fixed.")
else:
    print("Target line not found.")
