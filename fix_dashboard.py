import re

with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Violin fillcolor
content = content.replace('fillcolor=clr + "20"', 'fillcolor=f"rgba({int(clr[1:3], 16)}, {int(clr[3:5], 16)}, {int(clr[5:7], 16)}, 0.15)"')

# Fix radar chart fillcolor
content = content.replace('fillcolor=clr.replace("#", "rgba(")[:-1] + ", .08)" if clr.startswith("#") else clr', 'fillcolor=f"rgba({int(clr[1:3], 16)}, {int(clr[3:5], 16)}, {int(clr[5:7], 16)}, 0.08)"')

# Fix deprecation warning
content = content.replace('use_container_width=True', 'width="stretch"')
content = content.replace('use_container_width=False', 'width="content"')

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Dashboard updated successfully.')
