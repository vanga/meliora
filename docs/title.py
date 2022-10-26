import os
import glob
import re

package_name = "meliora"

for filepath in glob.glob(os.path.join("source/" + package_name, "*.rst")):
    if not package_name + "." in filepath:
        os.remove(filepath)
        continue

    with open(filepath) as file:
        lines = file.readlines()
    # file.close()

    line = re.search(r"\.([^\.\s]+)\s\b", lines[0])
    if not line:
        continue

    lines[0] = line.group().replace(".", "").replace("\_", " ") + "\n"

    with open(filepath, "w") as file:
        file.writelines(lines)
    # file.close()

print('Package module titles changed.')
