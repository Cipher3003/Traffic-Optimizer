import re

# Read the original .net.xml file
with open(r"C:\Users\sai\OneDrive\Documents\TLO\Traffic-Optimizer\ggn\osm.net.xml", "r", encoding="utf-8") as file:
    xml_data = file.read()

# Regex pattern to find all junctions with type="priority"
junction_pattern = r'<junction id="([^"]+)"[^>]*type="priority"'

# Modify all junctions to type="traffic_light"
modified_xml_data = re.sub(junction_pattern, r'<junction id="\1" type="traffic_light"', xml_data)

# Find all junction IDs that were modified
junction_ids = re.findall(junction_pattern, xml_data)

# Create the osm.tll.xml content
tl_logic_entries = "<additional>\n"
for j_id in junction_ids:
    tl_logic_entries += f'''
    <tlLogic id="{j_id}" type="static" programID="1" offset="0">
        <phase duration="30" state="GrGr" />
        <phase duration="3" state="yryr" />
        <phase duration="30" state="rGrG" />
        <phase duration="3" state="ryry" />
    </tlLogic>
    '''
tl_logic_entries += "\n</additional>"

# Save the modified .net.xml file
with open("osm_modified.net.xml", "w", encoding="utf-8") as file:
    file.write(modified_xml_data)

# Save the generated traffic light logic file
with open("osm.tll.xml", "w", encoding="utf-8") as file:
    file.write(tl_logic_entries)

print("Traffic lights updated in osm_modified.net.xml and osm.tll.xml created successfully.")
