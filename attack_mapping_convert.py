
import json


with open('./json/attack_mapping.json', 'r') as attack_mapping_file:
    attack_mapping = json.load(attack_mapping_file)

result = {}

for entry in attack_mapping:
    attack_id = entry['attack_id']
    eac_id = entry['eac_id']
    
    if attack_id not in result:
        result[attack_id] = []
    
    if eac_id not in result[attack_id]:
        result[attack_id].append(eac_id)

with open('./json/output.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)


