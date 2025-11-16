import json

def append_to_skg(new_event, prev_id='e5', path='data/skg.json'):
    with open(path, 'r') as f:
        data = json.load(f)

    new_id = f"e{len(data['nodes']) + 1}"
    data['nodes'].append({"id": new_id, "event": new_event})
    data['edges'].append({"source": prev_id, "target": new_id, "relation": "next"})

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return new_id
