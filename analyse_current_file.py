import json

def analyze_current_player_values_from_jsonl(filename, player=-1):
    positive = 0
    negative = 0
    zero = 0
    total = 0

    with open(filename, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if entry.get('current_player') == player:
                value = entry.get('value', 0)
                if value > 0:
                    positive += 1
                elif value < 0:
                    negative += 1
                else:
                    zero += 1
                total += 1

    print(f"Results for current_player = {player}:")
    print(f"  Positive values: {positive}")
    print(f"  Negative values: {negative}")
    print(f"  Zero values:     {zero}")
    print(f"  Total analyzed:  {total}")

if __name__ == "__main__":
    analyze_current_player_values_from_jsonl("training_data.jsonl", 1)