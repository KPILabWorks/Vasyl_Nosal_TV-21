from Levenshtein import distance

def merge_duplicates_with_levenshtein(dataset, threshold=1.1):
    merged = []
    
    for value in dataset:
        similar_found = False
        for merged_value in merged:
            if distance(value, merged_value) <= threshold:
                similar_found = True
                break

        if not similar_found:
            merged.append(value)

    return merged

with open('data.txt', 'r') as file:
    content = file.read()

content = content.replace('"', '')
fruit_list = content.split(', ')

merged_dataset = merge_duplicates_with_levenshtein(fruit_list, threshold=1)  # 1 символ різниці
print(merged_dataset)
