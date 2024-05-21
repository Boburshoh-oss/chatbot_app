# import regex as re

# def detect_script(text):
#     # Define regex patterns for Latin and Cyrillic scripts
#     latin_pattern = re.compile(r'\p{Latin}+')
#     cyrillic_pattern = re.compile(r'\p{Cyrillic}+')

#     latin_matches = latin_pattern.findall(text)
#     cyrillic_matches = cyrillic_pattern.findall(text)

#     return {
#         'latin': latin_matches,
#         'cyrillic': cyrillic_matches
#     }

# # Example usage
# text = "Hello, Привет, Salom!"
# result = detect_script(text)

# print("Latin script:", result['latin'])
# print("Cyrillic script (Krill):", result['cyrillic'])

import regex as re


def detect_script(text):
    # Define regex patterns for Latin and Cyrillic scripts
    latin_pattern = re.compile(r"\p{Latin}+")
    cyrillic_pattern = re.compile(r"\p{Cyrillic}+")

    # Find all matches for each script
    latin_matches = latin_pattern.findall(text)
    cyrillic_matches = cyrillic_pattern.findall(text)

    # Count the number of characters in each script
    latin_count = sum(len(match) for match in latin_matches)
    cyrillic_count = sum(len(match) for match in cyrillic_matches)

    # Determine the predominant script
    if latin_count > cyrillic_count:
        return "Latin"
    elif cyrillic_count > latin_count:
        return "Cyrillic (Krill)"
    else:
        return "Mixed or Undetermined"


# Example usage
text1 = "Hello, Привет, Salom!"
text2 = "Bu bir lotin matni."
text3 = "Бу бир кирилл матни."
text4 = "Я тебя люблю"

print(f"Text 1 is predominantly: {detect_script(text1)}")
print(f"Text 2 is predominantly: {detect_script(text2)}")
print(f"Text 3 is predominantly: {detect_script(text3)}")
print(f"Text 4 is predominantly: {detect_script(text4)}")
