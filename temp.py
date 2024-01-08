import re

def process_csv(file_path, file_path_new):
    output_lines = []
    with open(file_path, 'r') as file:
        current_line = ''
        inside_quotes = False

        for line in file:
            # Count the number of quotes in this line
            num_quotes = line.count('"')

            # Toggle the inside_quotes flag if there's an odd number of quotes
            if num_quotes % 2 != 0:
                inside_quotes = not inside_quotes

            # If we're inside quotes, replace newlines with tabs
            if inside_quotes:
                current_line += line.replace('\n', '\t')
            else:
                # Append the current line to output_lines
                current_line += line
                output_lines.append(current_line)
                current_line = ''  # Reset for the next record

    # Write to the same file or a new file
    with open(file_path_new, 'w') as file:
        file.writelines(output_lines)

# Replace 'your_file.csv' with the path to your CSV file
process_csv('full_Table_networkLevel_.csv', 'newfull_Table_networkLevel_.csv')
