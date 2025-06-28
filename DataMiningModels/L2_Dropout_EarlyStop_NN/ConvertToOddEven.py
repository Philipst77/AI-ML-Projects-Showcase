def process_truth_file(input_file, output_file):
    odd_count = 0

    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            number = int(line.strip())
            label = 1 if number % 2 == 1 else 0

            # Count odd numbers
            if label == 1:
                odd_count += 1

            # Write to the output file
            output_f.write(f'{label}\n')

    return odd_count

# Specify the paths to your input and output files
input_path = 'ground_truth_labels.txt'
output_path = 'truth-odd-even.txt'

# Call the function with the file paths
odd_count = process_truth_file(input_path, output_path)

# Output the count of odd numbers
#print(f'Number of odd numbers: {odd_count}')