import os

def generate_series(n, directory="output", file_name="Wear.txt"):
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Full file path
    output_file = os.path.join(directory, file_name)
    
    series = []
    
    for i in range(1, n + 1):  # Generate n pairs
        even_idx = 2 * (i // 2)  # Even index (alternates for two steps)
        if i % 2 == 0:  # If even iteration
            odd_idx = 2 * (i // 2) - 1  # Old odd index
        else:  # If odd iteration
            odd_idx = 2 * (i // 2) + 1  # New odd index

        # CSLIP2 term
        term_cslip = f"(abs(s3f{even_idx}_CSLIP2ROADTREADSURFROADSURF-s3f{odd_idx}_CSLIP2ROADTREADSURFROADSURF))"
        
        # CSHEAR2 term (with the same pairing pattern but summed)
        term_cshear = f"(abs(s3f{odd_idx}_CSHEARFROADTREADSURFROADSURF+s3f{even_idx}_CSHEARFROADTREADSURFROADSURF) / 2)"
        
        # Combine the terms into a single string
        series.append(f"{term_cslip} * {term_cshear}")
    
    # Write the series to a file
    with open(output_file, "w") as file:
        file.write(" +\n".join(series))
    print(f"Output successfully saved to {output_file}")


# Example usage
n = 90  # Define the number of pairs you want to generate
directory = r"G:\Godey\EXPLICIT"  # Specify the directory
file_name = "wear.txt"  # Specify the file name
generate_series(n, directory, file_name)