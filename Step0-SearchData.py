"""
Step 0: Search Data for PRS Tools
This script provides instructions for creating a comprehensive table of 
polygenic risk score (PRS) tools with their associated resources.
"""

def print_research_instructions():
    """Print instructions for researching PRS tools."""
    instructions = """
    Use deep research from Google AI, Claude, or Perplexity to find the following:
    
    Task: Create a comprehensive table (in Markdown format) of polygenic risk score (PRS) tools.
    
    For each tool, include the following columns (links only):
    • Tool Name – Name of the PRS tool
    • Tool PDF Document – Link to a PDF user manual or guide. Leave blank if not available
    • GitHub Link – Link to the GitHub repository (if available)
    • R Package Link – Link to the R package (if available)
    • Python Package Link – Link to the Python package (if available)
    • Published Article – Link to the main published article
    • DOI Link – Direct DOI link to the publication
    • Website Link – Official website of the tool (if available)
    
    Instructions:
    • Provide the response in Markdown table format
    • Include only links in each table cell (no extra text or annotations)
    • Leave any cell empty if the corresponding link is not available
    """
    print(instructions)


def print_example_format():
    """Print example format for the PRS tools data."""
    print("\nExample format for the data:")
    print("=" * 120)
    
    # Print table header
    headers = ["Tool", "Tool Information", "PDF Manual", "Python", "R Package", "Website", "Language", "Article", "GitHub Readme"]
    print("\t".join(headers))
    print("-" * 120)
    
    # Example data rows
    example_rows = [
        [
            "ldpred2_inf",
            "[LDpred2 inf](https://privefl.github.io/bigsnpr/articles/LDpred2.html)",
            "NA",
            "NA",
            "https://github.com/privefl/bigsnpr",
            "https://privefl.github.io/bigsnpr/articles/LDpred2.html",
            "R",
            "[10.1093/bioinformatics/btaa1029](https://doi.org/10.1093/bioinformatics/btaa1029)",
            "https://github.com/privefl/bigsnpr"
        ],
        [
            "ldpred-funct",
            "[LDpred-funct](https://github.com/carlaml/LDpred-funct)",
            "NA",
            "https://github.com/carlaml/LDpred-funct",
            "NA",
            "NA",
            "Python",
            "[10.1038/s41467-021-25171-9](https://doi.org/10.1038/s41467-021-25171-9)",
            "https://github.com/carlaml/LDpred-funct"
        ],
        [
            "SBayesR",
            "[SBayesR](https://cnsgenomics.com/software/gctb/#Download)",
            "NA",
            "NA",
            "NA",
            "https://cnsgenomics.com/software/gctb/",
            "C++",
            "[10.1038/s41588-024-01704-y](https://doi.org/10.1038/s41588-024-01704-y)",
            ""
        ],
        [
            "SBayesRC",
            "[SBayesRC](https://cnsgenomics.com/software/gctb/#Download)",
            "NA",
            "NA",
            "https://github.com/zhilizheng/SBayesRC",
            "https://cnsgenomics.com/software/gctb/",
            "C++",
            "[10.1038/s41467-019-12653-0](https://doi.org/10.1038/s41467-019-12653-0)",
            "https://github.com/zhilizheng/SBayesRC"
        ],
        [
            "LDAK-gwas",
            "[LDAK-gwas](https://dougspeed.com/quick-prs/)",
            "NA",
            "NA",
            "https://dougspeed.com/quick-prs/",
            "https://dougspeed.com/quick-prs/",
            "C++",
            "[10.1038/s41467-021-24485-y](https://doi.org/10.1038/s41467-021-24485-y)",
            "https://github.com/dougspeed/LDAK"
        ],
        [
            "PRScs",
            "[PRScs](https://github.com/getian107/PRScs)",
            "NA",
            "https://github.com/getian107/PRScs",
            "NA",
            "NA",
            "Python",
            "[10.1038/s41467-019-09718-5](https://doi.org/10.1038/s41467-019-09718-5)",
            "https://github.com/getian107/PRScs"
        ]
    ]
    
    # Print each row
    for row in example_rows:
        print("\t".join(row))
    
    print("-" * 120)


def print_reference_note():
    """Print reference note about the data format file."""
    print("\n" + "=" * 80)
    print("Reference: Check file 'PRSGPT-Tools-DatasetLinks.xlsx' to verify the format")
    print("=" * 80)


def main():
    """Main function to execute the script."""
    print("=" * 80)
    print("PRS TOOLS DATA COLLECTION INSTRUCTIONS")
    print("=" * 80)
    
    print_research_instructions()
    print_example_format()
    print_reference_note()


if __name__ == "__main__":
    main()


