"""
Parse FullKnowledge objects from the simulated_table_info.csv file.

This module provides utilities to parse string representations of FullKnowledge
objects from CSV files and convert them back into proper Pydantic model instances.

You can use this in a Jupyter notebook like:

    import sys
    sys.path.append('/home/kunumi/Área de trabalho/kunumi/kontex/src')
    from kontex.simulation.edd.table_knowledge import DomainKnowledge, FullKnowledge
    from parse_full_knowledge import read_full_knowledge_from_csv

    # Read a specific row
    knowledge = read_full_knowledge_from_csv(
        '/home/kunumi/Área de trabalho/kunumi/kontex/data/simulated_table_info.csv',
        row_index=0
    )

    # Or read all rows
    all_knowledge = read_all_full_knowledge_from_csv(
        '/home/kunumi/Área de trabalho/kunumi/kontex/data/simulated_table_info.csv'
    )
"""
import csv
import re
from pathlib import Path


def parse_full_knowledge_from_string(knowledge_str: str, FullKnowledge, DomainKnowledge):
    """
    Parse a string representation of FullKnowledge and return a FullKnowledge instance.

    The string format is like:
    "title='mining' domains={'StratumExtract': DomainKnowledge(title='StratumExtract',
    description='...', facts={'BLK_REF': '...', ...}), ...}"

    Parameters
    ----------
    knowledge_str : str
        String representation of a FullKnowledge object
    FullKnowledge : type
        The FullKnowledge class (imported from kontex.simulation.edd.table_knowledge)
    DomainKnowledge : type
        The DomainKnowledge class (imported from kontex.simulation.edd.table_knowledge)

    Returns
    -------
    FullKnowledge
        Instantiated FullKnowledge object with all domains and facts
    """
    # Extract the title
    title_match = re.search(r"title='([^']*)'", knowledge_str)
    if not title_match:
        raise ValueError("Could not extract title from knowledge string")
    title = title_match.group(1)

    # Initialize the FullKnowledge object
    knowledge = FullKnowledge(title=title)

    # Extract domains={'...'} section
    domains_match = re.search(r"domains=\{(.*)\}\s*$", knowledge_str, re.DOTALL)
    if not domains_match:
        # Return empty knowledge if no domains
        return knowledge

    domains_section = domains_match.group(1)

    # Parse each domain entry by finding DomainKnowledge boundaries
    current_pos = 0
    while True:
        # Find next domain key
        key_match = re.search(r"'([^']+)':\s*DomainKnowledge\(", domains_section[current_pos:])
        if not key_match:
            break

        domain_key = key_match.group(1)
        start_pos = current_pos + key_match.end()

        # Find the matching closing parenthesis for this DomainKnowledge
        paren_count = 1
        i = start_pos
        in_string = False
        escape_next = False

        while i < len(domains_section) and paren_count > 0:
            char = domains_section[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            if char == "'" and (i == 0 or domains_section[i-1] != '\\'):
                in_string = not in_string
            elif not in_string:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            i += 1

        domain_content = domains_section[start_pos:i-1]

        # Parse domain_content to extract title, description, and facts
        domain_title_match = re.search(r"title='([^']*)'", domain_content)
        domain_desc_match = re.search(
            r"description='((?:[^'\\]|\\.)*)'(?=,?\s*facts=)",
            domain_content,
            re.DOTALL
        )

        domain_title = domain_title_match.group(1) if domain_title_match else domain_key
        domain_description = None
        if domain_desc_match:
            domain_description = domain_desc_match.group(1).replace("\\'", "'")

        # Extract facts dictionary
        facts = {}
        facts_match = re.search(r"facts=\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", domain_content, re.DOTALL)
        if facts_match:
            facts_content = facts_match.group(1)
            # Parse key-value pairs in facts
            # Pattern: 'key': 'value', where value can contain escaped quotes
            fact_pattern = r"'([^']+)':\s*'((?:[^'\\]|\\.)*)'"
            for fact_match in re.finditer(fact_pattern, facts_content):
                fact_key = fact_match.group(1)
                fact_value = fact_match.group(2).replace("\\'", "'").replace('\\n', '\n').replace('\\"', '"')
                facts[fact_key] = fact_value

        # Create DomainKnowledge and add to FullKnowledge
        domain = DomainKnowledge(
            title=domain_title,
            description=domain_description,
            facts=facts
        )
        knowledge.domains[domain_key] = domain

        # Move to next domain
        current_pos = i
        # Skip comma and whitespace
        while current_pos < len(domains_section) and domains_section[current_pos] in ', \n\t':
            current_pos += 1

    return knowledge


def read_full_knowledge_from_csv(csv_path: str | Path, row_index: int = 0, FullKnowledge=None, DomainKnowledge=None):
    """
    Read a FullKnowledge object from the CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the simulated_table_info.csv file
    row_index : int, default=0
        Which row to read (0-indexed, not counting header)
    FullKnowledge : type, optional
        The FullKnowledge class. If None, will import from kontex
    DomainKnowledge : type, optional
        The DomainKnowledge class. If None, will import from kontex

    Returns
    -------
    FullKnowledge
        Parsed FullKnowledge object
    """
    if FullKnowledge is None or DomainKnowledge is None:
        from kontex.simulation.edd.general_knowledge import DomainKnowledge, FullKnowledge

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == row_index:
                return parse_full_knowledge_from_string(row['full_knowledge'], FullKnowledge, DomainKnowledge)

    raise IndexError(f"Row index {row_index} not found in CSV")


def read_all_full_knowledge_from_csv(csv_path: str | Path, FullKnowledge=None, DomainKnowledge=None):
    """
    Read all FullKnowledge objects from the CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the simulated_table_info.csv file
    FullKnowledge : type, optional
        The FullKnowledge class. If None, will import from kontex
    DomainKnowledge : type, optional
        The DomainKnowledge class. If None, will import from kontex

    Returns
    -------
    list[FullKnowledge]
        List of all parsed FullKnowledge objects
    """
    if FullKnowledge is None or DomainKnowledge is None:
        from kontex.simulation.edd.general_knowledge import DomainKnowledge, FullKnowledge

    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            knowledge = parse_full_knowledge_from_string(row['full_knowledge'], FullKnowledge, DomainKnowledge)
            results.append(knowledge)

    return results


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('/home/kunumi/Área de trabalho/kunumi/kontex/src')
    from kontex.simulation.edd.general_knowledge import DomainKnowledge, FullKnowledge

    csv_path = "/home/kunumi/Área de trabalho/kunumi/kontex/data/simulated_table_info.csv"

    # Read first row
    knowledge = read_full_knowledge_from_csv(csv_path, row_index=0, FullKnowledge=FullKnowledge, DomainKnowledge=DomainKnowledge)
    print(f"Title: {knowledge.title}")
    print(f"Number of domains: {len(knowledge.domains)}")

    for domain_name, domain in knowledge.domains.items():
        print(f"\nDomain: {domain_name}")
        print(f"  Title: {domain.title}")
        if domain.description:
            print(f"  Description: {domain.description[:100]}...")
        else:
            print("  Description: None")
        print(f"  Number of facts: {len(domain.facts)}")
        if domain.facts:
            first_fact = list(domain.facts.items())[0]
            print(f"  First fact: {first_fact[0]}: {first_fact[1][:80]}...")
