"""MRZ (Machine Readable Zone) utilities."""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MRZData:
    """Parsed MRZ data structure."""
    document_type: str
    document_number: str
    document_number_check: str
    nationality: str
    date_of_birth: str
    date_of_birth_check: str
    sex: str
    date_of_expiry: str
    date_of_expiry_check: str
    surname: str
    given_names: str
    valid: bool = False
    

def compute_check_digit(data: str) -> int:
    """
    Compute ICAO 9303 check digit for MRZ data.
    
    Args:
        data: String to compute check digit for
        
    Returns:
        Check digit (0-9)
    """
    weights = [7, 3, 1]
    total = 0
    
    for i, char in enumerate(data):
        if char >= '0' and char <= '9':
            value = ord(char) - ord('0')
        elif char >= 'A' and char <= 'Z':
            value = ord(char) - ord('A') + 10
        elif char == '<':
            value = 0
        else:
            continue
        
        total += value * weights[i % 3]
    
    return total % 10


def validate_mrz(mrz_lines: list) -> Tuple[bool, Dict]:
    """
    Validate MRZ check digits.
    
    Args:
        mrz_lines: List of MRZ lines (should be 3 for TD1 format)
        
    Returns:
        Tuple of (is_valid, validation_details)
    """
    if len(mrz_lines) != 3:
        return False, {"error": "MRZ must have 3 lines for TD1 format"}
    
    line1, line2, line3 = mrz_lines
    
    # Check line lengths
    if not all(len(line) == 30 for line in mrz_lines):
        return False, {"error": "Each MRZ line must be 30 characters"}
    
    validation = {
        "document_number_valid": False,
        "date_of_birth_valid": False,
        "date_of_expiry_valid": False,
        "composite_valid": False,
    }
    
    # Line 1: I<NNNDDDDDDDDDCD<<<<<<<<<<<<<<<
    # Document number at positions 5-13 (9 chars), check digit at 14
    doc_number = line1[5:14]
    doc_check = line1[14]
    expected_doc_check = str(compute_check_digit(doc_number))
    validation["document_number_valid"] = (doc_check == expected_doc_check)
    
    # Line 2: CDDOBCDOOOCD<<<<<<<<<<<<<<<<<<<
    # DOB at positions 0-5, check digit at 6
    dob = line2[0:6]
    dob_check = line2[6]
    expected_dob_check = str(compute_check_digit(dob))
    validation["date_of_birth_valid"] = (dob_check == expected_dob_check)
    
    # Expiry at positions 8-13 (after sex at position 7), check digit at 14
    expiry = line2[8:14]
    expiry_check = line2[14]
    expected_expiry_check = str(compute_check_digit(expiry))
    validation["date_of_expiry_valid"] = (expiry_check == expected_expiry_check)
    
    # Optional data check (positions 15-20)
    optional_data = line2[15:21]
    optional_check = line2[21] if len(line2) > 21 else ''
    if optional_check and optional_check != '<':
        expected_optional_check = str(compute_check_digit(optional_data))
        validation["optional_data_valid"] = (optional_check == expected_optional_check)
    
    is_valid = all([
        validation["document_number_valid"],
        validation["date_of_birth_valid"],
        validation["date_of_expiry_valid"],
    ])
    
    return is_valid, validation


def parse_mrz(mrz_lines: list) -> Optional[MRZData]:
    """
    Parse MRZ lines into structured data.
    
    Args:
        mrz_lines: List of 3 MRZ lines (TD1 format)
        
    Returns:
        MRZData object or None if parsing fails
    """
    if len(mrz_lines) != 3:
        return None
    
    try:
        line1, line2, line3 = mrz_lines
        
        # Line 1 parsing
        doc_type = line1[0:2]  # ID
        nationality = line1[2:5]  # DZA
        doc_number = line1[5:14]
        doc_check = line1[14]
        
        # Line 2 parsing
        dob = line2[0:6]
        dob_check = line2[6]
        sex = line2[7]
        expiry = line2[8:14]
        expiry_check = line2[14]
        # nationality again at 15-17, optional data follows
        
        # Line 3 parsing (names)
        name_field = line3
        surname_end = name_field.find('<<')
        if surname_end == -1:
            surname = name_field.rstrip('<')
            given_names = ""
        else:
            surname = name_field[0:surname_end]
            given_names = name_field[surname_end+2:].replace('<', ' ').strip()
        
        surname = surname.replace('<', ' ').strip()
        
        # Validate
        is_valid, _ = validate_mrz(mrz_lines)
        
        return MRZData(
            document_type=doc_type,
            document_number=doc_number,
            document_number_check=doc_check,
            nationality=nationality,
            date_of_birth=dob,
            date_of_birth_check=dob_check,
            sex=sex,
            date_of_expiry=expiry,
            date_of_expiry_check=expiry_check,
            surname=surname,
            given_names=given_names,
            valid=is_valid,
        )
    except Exception as e:
        print(f"Error parsing MRZ: {e}")
        return None


def clean_mrz_text(text: str) -> str:
    """
    Clean OCR output to match MRZ character set.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text with only valid MRZ characters
    """
    # Valid MRZ characters: 0-9, A-Z, <
    valid_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
    
    # Convert to uppercase
    text = text.upper()
    
    # Replace common OCR errors
    replacements = {
        'O': '0',  # Letter O to digit 0 (in numeric fields)
        'I': '1',  # Letter I to digit 1
        'L': '1',  # Letter L to digit 1
        'S': '5',  # Letter S to digit 5
        'B': '8',  # Letter B to digit 8
        ' ': '<',  # Space to filler
    }
    
    cleaned = ""
    for char in text:
        if char in valid_chars:
            cleaned += char
        elif char in replacements:
            cleaned += replacements[char]
        # Skip invalid characters
    
    return cleaned


def format_mrz_output(mrz_data: MRZData) -> Dict:
    """Format MRZ data for API output."""
    return {
        "document_type": mrz_data.document_type,
        "document_number": mrz_data.document_number,
        "nationality": mrz_data.nationality,
        "date_of_birth": {
            "raw": mrz_data.date_of_birth,
            "formatted": f"{mrz_data.date_of_birth[0:2]}/{mrz_data.date_of_birth[2:4]}/{mrz_data.date_of_birth[4:6]}",
        },
        "sex": mrz_data.sex,
        "date_of_expiry": {
            "raw": mrz_data.date_of_expiry,
            "formatted": f"{mrz_data.date_of_expiry[0:2]}/{mrz_data.date_of_expiry[2:4]}/{mrz_data.date_of_expiry[4:6]}",
        },
        "surname": mrz_data.surname,
        "given_names": mrz_data.given_names,
        "valid": mrz_data.valid,
    }
