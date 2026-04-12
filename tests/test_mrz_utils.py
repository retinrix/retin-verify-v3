"""Tests for MRZ utilities."""

import pytest
from src.utils.mrz_utils import (
    compute_check_digit,
    validate_mrz,
    parse_mrz,
    clean_mrz_text,
)


def test_compute_check_digit():
    """Test check digit computation."""
    # Test cases from ICAO 9303
    assert compute_check_digit("115452189") == 8
    assert compute_check_digit("670317") == 4
    assert compute_check_digit("290823") == 6
    assert compute_check_digit("004031") == 1
    
    # Test with filler characters
    assert compute_check_digit("BOUTIGHANE") == 0  # < treated as 0


def test_validate_mrz_valid():
    """Test MRZ validation with valid data."""
    # Valid MRZ for Algerian ID
    line1 = "IDDZA1154521568<<<<<<<<<<<<<<<"
    line2 = "6703174M2908236DZA<<<<<<<<<<<8"
    line3 = "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
    
    is_valid, validation = validate_mrz([line1, line2, line3])
    
    assert is_valid
    assert validation["document_number_valid"]
    assert validation["date_of_birth_valid"]
    assert validation["date_of_expiry_valid"]


def test_validate_mrz_invalid():
    """Test MRZ validation with invalid data."""
    # Invalid MRZ (wrong check digit)
    line1 = "IDDZA1154521569<<<<<<<<<<<<<<<"  # Wrong check digit
    line2 = "6703174M2908236DZA<<<<<<<<<<<8"
    line3 = "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
    
    is_valid, validation = validate_mrz([line1, line2, line3])
    
    assert not is_valid
    assert not validation["document_number_valid"]


def test_parse_mrz():
    """Test MRZ parsing."""
    line1 = "IDDZA1154521568<<<<<<<<<<<<<<<"
    line2 = "6703174M2908236DZA<<<<<<<<<<<8"
    line3 = "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
    
    mrz_data = parse_mrz([line1, line2, line3])
    
    assert mrz_data is not None
    assert mrz_data.document_type == "ID"
    assert mrz_data.nationality == "DZA"
    assert mrz_data.document_number == "115452156"
    assert mrz_data.date_of_birth == "670317"
    assert mrz_data.sex == "M"
    assert mrz_data.date_of_expiry == "290823"
    assert mrz_data.surname == "BOUTIGHANE"
    assert mrz_data.given_names == "MOHAMED NAAMAN"


def test_clean_mrz_text():
    """Test MRZ text cleaning."""
    # Test basic cleaning
    assert clean_mrz_text("abc123") == "ABC123"
    
    # Test OCR error correction
    assert clean_mrz_text("O") == "0"  # Letter O to digit 0
    assert clean_mrz_text("I") == "1"  # Letter I to digit 1
    
    # Test space to filler
    assert clean_mrz_text("A B") == "A<B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
