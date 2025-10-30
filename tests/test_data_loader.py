"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import apply_filters, get_filter_options


def create_sample_dataframe():
    """Create a sample dataframe for testing"""
    data = {
        'case_id': ['BUG-001', 'BUG-001', 'BUG-002', 'BUG-002', 'BUG-003'],
        'activity': ['Open', 'Close', 'Open', 'Fix', 'Open'],
        'timestamp': pd.to_datetime([
            '2024-01-01 09:00:00',
            '2024-01-01 17:00:00',
            '2024-01-02 10:00:00',
            '2024-01-02 14:00:00',
            '2024-01-03 08:00:00'
        ]),
        'category': ['Backend', 'Backend', 'Frontend', 'Frontend', 'Database'],
        'priority': ['High', 'High', 'Medium', 'Medium', 'Critical'],
        'severity': ['Major', 'Major', 'Minor', 'Minor', 'Blocker']
    }
    return pd.DataFrame(data)


def test_apply_filters_category():
    """Test category filtering"""
    df = create_sample_dataframe()
    
    # Filter by Backend
    filtered = apply_filters(df, categories=['Backend'])
    assert len(filtered) == 2
    assert all(filtered['category'] == 'Backend')


def test_apply_filters_priority():
    """Test priority filtering"""
    df = create_sample_dataframe()
    
    # Filter by High priority
    filtered = apply_filters(df, priorities=['High'])
    assert len(filtered) == 2
    assert all(filtered['priority'] == 'High')


def test_apply_filters_multiple():
    """Test multiple filters"""
    df = create_sample_dataframe()
    
    # Filter by category and priority
    filtered = apply_filters(
        df,
        categories=['Backend', 'Frontend'],
        priorities=['High', 'Medium']
    )
    assert len(filtered) == 4


def test_apply_filters_date_range():
    """Test date range filtering"""
    df = create_sample_dataframe()
    
    # Filter by date range
    start_date = pd.to_datetime('2024-01-01').date()
    end_date = pd.to_datetime('2024-01-02').date()
    
    filtered = apply_filters(df, date_range=(start_date, end_date))
    assert len(filtered) == 4


def test_get_filter_options():
    """Test getting filter options"""
    df = create_sample_dataframe()
    
    options = get_filter_options(df)
    
    assert 'categories' in options
    assert 'priorities' in options
    assert 'severities' in options
    assert 'min_date' in options
    assert 'max_date' in options
    
    assert len(options['categories']) == 3
    assert len(options['priorities']) == 3
    assert len(options['severities']) == 3


def test_empty_dataframe():
    """Test with empty dataframe"""
    df = pd.DataFrame(columns=['case_id', 'activity', 'timestamp', 'category', 'priority', 'severity'])
    
    filtered = apply_filters(df, categories=['Backend'])
    assert len(filtered) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

