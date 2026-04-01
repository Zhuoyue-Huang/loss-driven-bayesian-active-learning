import numpy as np

def create_named_weight_fn(fn, identifier, display_name=None):
    """
    Wrapper to add name attributes to weight functions.
    
    Args:
        fn: The weight function
        identifier: Short name for filenames (no spaces, special chars)
        display_name: Human-readable name for titles and legends (optional)
    
    Returns:
        Function with name and display_name attributes
    """
    fn.name = identifier
    fn.display_name = display_name or identifier
    return fn

def get_weight_identifier(weight):
    """
    Get a consistent identifier for weights (both arrays and functions).
    
    Args:
        weight: Either np.ndarray (classification) or callable (regression)
    
    Returns:
        str: Identifier for filenames (filename-safe)
    """
    if isinstance(weight, np.ndarray):
        # Classification case: join array values with underscores
        return '_'.join(map(str, weight))
    elif callable(weight):
        # Regression case: use function name or custom attribute
        if hasattr(weight, 'name'):
            return weight.name
        elif hasattr(weight, '__name__'):
            # Clean up lambda function names
            name = weight.__name__
            if name == '<lambda>':
                return 'custom_weight'
            return name.replace(' ', '_').replace('(', '').replace(')', '')
        else:
            return 'custom_weight'
    else:
        return str(weight).replace(' ', '_').replace('.', 'p')

def get_weight_display_name(weight):
    """
    Get a human-readable name for weights.
    
    Args:
        weight: Either np.ndarray (classification) or callable (regression)
    
    Returns:
        str: Display name for titles and legends
    """
    if isinstance(weight, np.ndarray):
        return f"w = {weight.tolist()}"
    elif callable(weight):
        if hasattr(weight, 'display_name'):
            return weight.display_name
        elif hasattr(weight, 'name'):
            return f"Weight: {weight.name}"
        elif hasattr(weight, '__name__'):
            name = weight.__name__
            if name == '<lambda>':
                return "Custom Weight Function"
            return f"Weight: {name}"
        else:
            return "Custom Weight Function"
    else:
        return f"Weight: {weight}"
