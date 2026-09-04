"""Dataset Splitting Utilities."""

from sklearn.model_selection import train_test_split

def split_data(features, target, test_size=0.2, random_state=42):
    """ Split Features And Target Into Training And Testing Sets."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


