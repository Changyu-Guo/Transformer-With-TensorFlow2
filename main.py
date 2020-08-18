# -*- coding: utf - 8 -*-


def _create_min_max_boundaries(max_length,
                               min_boundary,
                               boundary_scale):
    """Create min and max boundary lists up to max_length.
    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
      buckets_min = [0, 4, 8, 16, 24]
      buckets_max = [4, 8, 16, 24, 25]
    Args:
      max_length: The maximum length of example in dataset.
      min_boundary: Minimum length in boundary.
      boundary_scale: Amount to scale consecutive boundaries in the list.
    Returns:
      min and max boundary lists
    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    # Create min and max boundary lists from the initial list.
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


print(_create_min_max_boundaries(24, 4, 1.5))